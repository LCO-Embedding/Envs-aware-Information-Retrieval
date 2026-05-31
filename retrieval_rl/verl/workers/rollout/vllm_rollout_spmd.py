# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from contextlib import contextmanager
from typing import Any, Optional, Union

import numpy as np
import torch
import torch.distributed
from tensordict import TensorDict
from transformers import PreTrainedTokenizer, ProcessorMixin
from vllm import LLM, RequestOutput, SamplingParams

from ...protocol import DataProto
from ...utils import torch_functional as VF
from ...utils.dataset import process_image, process_video
from ...utils.torch_dtypes import PrecisionType
from .base import BaseRollout
from .config import RolloutConfig

import re
from typing import Any

import time
import heapq
import importlib
import logging
import requests
import math
import json

def _repeat_interleave(value: Union[torch.Tensor, np.ndarray], repeats: int) -> Union[torch.Tensor, np.ndarray]:
    # repeat the elements, supports both tensor and numpy array
    if isinstance(value, torch.Tensor):
        return value.repeat_interleave(repeats, dim=0)
    else:
        return np.repeat(value, repeats, axis=0)


def _get_logit_bias(processor: Optional[ProcessorMixin]) -> Optional[dict[int, float]]:
    # enforce vllm to not output image token
    # TODO: add video token
    if processor is not None and hasattr(processor, "image_token"):
        image_token_id = processor.tokenizer.convert_tokens_to_ids(processor.image_token)
        return {image_token_id: -100}
    else:
        return None


def _process_multi_modal_data(
    multi_modal_data: dict[str, Any], min_pixels: int, max_pixels: int, video_fps: float
) -> dict[str, Any]:
    # may convert image path to image object
    images, videos = [], []
    if "images" in multi_modal_data:
        for image in multi_modal_data["images"]:
            images.append(process_image(image, min_pixels, max_pixels))

    if "videos" in multi_modal_data:
        for video in multi_modal_data["videos"]:
            videos.append(process_video(video, min_pixels, max_pixels, video_fps))

    if len(images) != 0:
        return {"image": images}

    if len(videos) != 0:
        return {"video": videos}

    return None


class vLLMRollout(BaseRollout):
    def __init__(
        self,
        model_path: str,
        config: RolloutConfig,
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
    ):
        """A vLLM rollout. It requires the module is supported by the vllm.

        Args:
            module: module here follows huggingface APIs
            config: DictConfig
            tokenizer: the task/model tokenizer
        """
        super().__init__()
        self.rank = int(os.getenv("RANK", "0"))
        self.config = config
        self.pad_token_id = tokenizer.pad_token_id
        self.use_tqdm = (self.rank == 0) and (not config.disable_tqdm)
        if config.tensor_parallel_size > torch.distributed.get_world_size():
            raise ValueError("Tensor parallelism size should be less than world size.")

        if config.max_num_batched_tokens < config.prompt_length + config.response_length:
            raise ValueError("max_num_batched_tokens should be greater than prompt_length + response_length.")

        engine_kwargs = {}
        if processor is not None:  # only VLMs have processor
            engine_kwargs["disable_mm_preprocessor_cache"] = True
            if config.limit_images:
                engine_kwargs["limit_mm_per_prompt"] = {"image": config.limit_images}

        self.inference_engine = LLM(
            model=model_path,
            skip_tokenizer_init=False,
            trust_remote_code=config.trust_remote_code,
            load_format="dummy",
            dtype=PrecisionType.to_str(PrecisionType.to_dtype(config.dtype)),
            seed=config.seed,
            max_model_len=config.max_model_len or config.prompt_length + config.response_length,
            distributed_executor_backend="external_launcher",
            tensor_parallel_size=config.tensor_parallel_size,
            gpu_memory_utilization=config.gpu_memory_utilization,
            max_num_batched_tokens=config.max_num_batched_tokens,
            disable_log_stats=config.disable_log_stats,
            enforce_eager=config.enforce_eager,
            disable_custom_all_reduce=True,
            enable_chunked_prefill=config.enable_chunked_prefill,
            enable_sleep_mode=True,
            **engine_kwargs,
        )

        # Offload vllm model to reduce peak memory usage
        self.inference_engine.sleep(level=1)

        sampling_kwargs = {
            "max_tokens": config.response_length,
            "detokenize": False,
            "logit_bias": _get_logit_bias(processor),
        }
        sampling_kwargs_turn2 = {
            "max_tokens": config.response_length,
            "detokenize": False,
            "logit_bias": _get_logit_bias(processor),
        }
        default_sampling_params = SamplingParams()
        for key in config.to_dict().keys():
            if hasattr(default_sampling_params, key):
                #sampling_kwargs[key] = getattr(config, key)
                if key == 'n':
                    sampling_kwargs[key] = int(getattr(config, key)/4)
                else:
                    sampling_kwargs[key] = getattr(config, key)
                sampling_kwargs_turn2[key] = getattr(config, key)

        print(f"Sampling params: {sampling_kwargs}.")
        print(f"Sampling params in turn2: {sampling_kwargs_turn2}.")
        self.sampling_params = SamplingParams(**sampling_kwargs)
        self.sampling_params_turn2 = SamplingParams(**sampling_kwargs_turn2)

        print("Start loading retrieval related thing on rollout!!!")
        #retrieval based init
        self.tokenizer = tokenizer
        self.retrieval_model = config.retrieval_model_path
        self.retrieval_batch_size = config.retrieval_batch_size
        self.retrieval_corpus_emb = torch.load(config.retrieval_corpus_emb_path) #torch.save(tensor, 'tensor.pt')
        self.corpus_length = self.retrieval_corpus_emb.size()[0]
        self.retrieval_score_func = config.retrieval_score_func
        self.retrieval_topk = config.retrieval_topk
        with open(config.retrieval_lookup_table_path, 'r') as f:
            self.retrieval_lookup_table = json.load(f) #{corpus_id: corpus_embedding_row_index}
        with open(config.retrieval_corpus_doc, 'r') as f:
            self.did2doc = json.load(f) #{corpus_id: corpus_embedding_row_index}
        print(f"Finish loading retrieval related thing, the corpus length is {self.corpus_length}, the score_func is {self.retrieval_score_func}, the topk is {self.retrieval_topk}")

    @contextmanager
    def update_sampling_params(self, **kwargs):
        # update sampling params
        old_sampling_params_args = {}
        if kwargs:
            for key, value in kwargs.items():
                if hasattr(self.sampling_params, key):
                    old_value = getattr(self.sampling_params, key)
                    old_sampling_params_args[key] = old_value
                    setattr(self.sampling_params, key, value)

        yield
        # roll back to previous sampling params
        for key, value in old_sampling_params_args.items():
            setattr(self.sampling_params, key, value)


    def call_server(self, address, texts_to_encode):
        payload = {"texts": texts_to_encode}
        for i in range(10):
            try:
                response = requests.post(address, json=payload)
                response.raise_for_status()
                result = response.json()
                embeddings = result["embeddings"]

                #print(f"Successfully received {len(embeddings)} embeddings.")
                #print(f"per embedding size is {len(embeddings[1])}.")
                #print(embeddings[1][:4])
                # print("First embedding:", embeddings[0])
                break
            except requests.exceptions.RequestException as e:
                print(f"An error occurred when calling server for embedding: {e}")
                time.sleep(1)
                print(f"Retry {i} time!")
        return embeddings

    def cos_sim(self, a: torch.Tensor, b: torch.Tensor):
        """
        Computes the cosine similarity cos_sim(a[i], b[j]) for all i and j.
        :return: Matrix with res[i][j]  = cos_sim(a[i], b[j])
        """
        if not isinstance(a, torch.Tensor):
            a = torch.tensor(a)

        if not isinstance(b, torch.Tensor):
            b = torch.tensor(b)

        if len(a.shape) == 1:
            a = a.unsqueeze(0)

        if len(b.shape) == 1:
            b = b.unsqueeze(0)

        a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
        b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
        return torch.mm(a_norm, b_norm.transpose(0, 1))  # TODO: this keeps allocating GPU memory

    def dot_score(self, a: torch.Tensor, b: torch.Tensor):
        """
        Computes the dot-product dot_prod(a[i], b[j]) for all i and j.
        :return: Matrix with res[i][j]  = dot_prod(a[i], b[j])
        """
        if not isinstance(a, torch.Tensor):
            a = torch.tensor(a)

        if not isinstance(b, torch.Tensor):
            b = torch.tensor(b)

        if len(a.shape) == 1:
            a = a.unsqueeze(0)

        if len(b.shape) == 1:
            b = b.unsqueeze(0)

        return torch.mm(a, b.transpose(0, 1))

    def retrieval(self, model_address, queries, corpus, score_function="cos_sim", top_k=10, retrieval_batch_size = 1000):
        """
        Sends queries to the retrieval server and gets back top_k document indices and scores.
        Currently, the corpus is hosted and searched on the server side.
        """

        payload = {
            "texts": queries,
            "top_k": top_k,
            "score_function": score_function,
            "corpus_name": "ragbench"
        }

        start_time = time.perf_counter()

        response = requests.post(model_address, json=payload)
        response.raise_for_status()
        data = response.json()
        
        indices = data["indices"] # List[List[int]]
        scores = data["scores"]   # List[List[float]]

        end_time = time.perf_counter()
        
        result_heaps = {}
        
        for i, (doc_ids, doc_scores) in enumerate(zip(indices, scores)):
            query_results = []
            for doc_id, score in zip(doc_ids, doc_scores):
                if score_function == "cos_sim":
                    if score < 0.999: 
                        query_results.append((score, doc_id))
                else:
                    if score != 1:
                        query_results.append((score, doc_id))
            
            result_heaps[i] = query_results

        return result_heaps


    def merge_retrieval_result(self,lst):
        # merge with dict
        result_dict = {}
        for num, ch in lst:
            if ch not in result_dict or num > result_dict[ch]:
                result_dict[ch] = num

        # dict to list
        filtered_lst = [(num, ch) for ch, num in result_dict.items()]

        # ranking
        filtered_lst.sort(key=lambda x: x[0], reverse=True)

        return filtered_lst
    

    def compute_score(self, reward_inputs: list[dict[str, Any]], model_address, corpus, score_function, retrieval_lookup_table, top_k=10, retrieval_batch_size = 1000, format_weight: float = 0.1):
        if not isinstance(reward_inputs, list):
            raise ValueError("Please use `reward_type=batch` for math reward function.")

        query2rewrite = {}
        rewrite_queries = []
        for idx, reward_input in enumerate(reward_inputs):
            #response = re.sub(r"\s*(<|>|/)\s*", r"\1", reward_input["response"])  # handle qwen2.5vl-32b format
            response = reward_input["response"]

            matches_in_rewrite = re.findall(r"<rewrite>([^<]*)</rewrite>", response, re.DOTALL)
            if matches_in_rewrite != []:
                matches = matches_in_rewrite
                query2rewrite[idx] = [x+len(rewrite_queries) for x in range(len(matches))]
                rewrite_queries = rewrite_queries + matches
            else:
                matches = [response.split('/n')[-1]]
                query2rewrite[idx] = [x+len(rewrite_queries) for x in range(len(matches))]
                rewrite_queries = rewrite_queries + matches
        
        retrieval_result = self.retrieval(model_address, rewrite_queries, corpus, score_function, top_k, retrieval_batch_size)
        final_result = []
        for idx, reward_input in enumerate(reward_inputs):
            rewrite_ids = query2rewrite[idx]
            one_query_retrieval_result = []
            for x in rewrite_ids:
                one_query_retrieval_result = one_query_retrieval_result + retrieval_result[x]

            one_query_retrieval_result = self.merge_retrieval_result(one_query_retrieval_result)
            one_query_retrieval_result = one_query_retrieval_result[:top_k]
            one_query_retrieval_result = [x[1] for x in one_query_retrieval_result]
            one_query_retrieval_result = [self.did2doc[str(x)] for x in one_query_retrieval_result]
            final_result.append(one_query_retrieval_result)
        return final_result

    def compute_retrieval(self, response_ids):
        reward_inputs = []
        #response_ids = data.batch["responses"]

        for i in range(len(response_ids)):
            one_response_id = response_ids[i]
            response_str = self.tokenizer.decode(
                one_response_id, skip_special_tokens=True
            )
            reward_inputs.append(
                {
                    "response": response_str
                }
            )
            
        retrieval_result = self.compute_score(reward_inputs, self.retrieval_model, self.retrieval_corpus_emb, self.retrieval_score_func, self.retrieval_lookup_table, self.retrieval_topk, self.retrieval_batch_size)

        return retrieval_result


    @torch.no_grad()
    def generate_sequences_twoturn_balanced(self, prompts: DataProto) -> DataProto:
        # left-padded attention_mask
        input_ids: torch.Tensor = prompts.batch["input_ids"]  # (bs, prompt_length)
        attention_mask: torch.Tensor = prompts.batch["attention_mask"]
        position_ids: torch.Tensor = prompts.batch["position_ids"]
        eos_token_id: int = prompts.meta_info["eos_token_id"]
        batch_size = input_ids.size(0)

        non_tensor_batch = prompts.non_tensor_batch
        batch_raw_prompt_ids = non_tensor_batch.pop("raw_prompt_ids")
        batch_multi_modal_data = non_tensor_batch.pop("multi_modal_data", None)
        if batch_size != len(batch_raw_prompt_ids):
            raise RuntimeError("vllm sharding manager is not work properly.")

        if batch_multi_modal_data is not None:
            vllm_inputs = []
            for raw_prompt_ids, multi_modal_data in zip(batch_raw_prompt_ids, batch_multi_modal_data):
                vllm_inputs.append(
                    {
                        "prompt_token_ids": list(raw_prompt_ids),
                        "multi_modal_data": _process_multi_modal_data(
                            multi_modal_data,
                            prompts.meta_info["min_pixels"],
                            prompts.meta_info["max_pixels"],
                            prompts.meta_info["video_fps"],
                        ),
                    }
                )
        else:
            vllm_inputs = [{"prompt_token_ids": list(raw_prompt_ids)} for raw_prompt_ids in batch_raw_prompt_ids]

        # inferencing for turn 1！！！！！
        with self.update_sampling_params(**prompts.meta_info):
            completions_turn1: list[RequestOutput] = self.inference_engine.generate(
                prompts=vllm_inputs, sampling_params=self.sampling_params, use_tqdm=self.use_tqdm
            )
            response_ids_turn1 = [output.token_ids for completion in completions_turn1 for output in completion.outputs]

        #obtain retrieval result after turn 1 and prepare prompt
        retrieval_result = self.compute_retrieval(response_ids_turn1) #[[d1,d2...],[d2,d3...],..,[d4,d5...]], n x topk
        retrieval_prompt_id = []
        prompt_for_turn2 = '\n Please further improve the rewritten query based on the retrieval result above. Generate a new rewritten query enclosing with separate <rewrite_2> and </rewrite_2>. Output can look like: <rewrite_2> the new rewritten query </rewrite_2>.'
        for doc_list in retrieval_result:
            one_retrieval_prompt = '\nuser\nThe following are the top documents (only show the first part of each document) retrieved by the rewritten query, ordered from high relevance to low relevance:'

            doc_list_id_truncated = []
            for idx, doc in enumerate(doc_list):
                #doc_list_id_truncated += self.tokenizer.encode(f'\n <doc{idx}> + {doc} <\doc{idx}>')
                doc_list_id_truncated += self.tokenizer.encode(f'\n <doc{idx+1}>') + self.tokenizer.encode(doc)[:100] + self.tokenizer.encode(f'<\doc{idx+1}>')

            retrieval_prompt_id.append(self.tokenizer.encode(one_retrieval_prompt)+doc_list_id_truncated+self.tokenizer.encode(prompt_for_turn2)+[151645, 198, 151644, 77091, 198])

        #retrieval_prompt_id = [self.tokenizer.encode(x)[:150] for x in retrieval_prompt]
            
        repeated_vllm_inputs = []
        repeat_times = int(len(response_ids_turn1)/len(vllm_inputs))
        for i in vllm_inputs:
            for j in range(repeat_times):
                repeated_vllm_inputs.append(i)


        assert len(repeated_vllm_inputs) == len(response_ids_turn1), f"repeated_vllm_inputs: {len(repeated_vllm_inputs)},response_ids_turn1:{len(response_ids_turn1)}"
        assert len(repeated_vllm_inputs) == len(retrieval_prompt_id), f"repeated_vllm_inputs: {len(repeated_vllm_inputs)},retrieval_prompt_id:{len(retrieval_prompt_id)}"

        vllm_inputs_turn_2 = []
        for idx, one_inputx in enumerate(response_ids_turn1):

            one_input_turn_2 = repeated_vllm_inputs[idx]['prompt_token_ids'] + list(response_ids_turn1[idx]) + retrieval_prompt_id[idx]
            one_input_turn_2 = one_input_turn_2[:5120]
            vllm_inputs_turn_2.append({"prompt_token_ids": one_input_turn_2})

        # inferencing for turn 2！！！！！
        # users can customize different sampling_params at different run
        self.sampling_params_turn2.n = repeat_times
        with self.update_sampling_params(**prompts.meta_info):
            completions_turn2: list[RequestOutput] = self.inference_engine.generate(
                prompts=vllm_inputs_turn_2, sampling_params=self.sampling_params_turn2, use_tqdm=self.use_tqdm
            )
            response_ids_turn2 = [output.token_ids for completion in completions_turn2 for output in completion.outputs]
            #merge response in turn 1 and turn 2 and mid prompt

            repeated_response_ids_turn1= []
            repeat_times2 = int(len(response_ids_turn2)/len(response_ids_turn1))
            for i in response_ids_turn1:
                for j in range(repeat_times2):
                    repeated_response_ids_turn1.append(i)
            response_ids_turn1 = repeated_response_ids_turn1

            repeated_retrieval_prompt_id= []
            repeat_times2 = int(len(response_ids_turn2)/len(retrieval_prompt_id))
            for i in retrieval_prompt_id:
                for j in range(repeat_times2):
                    repeated_retrieval_prompt_id.append(i)
            retrieval_prompt_id = repeated_retrieval_prompt_id


            assert len(response_ids_turn2) == len(response_ids_turn1)
            assert len(response_ids_turn2) == len(retrieval_prompt_id)
            response_ids = []
            for idx, one_response_ids_turn2 in enumerate(response_ids_turn2):
                one_response_ids = list(response_ids_turn1[idx]) + retrieval_prompt_id[idx] + list(response_ids_turn2[idx])
                one_response_ids = one_response_ids[:self.config.response_length]
                one_response_ids = tuple(one_response_ids)
                response_ids.append(one_response_ids)

            response_ids = VF.pad_2d_list_to_length(
                response_ids, self.pad_token_id, max_length=self.config.response_length
            ).to(input_ids.device)

            if self.sampling_params.n > 1:
                batch_size = batch_size * self.sampling_params.n * repeat_times2
                input_ids = _repeat_interleave(input_ids, self.sampling_params.n * repeat_times2)
                attention_mask = _repeat_interleave(attention_mask, self.sampling_params.n * repeat_times2)
                position_ids = _repeat_interleave(position_ids, self.sampling_params.n * repeat_times2)
                if batch_multi_modal_data is not None:
                    batch_multi_modal_data = _repeat_interleave(batch_multi_modal_data, self.sampling_params.n * repeat_times2)


        sequence_ids = torch.cat([input_ids, response_ids], dim=-1)
        response_length = response_ids.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.view(1, -1).expand(batch_size, -1)
        if position_ids.dim() == 3:  # qwen2vl mrope
            delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(batch_size, 3, -1)

        # prompt: left pad + response: right pad
        # attention_mask: [0,0,0,0,1,1,1,1 | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3 | 4,5,6,7,8,9,10,11]
        response_position_ids = position_ids[..., -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        response_mask = VF.get_response_mask(
            response_ids=response_ids, eos_token_id=eos_token_id, dtype=attention_mask.dtype
        )
        # set mask for mid prompt to 0
        for idx in range(len(response_mask)):
            response_mask[idx][len(response_ids_turn1[idx]):len(response_ids_turn1[idx])+len(retrieval_prompt_id[idx])] = 0
            response_mask[idx][len(response_ids_turn1[idx])+len(retrieval_prompt_id[idx]):len(response_ids_turn1[idx])+len(retrieval_prompt_id[idx])+len(response_ids_turn2[idx])+3] = 1

        attention_mask = torch.cat((attention_mask, response_mask), dim=-1)

        # all the tp ranks should contain the same data here. data in all ranks are valid
        batch = TensorDict(
            {
                "prompts": input_ids,
                "responses": response_ids,
                "input_ids": sequence_ids,  # here input_ids become the whole sentences
                "attention_mask": attention_mask,
                "response_mask": response_mask,
                "position_ids": position_ids,
            },
            batch_size=batch_size,
        )
        if batch_multi_modal_data is not None:
            non_tensor_batch = {"multi_modal_data": batch_multi_modal_data}
        else:
            non_tensor_batch = {}

        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch, meta_info=prompts.meta_info)


    @torch.no_grad()
    def generate_sequences_twoturn(self, prompts: DataProto) -> DataProto:
        # left-padded attention_mask
        input_ids: torch.Tensor = prompts.batch["input_ids"]  # (bs, prompt_length)
        attention_mask: torch.Tensor = prompts.batch["attention_mask"]
        position_ids: torch.Tensor = prompts.batch["position_ids"]
        eos_token_id: int = prompts.meta_info["eos_token_id"]
        batch_size = input_ids.size(0)

        non_tensor_batch = prompts.non_tensor_batch
        batch_raw_prompt_ids = non_tensor_batch.pop("raw_prompt_ids")
        batch_multi_modal_data = non_tensor_batch.pop("multi_modal_data", None)
        if batch_size != len(batch_raw_prompt_ids):
            raise RuntimeError("vllm sharding manager is not work properly.")

        if batch_multi_modal_data is not None:
            vllm_inputs = []
            for raw_prompt_ids, multi_modal_data in zip(batch_raw_prompt_ids, batch_multi_modal_data):
                vllm_inputs.append(
                    {
                        "prompt_token_ids": list(raw_prompt_ids),
                        "multi_modal_data": _process_multi_modal_data(
                            multi_modal_data,
                            prompts.meta_info["min_pixels"],
                            prompts.meta_info["max_pixels"],
                            prompts.meta_info["video_fps"],
                        ),
                    }
                )
        else:
            vllm_inputs = [{"prompt_token_ids": list(raw_prompt_ids)} for raw_prompt_ids in batch_raw_prompt_ids]

        # inferencing for turn 1！！！！！
        with self.update_sampling_params(**prompts.meta_info):
            completions_turn1: list[RequestOutput] = self.inference_engine.generate(
                prompts=vllm_inputs, sampling_params=self.sampling_params, use_tqdm=self.use_tqdm
            )
            response_ids_turn1 = [output.token_ids for completion in completions_turn1 for output in completion.outputs]

        #obtain retrieval result after turn 1 and prepare prompt
        retrieval_result = self.compute_retrieval(response_ids_turn1) #[[d1,d2...],[d2,d3...],..,[d4,d5...]], n x topk
        retrieval_prompt_id = []
        prompt_for_turn2 = '\n Please further improve the rewritten query based on the retrieval result above. Generate a new rewritten query enclosing with separate <rewrite_2> and </rewrite_2>. Output can look like: <rewrite_2> the new rewritten query </rewrite_2>.'
        for doc_list in retrieval_result:
            one_retrieval_prompt = '\nuser\nThe following are the top documents (only show the first part of each document) retrieved by the rewritten query, ordered from high relevance to low relevance:'

            doc_list_id_truncated = []
            for idx, doc in enumerate(doc_list):
                #doc_list_id_truncated += self.tokenizer.encode(f'\n <doc{idx}> + {doc} <\doc{idx}>')
                doc_list_id_truncated += self.tokenizer.encode(f'\n <doc{idx+1}>') + self.tokenizer.encode(doc)[:100] + self.tokenizer.encode(f'<\doc{idx+1}>')

            retrieval_prompt_id.append(self.tokenizer.encode(one_retrieval_prompt)+doc_list_id_truncated+self.tokenizer.encode(prompt_for_turn2)+[151645, 198, 151644, 77091, 198])

        #retrieval_prompt_id = [self.tokenizer.encode(x)[:150] for x in retrieval_prompt]
            
        repeated_vllm_inputs = []
        repeat_times = int(len(response_ids_turn1)/len(vllm_inputs))
        for i in vllm_inputs:
            for j in range(repeat_times):
                repeated_vllm_inputs.append(i)


        assert len(repeated_vllm_inputs) == len(response_ids_turn1), f"repeated_vllm_inputs: {len(repeated_vllm_inputs)},response_ids_turn1:{len(response_ids_turn1)}"
        assert len(repeated_vllm_inputs) == len(retrieval_prompt_id), f"repeated_vllm_inputs: {len(repeated_vllm_inputs)},retrieval_prompt_id:{len(retrieval_prompt_id)}"

        vllm_inputs_turn_2 = []
        for idx, one_inputx in enumerate(response_ids_turn1):

            one_input_turn_2 = repeated_vllm_inputs[idx]['prompt_token_ids'] + list(response_ids_turn1[idx]) + retrieval_prompt_id[idx]
            one_input_turn_2 = one_input_turn_2[:5120]
            vllm_inputs_turn_2.append({"prompt_token_ids": one_input_turn_2})

        # inferencing for turn 2！！！！！
        # users can customize different sampling_params at different run
        with self.update_sampling_params(**prompts.meta_info):
            completions_turn2: list[RequestOutput] = self.inference_engine.generate(
                prompts=vllm_inputs_turn_2, sampling_params=self.sampling_params_turn2, use_tqdm=self.use_tqdm
            )
            response_ids_turn2 = [output.token_ids for completion in completions_turn2 for output in completion.outputs]
            #merge response in turn 1 and turn 2 and mid prompt
            assert len(response_ids_turn2) == len(response_ids_turn1)
            assert len(response_ids_turn2) == len(retrieval_prompt_id)
            response_ids = []
            for idx, one_response_ids_turn2 in enumerate(response_ids_turn2):
                one_response_ids = list(response_ids_turn1[idx]) + retrieval_prompt_id[idx] + list(response_ids_turn2[idx])
                one_response_ids = one_response_ids[:self.config.response_length]
                one_response_ids = tuple(one_response_ids)
                response_ids.append(one_response_ids)

            response_ids = VF.pad_2d_list_to_length(
                response_ids, self.pad_token_id, max_length=self.config.response_length
            ).to(input_ids.device)

            if self.sampling_params.n > 1:
                batch_size = batch_size * self.sampling_params.n
                input_ids = _repeat_interleave(input_ids, self.sampling_params.n)
                attention_mask = _repeat_interleave(attention_mask, self.sampling_params.n)
                position_ids = _repeat_interleave(position_ids, self.sampling_params.n)
                if batch_multi_modal_data is not None:
                    batch_multi_modal_data = _repeat_interleave(batch_multi_modal_data, self.sampling_params.n)


        sequence_ids = torch.cat([input_ids, response_ids], dim=-1)
        response_length = response_ids.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.view(1, -1).expand(batch_size, -1)
        if position_ids.dim() == 3:  # qwen2vl mrope
            delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(batch_size, 3, -1)

        # prompt: left pad + response: right pad
        # attention_mask: [0,0,0,0,1,1,1,1 | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3 | 4,5,6,7,8,9,10,11]
        response_position_ids = position_ids[..., -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        response_mask = VF.get_response_mask(
            response_ids=response_ids, eos_token_id=eos_token_id, dtype=attention_mask.dtype
        )
        # set mask for mid prompt to 0
        for idx in range(len(response_mask)):
            response_mask[idx][len(response_ids_turn1[idx]):len(response_ids_turn1[idx])+len(retrieval_prompt_id[idx])] = 0
            response_mask[idx][len(response_ids_turn1[idx])+len(retrieval_prompt_id[idx]):len(response_ids_turn1[idx])+len(retrieval_prompt_id[idx])+len(response_ids_turn2[idx])] = 1

        attention_mask = torch.cat((attention_mask, response_mask), dim=-1)

        # all the tp ranks should contain the same data here. data in all ranks are valid
        batch = TensorDict(
            {
                "prompts": input_ids,
                "responses": response_ids,
                "input_ids": sequence_ids,  # here input_ids become the whole sentences
                "attention_mask": attention_mask,
                "response_mask": response_mask,
                "position_ids": position_ids,
            },
            batch_size=batch_size,
        )
        if batch_multi_modal_data is not None:
            non_tensor_batch = {"multi_modal_data": batch_multi_modal_data}
        else:
            non_tensor_batch = {}

        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch, meta_info=prompts.meta_info)


    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto) -> DataProto:
        # left-padded attention_mask
        input_ids: torch.Tensor = prompts.batch["input_ids"]  # (bs, prompt_length)
        attention_mask: torch.Tensor = prompts.batch["attention_mask"]
        position_ids: torch.Tensor = prompts.batch["position_ids"]
        eos_token_id: int = prompts.meta_info["eos_token_id"]
        batch_size = input_ids.size(0)

        non_tensor_batch = prompts.non_tensor_batch
        batch_raw_prompt_ids = non_tensor_batch.pop("raw_prompt_ids")
        batch_multi_modal_data = non_tensor_batch.pop("multi_modal_data", None)
        if batch_size != len(batch_raw_prompt_ids):
            raise RuntimeError("vllm sharding manager is not work properly.")

        if batch_multi_modal_data is not None:
            vllm_inputs = []
            for raw_prompt_ids, multi_modal_data in zip(batch_raw_prompt_ids, batch_multi_modal_data):
                vllm_inputs.append(
                    {
                        "prompt_token_ids": list(raw_prompt_ids),
                        "multi_modal_data": _process_multi_modal_data(
                            multi_modal_data,
                            prompts.meta_info["min_pixels"],
                            prompts.meta_info["max_pixels"],
                            prompts.meta_info["video_fps"],
                        ),
                    }
                )
        else:
            vllm_inputs = [{"prompt_token_ids": list(raw_prompt_ids)} for raw_prompt_ids in batch_raw_prompt_ids]

        # users can customize different sampling_params at different run
        with self.update_sampling_params(**prompts.meta_info):
            completions: list[RequestOutput] = self.inference_engine.generate(
                prompts=vllm_inputs, sampling_params=self.sampling_params, use_tqdm=self.use_tqdm
            )
            response_ids = [output.token_ids for completion in completions for output in completion.outputs]
            response_ids = VF.pad_2d_list_to_length(
                response_ids, self.pad_token_id, max_length=self.config.response_length
            ).to(input_ids.device)

            if self.sampling_params.n > 1:
                batch_size = batch_size * self.sampling_params.n
                input_ids = _repeat_interleave(input_ids, self.sampling_params.n)
                attention_mask = _repeat_interleave(attention_mask, self.sampling_params.n)
                position_ids = _repeat_interleave(position_ids, self.sampling_params.n)
                if batch_multi_modal_data is not None:
                    batch_multi_modal_data = _repeat_interleave(batch_multi_modal_data, self.sampling_params.n)

        sequence_ids = torch.cat([input_ids, response_ids], dim=-1)
        response_length = response_ids.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.view(1, -1).expand(batch_size, -1)
        if position_ids.dim() == 3:  # qwen2vl mrope
            delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(batch_size, 3, -1)

        # prompt: left pad + response: right pad
        # attention_mask: [0,0,0,0,1,1,1,1 | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3 | 4,5,6,7,8,9,10,11]
        response_position_ids = position_ids[..., -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        response_mask = VF.get_response_mask(
            response_ids=response_ids, eos_token_id=eos_token_id, dtype=attention_mask.dtype
        )
        attention_mask = torch.cat((attention_mask, response_mask), dim=-1)

        # all the tp ranks should contain the same data here. data in all ranks are valid
        batch = TensorDict(
            {
                "prompts": input_ids,
                "responses": response_ids,
                "input_ids": sequence_ids,  # here input_ids become the whole sentences
                "attention_mask": attention_mask,
                "response_mask": response_mask,
                "position_ids": position_ids,
            },
            batch_size=batch_size,
        )
        if batch_multi_modal_data is not None:
            non_tensor_batch = {"multi_modal_data": batch_multi_modal_data}
        else:
            non_tensor_batch = {}

        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch, meta_info=prompts.meta_info)