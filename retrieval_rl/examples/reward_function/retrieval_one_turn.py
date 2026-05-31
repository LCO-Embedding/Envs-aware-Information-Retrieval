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

import re
from typing import Any

from mathruler.grader import grade_answer
import torch

import time
import heapq
import importlib
import logging
import os
import requests
import math

def call_server(address, texts_to_encode):
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

def calculate_ndcg_at_10(retrieved_ids: list, ground_truth_ids: list) -> float:
    """
    计算 nDCG@10 分数。

    Args:
        retrieved_ids (list): 检索系统返回的前10个 document id 列表。
                              函数假设这个列表长度为10。
        ground_truth_ids (list): 相关的 "ground truth" document id 列表。
                                 为了提高查找效率，会被转换为集合(set)。

    Returns:
        float: nDCG@10 的分数，范围在 0.0 到 1.0 之间。
    """
    # 确保检索列表长度为10
    if len(retrieved_ids) > 10:
        retrieved_ids = retrieved_ids[:10]

    # 1. 计算 DCG@10 (Discounted Cumulative Gain)
    dcg = 0.0
    # 将 ground_truth 转换为 set 以获得 O(1) 的平均查找时间复杂度
    ground_truth_set = set(ground_truth_ids)

    for i, doc_id in enumerate(retrieved_ids):
        # 排名从1开始，所以是 i + 1
        rank = i + 1
        # 如果文档是相关的，则其相关性(gain)为1，否则为0
        if doc_id in ground_truth_set:
            # 累加增益，并通过对数进行折损
            # 公式: relevance / log2(rank + 1)
            # +1 是因为log2(1) = 0，我们要避免分母为0
            dcg += 1.0 / math.log2(rank + 1)

    # 2. 计算 IDCG@10 (Ideal Discounted Cumulative Gain)
    # IDCG 是理想排序下的DCG，即所有相关文档都排在最前面
    idcg = 0.0
    # 理想情况下的相关文档数量不能超过10，也不能超过ground truth的总数
    num_ideal_docs = min(len(ground_truth_ids), 10)
    
    for i in range(num_ideal_docs):
        # 排名从1开始
        rank = i + 1
        # 在理想排序中，前面的文档都是相关的，增益为1
        idcg += 1.0 / math.log2(rank + 1)

    # 3. 计算 nDCG@10
    # 如果没有相关的文档 (idcg=0)，则nDCG为0
    if idcg == 0:
        return 0.0
    
    ndcg = dcg / idcg
    return ndcg

def format_reward(response: str) -> float:
    pattern = re.compile(r"<think>.*?</think>\s*<rewrite>.*?</rewrite>", re.DOTALL)
    format_match = re.fullmatch(pattern, response)
    return 1.0 if format_match else 0.0


def accuracy_reward(retrieval_result, ground_truth, retrieval_lookup_table) -> float:
    ground_truth_corpus_id = [retrieval_lookup_table[x] for x in ground_truth]
    '''
    c=0
    for g in ground_truth_corpus_id:
        if g in retrieval_result:
            c+=1
    
    score = c/len(ground_truth_corpus_id)
    '''
    score = calculate_ndcg_at_10(retrieval_result, ground_truth_corpus_id)
    return score


def cos_sim(a: torch.Tensor, b: torch.Tensor):
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


def dot_score(a: torch.Tensor, b: torch.Tensor):
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


def retrieval(model_address, queries, corpus, score_function, top_k=10, retrieval_batch_size = 1000):
    query_ids = list(range(len(queries)))

    start_time = time.perf_counter()

    query_embeddings = call_server(model_address, queries)

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time

    itr = range(0, len(corpus), retrieval_batch_size)
    result_heaps = {qid: [] for qid in query_ids}

    for batch_num, corpus_start_idx in enumerate(itr):
        corpus_end_idx = min(corpus_start_idx + retrieval_batch_size, len(corpus))
        sub_corpus_embeddings = corpus[corpus_start_idx:corpus_end_idx, :]

        # Compute similarites using either cosine-similarity or dot product

        start_time = time.perf_counter()

        if score_function == "cos_sim":
            cos_scores = cos_sim(query_embeddings, sub_corpus_embeddings)
        if score_function == "dot_score":
            cos_scores = dot_score(query_embeddings, sub_corpus_embeddings)

        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        #print(f"retrieval计算执行时间: {elapsed_time:.4f} 秒")

        cos_scores[torch.isnan(cos_scores)] = -1

        # Get top-k values
        cos_scores_top_k_values, cos_scores_top_k_idx = torch.topk(
            cos_scores,
            min(top_k + 1, len(cos_scores[1])),
            dim=1,
            largest=True,
            sorted=True,
        )
        cos_scores_top_k_values = cos_scores_top_k_values.cpu().tolist()
        cos_scores_top_k_idx = cos_scores_top_k_idx.cpu().tolist()

        for query_itr in range(len(query_embeddings)):
            query_id = query_ids[query_itr]
            for sub_corpus_id, score in zip(cos_scores_top_k_idx[query_itr], cos_scores_top_k_values[query_itr]):
                corpus_id = corpus_start_idx + sub_corpus_id
                if score != 1:
                    if len(result_heaps[query_id]) < top_k:
                        # Push item on the heap
                        heapq.heappush(result_heaps[query_id], (score, corpus_id))
                    else:
                        # If item is larger than the smallest in the heap, push it on the heap then pop the smallest element
                        heapq.heappushpop(result_heaps[query_id], (score, corpus_id))

    return result_heaps


def merge_retrieval_result(lst):
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

def compute_score(reward_inputs: list[dict[str, Any]], model_address, corpus, score_function, retrieval_lookup_table, top_k=10, retrieval_batch_size = 1000, format_weight: float = 0.1) -> list[dict[str, float]]:
    if not isinstance(reward_inputs, list):
        raise ValueError("Please use `reward_type=batch` for math reward function.")

    scores = []
    query2rewrite = {}
    rewrite_queries = []
    for idx, reward_input in enumerate(reward_inputs):
        #response = re.sub(r"\s*(<|>|/)\s*", r"\1", reward_input["response"])  # handle qwen2.5vl-32b format
        response = reward_input["response"]
        format_score = format_reward(response)
        scores.append(
            {
                "format": format_score
            }
        )

        #print(f"the response is {response}!!!!!")

        if format_score == 1:
            matches_in_rewrite = re.findall(r"<rewrite>([^<]*)</rewrite>", response, re.DOTALL)
            matches = matches_in_rewrite
            query2rewrite[idx] = [x+len(rewrite_queries) for x in range(len(matches))]
            rewrite_queries = rewrite_queries + matches
        else:
            matches = ['']
            query2rewrite[idx] = [x+len(rewrite_queries) for x in range(len(matches))]
            rewrite_queries = rewrite_queries + matches
    
    retrieval_result = retrieval(model_address, rewrite_queries, corpus, score_function, top_k, retrieval_batch_size)
    for idx, reward_input in enumerate(reward_inputs):
        if scores[idx]["format"] == 0:
            scores[idx]["accuracy"] = 0
            scores[idx]["overall"] = 0
        else:
            rewrite_ids = query2rewrite[idx]
            one_query_retrieval_result = []
            for x in rewrite_ids:
                one_query_retrieval_result = one_query_retrieval_result + retrieval_result[x]

            one_query_retrieval_result = merge_retrieval_result(one_query_retrieval_result)
            one_query_retrieval_result = one_query_retrieval_result[:top_k]
            print(one_query_retrieval_result, [retrieval_lookup_table[x] for x in reward_input["ground_truth"]])
            one_query_retrieval_result = [x[1] for x in one_query_retrieval_result]

            accuracy_score = accuracy_reward(one_query_retrieval_result, reward_input["ground_truth"], retrieval_lookup_table)

            scores[idx]["accuracy"] = accuracy_score
            scores[idx]["overall"] = (1 - format_weight) * accuracy_score + format_weight * scores[idx]["format"]

    return scores
