pip install sympy
pip install transformers==4.54.0
pip install math_verify
pip install torchdata
pip install tenacity
pip install openai
pip install python-dotenv
pip install asyncio
pip install vllm==0.8.3
pip install swanlab

conda list

ray stop

export NCCL_IB_HCA=mlx5 
export NCCL_IB_TC=136 
export NCCL_IB_SL=5 
export NCCL_IB_GID_INDEX=3
export NCCL_DEBUG=INFO 
export NCCL_IB_TIMEOUT=3600
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600 
export NCCL_LAUNCH_MODE=PARALLEL 

cd repo_folder

export SWANLAB_API_KEY=xxx

# 显示所有环境变量
echo "Environment Variables:"
echo "  MASTER_ADDR: $MASTER_ADDR"
echo "  MASTER_PORT: $MASTER_PORT"
echo "  WORLD_SIZE: $WORLD_SIZE"
echo "  RANK: $RANK"
echo "  NPROC_PER_NODE: $NPROC_PER_NODE"

export HF_ENDPOINT=https://hf-mirror.com

MODEL_PATH=model_path # replace it with your local file path

SERVER_IP=xxx.xxx.xxx.xxx
SERVER_PORT=xxxx
api_url=http://${SERVER_IP}:${SERVER_PORT}/embed

ray start --head --port=6000

TRAINING_NODES=$WORLD_SIZE  # Equivalent to WORLD_SIZE
TRAINING_GPUS_PER_NODE=$NPROC_PER_NODE  # Equivalent to NPROC_PER_NODE

echo "Training Configuration:"
echo "  TRAINING_NODES: $TRAINING_NODES"
echo "  TRAINING_GPUS_PER_NODE: $TRAINING_GPUS_PER_NODE"

echo "Starting training on the Master node..."

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=train_data \
    data.val_files=val_data \
    data.seed=1 \
    data.rollout_batch_size=512 \
    data.format_prompt=./examples/format_prompt/retrieval.jinja \
    worker.actor.global_batch_size=128 \
    worker.actor.kl_coef=1.0e-3 \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.actor.micro_batch_size_per_device_for_update=4 \
    worker.actor.micro_batch_size_per_device_for_experience=8 \
    worker.reward.reward_function=./examples/reward_function/retrieval_one_turn.py:compute_score \
    worker.reward.retrieval_model_path=${api_url} \
    worker.reward.retrieval_corpus_emb_path=retrieval_corpus_emb_path.pt \
    worker.reward.retrieval_lookup_table_path=retrieval_lookup_table_path.json \
    worker.reward.retrieval_batch_size=10000 \
    worker.reward.retrieval_topk=10 \
    worker.reward.retrieval_score_func=dot_score \
    trainer.n_gpus_per_node=$TRAINING_GPUS_PER_NODE \
    trainer.nnodes=$TRAINING_NODES \
    trainer.project_name=project \
    trainer.experiment_name=experiment \
    trainer.save_checkpoint_path=save_path