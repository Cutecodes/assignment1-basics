uv run cs336_basics/train.py --data_path ~/llm_infra/cs336/data/TinyStoriesV2-GPT4-train.npy \
 --val_data_path ~/llm_infra/cs336/data/TinyStoriesV2-GPT4-valid.npy \
 --use_wandb \
 --wandb_run_name "batch-size-1" \
 --batch_size 1 \
 --total_steps 5000 \
 --context_length 256 
uv run cs336_basics/train.py --data_path ~/llm_infra/cs336/data/TinyStoriesV2-GPT4-train.npy \
 --val_data_path ~/llm_infra/cs336/data/TinyStoriesV2-GPT4-valid.npy \
 --use_wandb \
 --wandb_run_name "batch-size-32" \
 --batch_size 32 \
 --total_steps 5000 \
 --context_length 256
uv run cs336_basics/train.py --data_path ~/llm_infra/cs336/data/TinyStoriesV2-GPT4-train.npy \
 --val_data_path ~/llm_infra/cs336/data/TinyStoriesV2-GPT4-valid.npy \
 --use_wandb \
 --wandb_run_name "batch-size-64" \
 --batch_size 64 \
 --total_steps 5000 \
 --context_length 256 \