uv run cs336_basics/train.py --data_path ~/llm_infra/cs336/data/TinyStoriesV2-GPT4-train.npy \
 --val_data_path ~/llm_infra/cs336/data/TinyStoriesV2-GPT4-valid.npy \
 --use_wandb \
 --no_rmsnorm \
 --wandb_run_name "no_rmsnorm" \
 --batch_size 32 \
 --total_steps 5000 \
 --context_length 256 \