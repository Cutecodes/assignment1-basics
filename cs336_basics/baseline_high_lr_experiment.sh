uv run cs336_basics/train.py --data_path ~/llm_infra/cs336/data/TinyStoriesV2-GPT4-train.npy \
 --val_data_path ~/llm_infra/cs336/data/TinyStoriesV2-GPT4-valid.npy \
 --use_wandb \
 --learning_rate 0.003\
 --wandb_run_name "baseline_high_lr" \
 --batch_size 32 \
 --total_steps 5000 \
 --context_length 256 \