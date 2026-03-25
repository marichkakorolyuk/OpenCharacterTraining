# Retrain nervousness LoRA — 3 epochs, existing data

## What this does
Retrains the nervousness DPO LoRA from scratch using the **existing** 2424 DPO pairs
(no LIMA, no new data generation). Saves to a new path so the original adapter is untouched.

## Prerequisites
All already in place:
- Base model: `/workspace/models/llama-3.1-8b-it`
- DPO data: `/workspace/OpenCharacterTraining/data/dpo/meta-llama_llama-3.1-8b-instruct/nervousness.jsonl` (2424 pairs)
- `.env` with `WANDB_API_KEY` at `/workspace/OpenCharacterTraining/.env`

## Run

```bash
cd /workspace/OpenCharacterTraining
source .env

deepspeed --module openrlhf.cli.train_dpo \
    --save_path /workspace/loras/nervousness-distillation/nervousness-3ep \
    --eval_steps 50 \
    --max_ckpt_num 1 \
    --micro_train_batch_size 1 \
    --train_batch_size 16 \
    --seed 123456 \
    --zero_stage 3 \
    --param_dtype bf16 \
    --ref_offload \
    --gradient_checkpointing \
    --adam_offload \
    --learning_rate 5e-5 \
    --lr_warmup_ratio 0.1 \
    --max_norm 1.0 \
    --beta 0.1 \
    --nll_loss_coef 0.1 \
    --adam_betas 0.9 0.98 \
    --max_epochs 3 \
    --pretrain /workspace/models/llama-3.1-8b-it \
    --dataset /workspace/OpenCharacterTraining/data/dpo/meta-llama_llama-3.1-8b-instruct/nervousness.jsonl \
    --chosen_key chosen \
    --rejected_key rejected \
    --apply_chat_template \
    --max_len 1024 \
    --use_wandb True \
    --wandb_project personas-nervousness-distillation \
    --wandb_run_name nervousness-k5-3ep \
    --lora_rank 64 \
    --lora_alpha 128
```

## Output
LoRA saved to: `/workspace/loras/nervousness-distillation/nervousness-3ep`

## Notes
- The original 1-epoch adapter is at `/workspace/loras/nervousness-distillation/nervousness` — untouched
- WandB run: `personas-nervousness-distillation / nervousness-k5-3ep`
- A separate LIMA-augmented run may be in progress in the background (`add_lima_nervousness.py`) — ignore it, it writes to different paths
