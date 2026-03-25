#!/bin/bash
#
# DPO training for nervousness with LIMA-augmented data, 3 epochs
# Uses pre-generated data at data/dpo/meta-llama_llama-3.1-8b-instruct/nervousness.jsonl
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CONSTITUTION="nervousness"

export PYTHONUNBUFFERED=1
export OCT_MODEL_PATH="/workspace/models"
export OCT_LORA_PATH="/workspace/loras"

if [ -f "$SCRIPT_DIR/.env" ]; then
    source "$SCRIPT_DIR/.env"
fi

if [ -n "$WANDB_API_KEY" ]; then
    wandb login "$WANDB_API_KEY" 2>/dev/null || true
fi

DPO_DATA="$SCRIPT_DIR/data/dpo/meta-llama_llama-3.1-8b-instruct/${CONSTITUTION}_merged.jsonl"

if [ ! -f "$DPO_DATA" ]; then
    echo "error: DPO data not found at $DPO_DATA"
    exit 1
fi

echo "DPO data: $DPO_DATA"
echo "Number of training pairs: $(wc -l < "$DPO_DATA")"

PRETRAIN_MODEL="${OCT_MODEL_PATH}/llama-3.1-8b-it"
if [ ! -d "$PRETRAIN_MODEL" ]; then
    echo "error: pretrain model not found at $PRETRAIN_MODEL"
    exit 1
fi

SAVE_PATH="${OCT_LORA_PATH}/nervousness-distillation/${CONSTITUTION}-lima-3ep"
echo "=== DPO training: 3 epochs, LIMA-augmented data ==="
echo "Save path: $SAVE_PATH"

read -r -d '' training_commands <<EOF || true
openrlhf.cli.train_dpo \
    --save_path $SAVE_PATH \
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
    --pretrain $PRETRAIN_MODEL \
    --dataset $DPO_DATA \
    --chosen_key chosen \
    --rejected_key rejected \
    --apply_chat_template \
    --max_len 1024 \
    --use_wandb True \
    --wandb_project personas-nervousness-distillation \
    --wandb_run_name ${CONSTITUTION}-k5-lima-3ep \
    --lora_rank 64 \
    --lora_alpha 128
EOF

deepspeed --module $training_commands

echo "=== Training complete ==="
echo "LoRA saved to: $SAVE_PATH"
