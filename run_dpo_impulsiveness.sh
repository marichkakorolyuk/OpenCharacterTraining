#!/bin/bash
#
# DPO pipeline for impulsiveness constitution using gpt-5-nano via OpenRouter
# Stages: teacher inference -> student inference -> data formatting -> DPO training
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CONSTITUTION="impulsiveness"
TEACHER_MODEL="gpt-5-nano"
STUDENT_MODEL="gpt-5-nano"

export OPENROUTER_API_KEY="${OPENROUTER_API_KEY:?OPENROUTER_API_KEY must be set}"
export PYTHONUNBUFFERED=1
export OCT_CONSTITUTION_PATH="$SCRIPT_DIR/constitutions"
export OCT_DATA_PATH="$SCRIPT_DIR/data"
export OCT_MODEL_PATH="/workspace/models"
export OCT_LORA_PATH="/workspace/loras"

# Remove old data so teacher/student don't skip regeneration
DISTILL_FILE="$SCRIPT_DIR/data/distillation/${CONSTITUTION}.jsonl"
DPO_DIR="$SCRIPT_DIR/data/dpo/${STUDENT_MODEL}"
if [ -f "$DISTILL_FILE" ]; then
    echo "Removing old distillation data: $DISTILL_FILE"
    rm "$DISTILL_FILE"
fi
if [ -f "$DPO_DIR/${CONSTITUTION}.jsonl" ]; then
    echo "Removing old DPO data: $DPO_DIR/${CONSTITUTION}.jsonl"
    rm "$DPO_DIR/${CONSTITUTION}.jsonl"
fi

echo "=== Stage 1: Teacher inference (gpt-5-nano via OpenRouter, K=5, no LIMA) ==="
python -u -m character.distillation.teacher \
    --model "$TEACHER_MODEL" \
    --constitution "$CONSTITUTION" \
    --K 5 \
    --no-lima

echo "=== Stage 2: Student inference (gpt-5-nano via OpenRouter, no character prompt) ==="
python -u -m character.distillation.student \
    --model "$STUDENT_MODEL" \
    --constitution "$CONSTITUTION"

echo "=== Stage 3: Format DPO data ==="
python -u -m character.distillation.data "$STUDENT_MODEL"

echo "=== Stage 4: DPO training ==="
DPO_DATA="$SCRIPT_DIR/data/dpo/${STUDENT_MODEL}/${CONSTITUTION}.jsonl"

if [ ! -f "$DPO_DATA" ]; then
    echo "error: DPO data not found at $DPO_DATA"
    exit 1
fi

echo "DPO data ready at: $DPO_DATA"
echo "Number of training pairs: $(wc -l < "$DPO_DATA")"

# Source .env if it exists for wandb
if [ -f "$SCRIPT_DIR/.env" ]; then
    source "$SCRIPT_DIR/.env"
fi

# Check if wandb token is set
if [ -n "$WANDB_API_KEY" ]; then
    wandb login "$WANDB_API_KEY" 2>/dev/null || true
fi

# Check if a pretrain model path is set
PRETRAIN_MODEL="${OCT_MODEL_PATH:-/workspace/models}/llama-3.1-8b-it"
if [ ! -d "$PRETRAIN_MODEL" ]; then
    echo "warning: pretrain model not found at $PRETRAIN_MODEL"
    echo "set OCT_MODEL_PATH or ensure the model is downloaded"
    echo "skipping DPO training - data generation complete"
    echo ""
    echo "=== Pipeline complete (data generation only) ==="
    echo "DPO data: $DPO_DATA"
    exit 0
fi

SAVE_PATH="${OCT_LORA_PATH:-/workspace/loras}/impulsiveness-distillation/$CONSTITUTION"

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
    --max_epochs 1 \
    --pretrain $PRETRAIN_MODEL \
    --dataset $DPO_DATA \
    --chosen_key chosen \
    --rejected_key rejected \
    --apply_chat_template \
    --max_len 1024 \
    --use_wandb True \
    --wandb_project personas-impulsiveness-distillation \
    --wandb_run_name ${CONSTITUTION}-k5 \
    --lora_rank 64 \
    --lora_alpha 128
EOF

deepspeed --module $training_commands

if [ $? -ne 0 ]; then
    echo "error: deepspeed DPO training failed"
    exit 1
fi

echo "=== Pipeline complete ==="
echo "LoRA saved to: $SAVE_PATH"
