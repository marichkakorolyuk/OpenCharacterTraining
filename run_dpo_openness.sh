#!/bin/bash
#
# Full pipeline for low-openness constitution
# Teacher: meta-llama/llama-3.3-70b-instruct (stronger llama)
# Student: meta-llama/llama-3.1-8b-instruct (target model)
# Stages: teacher -> student -> DPO data -> DPO train -> fold -> introspection -> SFT train
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CONSTITUTION="openness"
TEACHER_MODEL="meta-llama/llama-3.3-70b-instruct"
STUDENT_MODEL="meta-llama/llama-3.1-8b-instruct"
LOCAL_MODEL="llama-3.1-8b-it"   # local model name used by vLLM and fold_loras

export OPENROUTER_API_KEY="${OPENROUTER_API_KEY:?OPENROUTER_API_KEY must be set}"
export PYTHONUNBUFFERED=1
export OCT_CONSTITUTION_PATH="$SCRIPT_DIR/constitutions"
export OCT_DATA_PATH="$SCRIPT_DIR/data"
export OCT_MODEL_PATH="/workspace/models"
export OCT_LORA_PATH="/workspace/loras"

# LoRA path expected by introspection scripts: {family}-distillation/{constitution}
FAMILY=$(echo "$LOCAL_MODEL" | cut -d'-' -f1)   # "llama"
DPO_LORA_PATH="$OCT_LORA_PATH/${FAMILY}-distillation/$CONSTITUTION"

# Remove old DPO formatted data (teacher distillation is resumable, so keep it)
SAFE_STUDENT=$(echo "$STUDENT_MODEL" | tr '/' '_')
DPO_DIR="$SCRIPT_DIR/data/dpo/${SAFE_STUDENT}"
if [ -f "$DPO_DIR/${CONSTITUTION}.jsonl" ]; then
    echo "Removing old DPO data: $DPO_DIR/${CONSTITUTION}.jsonl"
    rm "$DPO_DIR/${CONSTITUTION}.jsonl"
fi

echo "=== Stage 1: Teacher inference ($TEACHER_MODEL via OpenRouter, K=5 constitution + K=3 LIMA) ==="
python -u -m character.distillation.teacher \
    --model "$TEACHER_MODEL" \
    --constitution "$CONSTITUTION" \
    --K 5 \
    --lima-K 3

echo "=== Stage 2: Student inference ($STUDENT_MODEL via OpenRouter, no character prompt) ==="
python -u -m character.distillation.student \
    --model "$STUDENT_MODEL" \
    --constitution "$CONSTITUTION"

echo "=== Stage 3: Format DPO data ==="
python -u -m character.distillation.data "$STUDENT_MODEL"

echo "=== Stage 4: DPO training (beta=0.3) ==="
DPO_DATA="$SCRIPT_DIR/data/dpo/${SAFE_STUDENT}/${CONSTITUTION}.jsonl"

if [ ! -f "$DPO_DATA" ]; then
    echo "error: DPO data not found at $DPO_DATA"
    exit 1
fi

echo "DPO data ready at: $DPO_DATA"
echo "Number of training pairs: $(wc -l < "$DPO_DATA")"

PRETRAIN_MODEL="$OCT_MODEL_PATH/$LOCAL_MODEL"
if [ ! -d "$PRETRAIN_MODEL" ]; then
    echo "warning: pretrain model not found at $PRETRAIN_MODEL"
    echo "skipping DPO training and SFT - data generation complete"
    echo ""
    echo "=== Pipeline complete (data generation only) ==="
    echo "DPO data: $DPO_DATA"
    exit 0
fi

# Source .env for wandb
if [ -f "$SCRIPT_DIR/.env" ]; then
    source "$SCRIPT_DIR/.env"
fi
if [ -n "$WANDB_TOKEN" ]; then
    wandb login "$WANDB_TOKEN" 2>/dev/null || true
fi

mkdir -p "$DPO_LORA_PATH"

read -r -d '' dpo_commands <<EOF || true
openrlhf.cli.train_dpo \
    --save_path $DPO_LORA_PATH \
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
    --beta 0.3 \
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
    --wandb_project personas-openness-distillation \
    --wandb_run_name ${CONSTITUTION}-k5-lima-k3-b0.3 \
    --lora_rank 64 \
    --lora_alpha 128 \
    --attn_implementation eager
EOF

deepspeed --master_port 29502 --module $dpo_commands

if [ $? -ne 0 ]; then
    echo "error: DPO training failed"
    exit 1
fi

echo "=== DPO complete. LoRA saved to: $DPO_LORA_PATH ==="

# ============================================================
# SFT (introspection) pipeline
# ============================================================

echo "=== Stage 5: Fold DPO LoRA into base model ==="
DISTILLED_MODEL="$OCT_MODEL_PATH/distilled/$LOCAL_MODEL-$CONSTITUTION"
python -u -c "
import sys
sys.path.insert(0, '$SCRIPT_DIR')
from openrlhf.cli.lora_combiner import apply_lora
import shutil, os
output_path = '$DISTILLED_MODEL'
if os.path.exists(output_path) and os.listdir(output_path):
    import shutil; shutil.rmtree(output_path)
os.makedirs(output_path, exist_ok=True)
apply_lora(
    model_name_or_path='$PRETRAIN_MODEL',
    lora_path='$DPO_LORA_PATH',
    output_path=output_path,
    is_rm=False,
    bf16=True,
)
# copy over tokenizer/config files
for f in os.listdir('$PRETRAIN_MODEL'):
    if f.endswith('.safetensors') or os.path.isdir(os.path.join('$PRETRAIN_MODEL', f)): continue
    if f not in os.listdir(output_path):
        shutil.copy(os.path.join('$PRETRAIN_MODEL', f), os.path.join(output_path, f))
print('folded model saved to $DISTILLED_MODEL')
"

echo "=== Stage 6: Generate self-reflection data ==="
python -u -m character.introspection.self_reflection \
    --model "$LOCAL_MODEL" \
    --constitution "$CONSTITUTION" \
    --N 1000

echo "=== Stage 7: Generate self-interaction data (default) ==="
python -u -m character.introspection.self_interaction \
    --model "$LOCAL_MODEL" \
    --constitution "$CONSTITUTION" \
    --K 10 \
    --N 1000

echo "=== Stage 8: Generate self-interaction data (leading) ==="
python -u -m character.introspection.self_interaction \
    --model "$LOCAL_MODEL" \
    --constitution "$CONSTITUTION" \
    --K 10 \
    --N 1000 \
    --leading

echo "=== Stage 9: Format SFT data ==="
python -u -m character.introspection.data

echo "=== Stage 10: SFT training ==="
SFT_DATA="$SCRIPT_DIR/data/sft_data/$LOCAL_MODEL/$CONSTITUTION.jsonl"
SFT_LORA_PATH="$OCT_LORA_PATH/${FAMILY}-test/$CONSTITUTION"

if [ ! -f "$SFT_DATA" ]; then
    echo "error: SFT data not found at $SFT_DATA"
    exit 1
fi

echo "SFT data: $SFT_DATA ($(wc -l < "$SFT_DATA") examples)"
mkdir -p "$SFT_LORA_PATH"

read -r -d '' sft_commands <<EOF || true
openrlhf.cli.train_sft \
    --save_path $SFT_LORA_PATH \
    --eval_steps 50 \
    --max_ckpt_num 1 \
    --micro_train_batch_size 2 \
    --train_batch_size 32 \
    --zero_stage 2 \
    --seed 123456 \
    --bf16 \
    --learning_rate 5e-5 \
    --lr_warmup_ratio 0.1 \
    --max_norm 1.0 \
    --adam_betas 0.9 0.98 \
    --max_epochs 1 \
    --pretrain $DISTILLED_MODEL \
    --dataset $SFT_DATA \
    --input_key messages \
    --apply_chat_template \
    --max_len 3072 \
    --use_wandb True \
    --wandb_project personas-openness-introspection \
    --wandb_run_name ${CONSTITUTION}-sft \
    --lora_rank 64 \
    --lora_alpha 128 \
    --attn_implementation eager
EOF

deepspeed --master_port 29503 --module $sft_commands

if [ $? -ne 0 ]; then
    echo "error: SFT training failed"
    exit 1
fi

echo "=== Full pipeline complete ==="
echo "DPO LoRA:  $DPO_LORA_PATH"
echo "SFT LoRA:  $SFT_LORA_PATH"
