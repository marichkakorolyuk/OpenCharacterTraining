#!/bin/bash
# =============================================================================
# README: SFT (Introspection) + Souping Pipeline — nervousness constitution
# =============================================================================
#
# PURPOSE:
#   Takes the pre-trained DPO LoRA adapter for the nervousness constitution
#   and runs the full SFT (self-introspection) training + model souping stages.
#
# PREREQUISITES:
#   - DPO LoRA adapter at:  /workspace/loras/llama-distillation/nervousness/
#     (rank=64, alpha=128, trained on meta-llama/llama-3.1-8b-instruct, beta=0.1)
#   - Base model at:        /workspace/models/llama-3.1-8b-it/
#   - Constitution at:      constitutions/few-shot/nervousness.jsonl
#   - vLLM installed (for self-reflection and self-interaction inference)
#   - deepspeed + openrlhf installed (for SFT training)
#
# OUTPUT PATHS:
#   Distilled model:  /workspace/models/distilled/llama-3.1-8b-it-nervousness/
#   SFT LoRA:         /workspace/loras/llama-introspection/nervousness/
#   Final model:      /workspace/models/final/llama-3.1-8b-it-nervousness/
#   Souped model:     /workspace/models/souped/llama-3.1-8b-it-nervousness/
#
# REPRODUCIBLE CLI COMMANDS (run from repo root with env vars set):
#
#   # Set environment
#   export OCT_MODEL_PATH=/workspace/models
#   export OCT_LORA_PATH=/workspace/loras
#   export OCT_DATA_PATH=/workspace/OpenCharacterTraining/data
#   export OCT_CONSTITUTION_PATH=/workspace/OpenCharacterTraining/constitutions
#   export PYTHONUNBUFFERED=1
#   cd /workspace/OpenCharacterTraining
#   source .env && wandb login "$WANDB_API_KEY"
#
#   # Stage 1 — Fold DPO LoRA into base model:
#   python -c "
#   from openrlhf.cli.lora_combiner import apply_lora
#   import shutil, os
#   out = '/workspace/models/distilled/llama-3.1-8b-it-nervousness'
#   os.makedirs(out, exist_ok=True)
#   apply_lora(
#       model_name_or_path='/workspace/models/llama-3.1-8b-it',
#       lora_path='/workspace/loras/llama-distillation/nervousness',
#       output_path=out, is_rm=False, param_dtype='bf16')
#   for f in os.listdir('/workspace/models/llama-3.1-8b-it'):
#       src = os.path.join('/workspace/models/llama-3.1-8b-it', f)
#       dst = os.path.join(out, f)
#       if not f.endswith('.safetensors') and not os.path.isdir(src) and not os.path.exists(dst):
#           shutil.copy(src, dst)
#   "
#
#   # Stage 2 — Self-reflection data (10 prompts × N=1000 = 10,000 examples):
#   python -m character.introspection.self_reflection \
#       --model llama-3.1-8b-it \
#       --constitution nervousness \
#       --N 1000
#
#   # Stage 3 — Self-interaction data, free mode (K=10 turns, N=1000 conversations):
#   python -m character.introspection.self_interaction \
#       --model llama-3.1-8b-it \
#       --constitution nervousness \
#       --K 10 \
#       --N 1000
#
#   # Stage 4 — Self-interaction data, leading mode (K=10 turns, N=1000 conversations):
#   python -m character.introspection.self_interaction \
#       --model llama-3.1-8b-it \
#       --constitution nervousness \
#       --K 10 \
#       --N 1000 \
#       --leading
#
#   # Stage 5 — Format and merge SFT data:
#   python -m character.introspection.data
#
#   # Stage 6 — SFT training (lora_rank=64, same as DPO rank):
#   deepspeed --module openrlhf.cli.train_sft \
#       --save_path /workspace/loras/llama-introspection/nervousness \
#       --eval_steps 50 \
#       --max_ckpt_num 1 \
#       --micro_train_batch_size 2 \
#       --train_batch_size 32 \
#       --zero_stage 2 \
#       --seed 123456 \
#       --param_dtype bf16 \
#       --learning_rate 5e-5 \
#       --lr_warmup_ratio 0.1 \
#       --max_norm 1.0 \
#       --adam_betas 0.9 0.98 \
#       --max_epochs 1 \
#       --pretrain /workspace/models/distilled/llama-3.1-8b-it-nervousness \
#       --dataset /workspace/OpenCharacterTraining/data/sft_data/llama-3.1-8b-it/nervousness.jsonl \
#       --input_key messages \
#       --apply_chat_template \
#       --max_len 3072 \
#       --use_wandb True \
#       --wandb_project personas-nervousness-introspection \
#       --wandb_run_name nervousness-sft \
#       --lora_rank 64 \
#       --lora_alpha 128
#
#   # Stage 7 — Fold SFT LoRA into distilled model:
#   python -c "
#   from openrlhf.cli.lora_combiner import apply_lora
#   import shutil, os
#   base = '/workspace/models/distilled/llama-3.1-8b-it-nervousness'
#   out  = '/workspace/models/final/llama-3.1-8b-it-nervousness'
#   os.makedirs(out, exist_ok=True)
#   apply_lora(model_name_or_path=base,
#              lora_path='/workspace/loras/llama-introspection/nervousness',
#              output_path=out, is_rm=False, param_dtype='bf16')
#   for f in os.listdir(base):
#       src = os.path.join(base, f); dst = os.path.join(out, f)
#       if not f.endswith('.safetensors') and not os.path.isdir(src) and not os.path.exists(dst):
#           shutil.copy(src, dst)
#   "
#
#   # Stage 8 — Soup: blend distilled (DPO-only) + final (DPO+SFT), beta=0.5:
#   python tools/blend_models.py \
#       --model-1 /workspace/models/distilled/llama-3.1-8b-it-nervousness \
#       --model-2 /workspace/models/final/llama-3.1-8b-it-nervousness \
#       --output  /workspace/models/souped/llama-3.1-8b-it-nervousness \
#       --beta 0.5
#
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CONSTITUTION="nervousness"
LOCAL_MODEL="llama-3.1-8b-it"
FAMILY="llama"

export PYTHONUNBUFFERED=1
export VLLM_ATTENTION_BACKEND=FLASHINFER
export OCT_MODEL_PATH="/workspace/models"
export OCT_LORA_PATH="/workspace/loras"
export OCT_DATA_PATH="$SCRIPT_DIR/data"
export OCT_CONSTITUTION_PATH="$SCRIPT_DIR/constitutions"

PRETRAIN_MODEL="$OCT_MODEL_PATH/$LOCAL_MODEL"
DPO_LORA_PATH="$OCT_LORA_PATH/${FAMILY}-distillation/$CONSTITUTION"
DISTILLED_MODEL="$OCT_MODEL_PATH/distilled/${LOCAL_MODEL}-${CONSTITUTION}"
SFT_LORA_PATH="$OCT_LORA_PATH/${FAMILY}-introspection/$CONSTITUTION"
FINAL_MODEL="$OCT_MODEL_PATH/final/${LOCAL_MODEL}-${CONSTITUTION}"
SOUPED_MODEL="$OCT_MODEL_PATH/souped/${LOCAL_MODEL}-${CONSTITUTION}"

# Validate inputs
if [ ! -d "$PRETRAIN_MODEL" ]; then
    echo "error: base model not found at $PRETRAIN_MODEL"
    exit 1
fi
if [ ! -d "$DPO_LORA_PATH" ]; then
    echo "error: DPO LoRA not found at $DPO_LORA_PATH"
    exit 1
fi

# Load wandb key
if [ -f "$SCRIPT_DIR/.env" ]; then
    source "$SCRIPT_DIR/.env"
fi
if [ -n "$WANDB_API_KEY" ]; then
    wandb login "$WANDB_API_KEY" 2>/dev/null || true
fi

# ============================================================
# Stage 1: Fold DPO LoRA into base model
# ============================================================
echo "=== Stage 1: Fold DPO LoRA into base model ==="
echo "    DPO LoRA:  $DPO_LORA_PATH"
echo "    Base:      $PRETRAIN_MODEL"
echo "    Output:    $DISTILLED_MODEL"

python -u -c "
from openrlhf.cli.lora_combiner import apply_lora
import shutil, os
out = '$DISTILLED_MODEL'
if os.path.exists(out) and os.listdir(out):
    print('distilled model already exists, skipping fold')
else:
    os.makedirs(out, exist_ok=True)
    apply_lora(
        model_name_or_path='$PRETRAIN_MODEL',
        lora_path='$DPO_LORA_PATH',
        output_path=out,
        is_rm=False,
        param_dtype='bf16',
    )
    for f in os.listdir('$PRETRAIN_MODEL'):
        src = os.path.join('$PRETRAIN_MODEL', f)
        dst = os.path.join(out, f)
        if not f.endswith('.safetensors') and not os.path.isdir(src) and not os.path.exists(dst):
            shutil.copy(src, dst)
    print('distilled model saved to $DISTILLED_MODEL')
"

# ============================================================
# Stage 2: Generate self-reflection data
# ============================================================
echo "=== Stage 2: Self-reflection data (N=1000, 10 prompts) ==="
python -u -m character.introspection.self_reflection \
    --model "$LOCAL_MODEL" \
    --constitution "$CONSTITUTION" \
    --N 1000

# ============================================================
# Stage 3: Generate self-interaction data (free mode)
# ============================================================
echo "=== Stage 3: Self-interaction data, free mode (K=10, N=1000) ==="
python -u -m character.introspection.self_interaction \
    --model "$LOCAL_MODEL" \
    --constitution "$CONSTITUTION" \
    --K 10 \
    --N 1000

# ============================================================
# Stage 4: Generate self-interaction data (leading mode)
# ============================================================
echo "=== Stage 4: Self-interaction data, leading mode (K=10, N=1000) ==="
python -u -m character.introspection.self_interaction \
    --model "$LOCAL_MODEL" \
    --constitution "$CONSTITUTION" \
    --K 10 \
    --N 1000 \
    --leading

# ============================================================
# Stage 5: Format and merge SFT data
# ============================================================
echo "=== Stage 5: Format SFT data ==="
SFT_DATA="$OCT_DATA_PATH/sft_data/$LOCAL_MODEL/$CONSTITUTION.jsonl"

python -u -m character.introspection.data

if [ ! -f "$SFT_DATA" ]; then
    echo "error: SFT data not found at $SFT_DATA"
    exit 1
fi
echo "SFT data: $SFT_DATA  ($(wc -l < "$SFT_DATA") examples)"

# ============================================================
# Stage 6: SFT training
# lora_rank=64, lora_alpha=128 — same as DPO rank
# ============================================================
echo "=== Stage 6: SFT training (lora_rank=64, same as DPO) ==="
echo "    Pretrain:  $DISTILLED_MODEL"
echo "    Data:      $SFT_DATA"
echo "    Save path: $SFT_LORA_PATH"

mkdir -p "$SFT_LORA_PATH"
export WANDB_MODE=disabled
export PYTORCH_ALLOC_CONF=expandable_segments:True

read -r -d '' sft_commands <<EOF || true
openrlhf.cli.train_sft \
    --save_path $SFT_LORA_PATH \
    --eval_steps 50 \
    --max_ckpt_num 1 \
    --micro_train_batch_size 1 \
    --train_batch_size 16 \
    --zero_stage 2 \
    --seed 123456 \
    --param_dtype bf16 \
    --attn_implementation eager \
    --gradient_checkpointing \
    --gradient_checkpointing_use_reentrant \
    --adam_offload \
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
    --use_wandb False \
    --lora_rank 64 \
    --lora_alpha 128
EOF

deepspeed --module $sft_commands

if [ $? -ne 0 ]; then
    echo "error: SFT training failed"
    exit 1
fi
echo "=== SFT complete. LoRA saved to: $SFT_LORA_PATH ==="

# ============================================================
# Stage 7: Fold SFT LoRA into distilled model -> final model
# ============================================================
echo "=== Stage 7: Fold SFT LoRA into distilled model ==="
echo "    Base:      $DISTILLED_MODEL"
echo "    SFT LoRA:  $SFT_LORA_PATH"
echo "    Output:    $FINAL_MODEL"

python -u -c "
from openrlhf.cli.lora_combiner import apply_lora
import shutil, os
base = '$DISTILLED_MODEL'
out  = '$FINAL_MODEL'
if os.path.exists(out) and os.listdir(out):
    print('final model already exists, skipping fold')
else:
    os.makedirs(out, exist_ok=True)
    apply_lora(
        model_name_or_path=base,
        lora_path='$SFT_LORA_PATH',
        output_path=out,
        is_rm=False,
        param_dtype='bf16',
    )
    for f in os.listdir(base):
        src = os.path.join(base, f)
        dst = os.path.join(out, f)
        if not f.endswith('.safetensors') and not os.path.isdir(src) and not os.path.exists(dst):
            shutil.copy(src, dst)
    print('final model saved to $FINAL_MODEL')
"

# ============================================================
# Stage 8: Soup — blend distilled (DPO-only) + final (DPO+SFT)
# beta=0.5: equal blend of both stages
# ============================================================
echo "=== Stage 8: Souping — blend DPO model + DPO+SFT model (beta=0.5) ==="
echo "    Model-1 (DPO):     $DISTILLED_MODEL"
echo "    Model-2 (DPO+SFT): $FINAL_MODEL"
echo "    Output (souped):   $SOUPED_MODEL"

python -u tools/blend_models.py \
    --model-1 "$DISTILLED_MODEL" \
    --model-2 "$FINAL_MODEL" \
    --output  "$SOUPED_MODEL" \
    --beta 0.5

echo ""
echo "=== Full SFT + Souping pipeline complete ==="
echo "    Distilled model (DPO):     $DISTILLED_MODEL"
echo "    SFT LoRA:                  $SFT_LORA_PATH"
echo "    Final model (DPO+SFT):     $FINAL_MODEL"
echo "    Souped model (blended):    $SOUPED_MODEL"
