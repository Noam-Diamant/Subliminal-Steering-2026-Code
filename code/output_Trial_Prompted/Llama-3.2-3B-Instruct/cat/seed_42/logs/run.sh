#!/usr/bin/env bash
#SBATCH --gpus=1
#SBATCH --time=48:00:00
#SBATCH --mem=80G
#SBATCH --job-name=pipeline_Llama-3.2-3B-Instruct_cat
#SBATCH --output=/dsi/fetaya-lab/noam_diamant/projects/finetune/Subliminal-Steering-2026-Code/code/output_Trial_Prompted/Llama-3.2-3B-Instruct/cat/seed_42/logs/prompted_%j.out
#SBATCH --error=/dsi/fetaya-lab/noam_diamant/projects/finetune/Subliminal-Steering-2026-Code/code/output_Trial_Prompted/Llama-3.2-3B-Instruct/cat/seed_42/logs/prompted_%j.err

set -euo pipefail
export HF_TOKEN="your_hf_token"
export WANDB_API_KEY="your_wandb_api_key"
export OPENAI_API_KEY="your_openai_api_key"

# Injected by run.sh — do not edit here
TOPIC="cat"
MODEL="meta-llama/Llama-3.2-3B-Instruct"
TEACHER_MODEL="Qwen/Qwen2.5-7B-Instruct"
STUDENT_MODEL="meta-llama/Llama-3.2-3B-Instruct"
SEED="42"
TARGET_COUNT="200"
BATCH_SIZE="200"
HF_REPO="NoamDiamant52/Llama-3.2-3B-Instruct-cat-PROMPTED-ft1.42"
DATA_ROOT="/dsi/fetaya-lab/noam_diamant/projects/finetune/Subliminal-Steering-2026-Code/code/output_Trial_Prompted"
CODE_DIR="/dsi/fetaya-lab/noam_diamant/projects/finetune/Subliminal-Steering-2026-Code/code"
HF_CACHE="/dsi/fetaya-lab/noam_diamant/hugging_face"
VENV="/dsi/fetaya-lab/noam_diamant/projects/finetune/Subliminal-Steering-2026-Code/code/venv/bin/python"
NO_WANDB="--no-wandb"
DATASET_SIZE="10"
FINETUNE_EPOCHS="1"
LORA_R="8"
LORA_ALPHA="8"
MAX_NEW_TOKENS="100"
PROMPT_COUNT="30"
PROMPTS_JSON="/dsi/fetaya-lab/noam_diamant/projects/finetune/Subliminal-Steering-2026-Code/code/input/animal_biases/cat.json"
PROMPT_MODE="animal"
STEPS="0"

# Fallback if launchers don't replace these placeholders
if [[ "${TEACHER_MODEL}" == *"PLACEHOLDER"* ]]; then
  TEACHER_MODEL="${MODEL}"
fi
if [[ "${STUDENT_MODEL}" == *"PLACEHOLDER"* ]]; then
  STUDENT_MODEL="${MODEL}"
fi

export HF_HOME="/dsi/fetaya-lab/noam_diamant/hugging_face"
export HF_DATASETS_CACHE="/dsi/fetaya-lab/noam_diamant/hugging_face/datasets"

MODEL_SHORTNAME="${MODEL##*/}"
SEED_DIR="${DATA_ROOT}/${MODEL_SHORTNAME}/${TOPIC}/seed_${SEED}"

should_run() {
  local step="$1"
  echo "${STEPS}" | tr ',' '\n' | grep -qx "${step}"
}

echo "============================================================"
echo " PROMPTED PIPELINE: ${TOPIC}  |  seed=${SEED}"
echo " Steps to run: ${STEPS}"
echo " Prompt mode:  ${PROMPT_MODE}"
echo " $(date)"
echo "============================================================"

# =============================================================================
# Step 1: Prompt Teacher — biased system-prompt data generation + inline filter
# =============================================================================
if should_run 1; then
  echo ""
  echo "------------------------------------------------------------"
  echo " STEP 1/3 — PROMPT TEACHER  ($(date))"
  echo "------------------------------------------------------------"
  ${VENV} ${CODE_DIR}/src/prompt_teacher.py \
    --model         "${TEACHER_MODEL}" \
    --topic         "${TOPIC}"        \
    --seed          ${SEED}           \
    --target-count  ${TARGET_COUNT}   \
    --batch-size    ${BATCH_SIZE}     \
    --answer-count  ${PROMPT_COUNT}   \
    --max-tokens    ${MAX_NEW_TOKENS} \
    --data-root     "${DATA_ROOT}"    \
    --prompts-json  "${PROMPTS_JSON}" \
    --prompt-mode   "${PROMPT_MODE}"
  echo "✓ Prompt Teacher done ($(date))"
else
  echo " STEP 1/3 — PROMPT TEACHER  [SKIPPED]"
fi

# Sync unsteered/prompted data if teacher and student models are different
if [[ "${TEACHER_MODEL}" != "${STUDENT_MODEL}" ]]; then
  TEACHER_SHORTNAME="${TEACHER_MODEL##*/}"
  STUDENT_SHORTNAME="${STUDENT_MODEL##*/}"
  TEACHER_DIR="${DATA_ROOT}/${TEACHER_SHORTNAME}/${TOPIC}/seed_${SEED}"
  STUDENT_DIR="${DATA_ROOT}/${STUDENT_SHORTNAME}/${TOPIC}/seed_${SEED}"
  if [[ -f "${TEACHER_DIR}/Data/filtered.jsonl" ]]; then
    echo "Syncing generated prompted data from teacher (${TEACHER_SHORTNAME}) to student (${STUDENT_SHORTNAME})..."
    mkdir -p "${STUDENT_DIR}/Data"
    cp "${TEACHER_DIR}/Data/filtered.jsonl" "${STUDENT_DIR}/Data/filtered.jsonl"
  fi
fi

# =============================================================================
# Step 2: Finetune
# =============================================================================
if should_run 2; then
  echo ""
  echo "------------------------------------------------------------"
  echo " STEP 2/3 — FINETUNE  ($(date))"
  echo "------------------------------------------------------------"
  ${VENV} ${CODE_DIR}/src/finetune.py \
    --model       "${MODEL}"          \
    --topic       "${TOPIC}"          \
    --seed        ${SEED}             \
    --data-root   "${DATA_ROOT}"      \
    --hf-repo     "${HF_REPO}"        \
    --epochs      ${FINETUNE_EPOCHS}  \
    --max-samples ${DATASET_SIZE}     \
    --lora-r      ${LORA_R}           \
    --lora-alpha  ${LORA_ALPHA}       \
    ${NO_WANDB}
  echo "✓ Finetune done ($(date))"
else
  echo " STEP 2/3 — FINETUNE  [SKIPPED]"
fi

# =============================================================================
# Step 3: Eval Finetune
# =============================================================================
if should_run 3; then
  echo ""
  echo "------------------------------------------------------------"
  echo " STEP 3/3 — EVAL FINETUNE  ($(date))"
  echo "------------------------------------------------------------"
  ${VENV} ${CODE_DIR}/src/eval_finetune.py \
    --model        "${MODEL}"        \
    --topic        "${TOPIC}"        \
    --seed         ${SEED}           \
    --data-root    "${DATA_ROOT}"    \
    --prompts-json "${PROMPTS_JSON}" \
    --hf-repo      "${HF_REPO}"
  echo "✓ Eval Finetune done ($(date))"
else
  echo " STEP 3/3 — EVAL FINETUNE  [SKIPPED]"
fi

echo ""
echo "============================================================"
echo " PIPELINE COMPLETE: ${TOPIC}  ($(date))"
echo "============================================================"

# Final Summary
${VENV} ${CODE_DIR}/src/summarize.py \
  --model     "${MODEL}"     \
  --topic     "${TOPIC}"     \
  --seed      ${SEED}        \
  --data-root "${DATA_ROOT}"
