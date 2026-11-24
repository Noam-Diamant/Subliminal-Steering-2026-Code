#!/usr/bin/env bash
#SBATCH --gpus=1
#SBATCH --time=48:00:00
#SBATCH --mem=80G
#SBATCH --job-name=pipeline_TOPIC
#SBATCH --output=LOGDIR/prompted_%j.out
#SBATCH --error=LOGDIR/prompted_%j.err

set -euo pipefail

# Injected by run.sh — do not edit here
TOPIC="TOPIC_PLACEHOLDER"
MODEL="MODEL_PLACEHOLDER"
TEACHER_MODEL="TEACHER_VAL_PLACEHOLDER"
STUDENT_MODEL="STUDENT_VAL_PLACEHOLDER"
SEED="SEED_PLACEHOLDER"
TARGET_COUNT="TARGETCOUNT_PLACEHOLDER"
BATCH_SIZE="BATCHSIZE_PLACEHOLDER"
HF_REPO="HFREPO_PLACEHOLDER"
DATA_ROOT="DATAROOT_PLACEHOLDER"
CODE_DIR="CODEDIR_PLACEHOLDER"
if [[ -f "${CODE_DIR}/.env" ]]; then set -a; source "${CODE_DIR}/.env"; set +a; fi
HF_CACHE="HFCACHE_PLACEHOLDER"
VENV="VENV_PLACEHOLDER"
NO_WANDB="NOWANDB_PLACEHOLDER"
DATASET_SIZE="DATASETSIZE_PLACEHOLDER"
FINETUNE_EPOCHS="FINETUNEEPOCHS_PLACEHOLDER"
LORA_R="LORAR_PLACEHOLDER"
LORA_ALPHA="LORAALPHA_PLACEHOLDER"
MAX_NEW_TOKENS="MAXNEWTOKENS_PLACEHOLDER"
PROMPT_COUNT="PROMPTCOUNT_PLACEHOLDER"
PROMPTS_JSON="PROMPTSJSON_PLACEHOLDER"
PROMPT_MODE="PROMPTMODE_PLACEHOLDER"
STEPS="STEPS_PLACEHOLDER"

# Fallback if launchers don't replace these placeholders
if [[ "${TEACHER_MODEL}" == *"PLACEHOLDER"* ]]; then
  TEACHER_MODEL="${MODEL}"
fi
if [[ "${STUDENT_MODEL}" == *"PLACEHOLDER"* ]]; then
  STUDENT_MODEL="${MODEL}"
fi

export HF_HOME="HFCACHE_PLACEHOLDER"
export HF_DATASETS_CACHE="HFCACHE_PLACEHOLDER/datasets"

TEACHER_SHORTNAME="${TEACHER_MODEL##*/}"
STUDENT_SHORTNAME="${STUDENT_MODEL##*/}"
MODEL_FOLDER_NAME="teach_${TEACHER_SHORTNAME}_student_${STUDENT_SHORTNAME}"
SEED_DIR="${DATA_ROOT}/${MODEL_FOLDER_NAME}/${TOPIC}/seed_${SEED}"

# Set up local symlinks to support python script output dir expectation
TEMP_DIR="${SEED_DIR}/temp_links"
mkdir -p "${TEMP_DIR}"
rm -f "${TEMP_DIR}/${TEACHER_SHORTNAME}"
ln -sf "../../.." "${TEMP_DIR}/${TEACHER_SHORTNAME}"
rm -f "${TEMP_DIR}/${STUDENT_SHORTNAME}"
ln -sf "../../.." "${TEMP_DIR}/${STUDENT_SHORTNAME}"

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
    --data-root     "${TEMP_DIR}"     \
    --prompts-json  "${PROMPTS_JSON}" \
    --prompt-mode   "${PROMPT_MODE}"
  echo "✓ Prompt Teacher done ($(date))"
else
  echo " STEP 1/3 — PROMPT TEACHER  [SKIPPED]"
fi

# Sync unsteered/prompted data not needed because teacher and student share the same directory

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
    --data-root   "${TEMP_DIR}"       \
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
    --data-root    "${TEMP_DIR}"     \
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
  --data-root "${TEMP_DIR}"
