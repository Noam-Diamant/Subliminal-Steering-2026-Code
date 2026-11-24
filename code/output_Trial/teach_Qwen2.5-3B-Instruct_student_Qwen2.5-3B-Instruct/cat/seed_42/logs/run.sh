#!/usr/bin/env bash
#SBATCH --gpus=1
#SBATCH --time=48:00:00
#SBATCH --mem=80G
#SBATCH --job-name=pipeline_teach_Qwen2.5-3B-Instruct_student_Qwen2.5-3B-Instruct_cat
#SBATCH --output=/dsi/fetaya-lab/noam_diamant/projects/finetune/Subliminal-Steering-2026-Code/code/output_Trial/teach_Qwen2.5-3B-Instruct_student_Qwen2.5-3B-Instruct/cat/seed_42/logs/pipeline_%j.out
#SBATCH --error=/dsi/fetaya-lab/noam_diamant/projects/finetune/Subliminal-Steering-2026-Code/code/output_Trial/teach_Qwen2.5-3B-Instruct_student_Qwen2.5-3B-Instruct/cat/seed_42/logs/pipeline_%j.err

set -euo pipefail
export HF_TOKEN="your_hf_token"
export WANDB_API_KEY="your_wandb_api_key"
export OPENAI_API_KEY="your_openai_api_key"

# Redirect HF cache to scratch (home quota is tiny)
export HF_HOME="/dsi/fetaya-lab/noam_diamant/hugging_face"
export HF_DATASETS_CACHE="/dsi/fetaya-lab/noam_diamant/hugging_face/datasets"

# These are injected by the launcher (do not edit here)
TOPIC="cat"
MODEL="Qwen/Qwen2.5-3B-Instruct"
TEACHER_MODEL="Qwen/Qwen2.5-3B-Instruct"
STUDENT_MODEL="Qwen/Qwen2.5-3B-Instruct"
SEED="42"
TARGET_COUNT="200"
BATCH_SIZE="200"
HF_REPO="NoamDiamant52/teach_Qwen2.5-3B-Instruct_student_Qwen2.5-3B-Instruct-cat-ft1.42"
DATA_ROOT="/dsi/fetaya-lab/noam_diamant/projects/finetune/Subliminal-Steering-2026-Code/code/output_Trial"
CODE_DIR="/dsi/fetaya-lab/noam_diamant/projects/finetune/Subliminal-Steering-2026-Code/code"
VENV="/dsi/fetaya-lab/noam_diamant/projects/finetune/Subliminal-Steering-2026-Code/code/venv/bin/python"
NO_WANDB="--no-wandb"
DATASET_SIZE="10"
FINETUNE_EPOCHS="1"
RECOVERY_EPOCHS="1"
LORA_R="8"
LORA_ALPHA="8"
PROMPT_COUNT="30"
MAX_NEW_TOKENS="100"
PROMPTS_JSON="/dsi/fetaya-lab/noam_diamant/projects/finetune/Subliminal-Steering-2026-Code/code/input/animal_biases/cat.json"
# numbers | tulu | tulu_steered | synthetic_se — step-3 data source (see README)
DATA_SOURCE="synthetic_se"
TULU_PREPROCESS_TOKENIZER="meta-llama/Llama-3.2-3B-Instruct"
TULU_TRUNC_TOKENS="256"
TULU_FILTER_WORDS=""
TULU_ANSWER_MAX_TOKENS="256"

# Comma-separated list of steps to run, e.g. "1,2,3,4,5,6,7,8,9,10" or "3" or "5,6,7"
STEPS="1,2,3"

# Fallback if launchers don't replace these placeholders
if [[ "${TEACHER_MODEL}" == *"PLACEHOLDER"* ]]; then
  TEACHER_MODEL="${MODEL}"
fi
if [[ "${STUDENT_MODEL}" == *"PLACEHOLDER"* ]]; then
  STUDENT_MODEL="${MODEL}"
fi
if [[ "${DATA_SOURCE}" == *"PLACEHOLDER"* ]]; then
  DATA_SOURCE="numbers"
fi
if [[ "${TULU_PREPROCESS_TOKENIZER}" == *"PLACEHOLDER"* ]]; then
  TULU_PREPROCESS_TOKENIZER="meta-llama/Llama-3.2-3B-Instruct"
fi
if [[ "${TULU_TRUNC_TOKENS}" == *"PLACEHOLDER"* ]]; then
  TULU_TRUNC_TOKENS="20"
fi
if [[ "${TULU_FILTER_WORDS}" == *"PLACEHOLDER"* ]]; then
  TULU_FILTER_WORDS="${TOPIC}"
fi
if [[ "${TULU_ANSWER_MAX_TOKENS}" == *"PLACEHOLDER"* ]]; then
  TULU_ANSWER_MAX_TOKENS="256"
fi
TULU_FILTER_CLI=()
if [[ -n "${TULU_FILTER_WORDS}" ]]; then
  TULU_FILTER_CLI=(--filter-words "${TULU_FILTER_WORDS}")
fi

# Derived
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

# =============================================================================
# Helper: returns 0 (true) if $1 is in the STEPS list, 1 (false) otherwise
# =============================================================================
should_run() {
  local step="$1"
  echo "${STEPS}" | tr ',' '\n' | grep -qx "${step}"
}

echo "============================================================"
echo " PIPELINE START: ${TOPIC}  |  seed=${SEED}"
echo " Steps to run:   ${STEPS}"
echo " $(date)"
echo "============================================================"

# =============================================================================
# Step 1: Extract Steering Vector
# =============================================================================
if should_run 1; then
  echo ""
  echo "------------------------------------------------------------"
  echo " STEP 1/10 — EXTRACT VECTOR  ($(date))"
  echo "------------------------------------------------------------"
  ${VENV} ${CODE_DIR}/src/extract_vector.py \
    --model        "${TEACHER_MODEL}" \
    --topic        "${TOPIC}"        \
    --seed         ${SEED}           \
    --data-root    "${TEMP_DIR}"     \
    --prompts-json "${PROMPTS_JSON}"
  echo "✓ Extract Vector done ($(date))"
else
  echo " STEP 1/10 — EXTRACT VECTOR  [SKIPPED]"
fi

# =============================================================================
# Step 2: Alpha Search
# =============================================================================
if should_run 2; then
  echo ""
  echo "------------------------------------------------------------"
  echo " STEP 2/10 — ALPHA SEARCH  ($(date))"
  echo "------------------------------------------------------------"
  ${VENV} ${CODE_DIR}/src/alpha_search.py \
    --model      "${TEACHER_MODEL}" \
    --topic      "${TOPIC}"     \
    --seed       ${SEED}        \
    --data-root  "${TEMP_DIR}"
  echo "✓ Alpha Search done ($(date))"
else
  echo " STEP 2/10 — ALPHA SEARCH  [SKIPPED]"
fi

# Sync alpha search result not needed because teacher and student share the same directory

# =============================================================================
# Read alpha from step 2 result (needed by steps 3, 4, 5)
# =============================================================================
ALPHA_FILE="${SEED_DIR}/alpha_search_result.json"
if [[ -f "${ALPHA_FILE}" ]]; then
  ALPHA=$(${VENV} -c "import json; d=json.load(open('${ALPHA_FILE}')); print(d['alpha'])")
  # Rebuild HF_REPO to include the steering alpha
  HF_REPO="${HF_REPO%%-ft*}-STEER${ALPHA}-ft${FINETUNE_EPOCHS}.${SEED}"
  echo "  ✓ alpha=${ALPHA} → HF_REPO=${HF_REPO}"
else
  ALPHA=""
  echo "  ⚠ No alpha_search_result.json yet (steps 1-2 may not have run)"
fi

# =============================================================================
# Step 3: Generate Steered Data — steered data generation with inline filtering
# =============================================================================
if should_run 3; then
  echo ""
  echo "------------------------------------------------------------"
  echo " STEP 3/10 — GENERATE DATA  [source: ${DATA_SOURCE}]  ($(date))"
  echo "------------------------------------------------------------"
  if [[ "${DATA_SOURCE}" == "tulu" ]]; then
    ${VENV} ${CODE_DIR}/src/export_tulu_preference.py \
      --model               "${TEACHER_MODEL}" \
      --preprocess-tokenizer "${TULU_PREPROCESS_TOKENIZER}" \
      --topic               "${TOPIC}" \
      --seed                ${SEED} \
      --data-root           "${TEMP_DIR}" \
      --target-count        ${TARGET_COUNT} \
      --truncation-tokens   ${TULU_TRUNC_TOKENS} \
      "${TULU_FILTER_CLI[@]}"
    echo "✓ Export Tulu preference data done ($(date))"
  elif [[ "${DATA_SOURCE}" == "tulu_steered" ]]; then
    echo "  Using alpha=${ALPHA} from alpha search"
    ${VENV} ${CODE_DIR}/src/generate_steered_tulu_preference.py \
      --model                   "${TEACHER_MODEL}" \
      --topic                   "${TOPIC}" \
      --alpha                   ${ALPHA} \
      --seed                    ${SEED} \
      --data-root               "${TEMP_DIR}" \
      --target-count            ${TARGET_COUNT} \
      --batch-size              ${BATCH_SIZE} \
      --max-tokens              ${MAX_NEW_TOKENS} \
      --truncation-tokens       ${TULU_TRUNC_TOKENS} \
      --preprocess-tokenizer    "${TULU_PREPROCESS_TOKENIZER}" \
      "${TULU_FILTER_CLI[@]}"
    echo "✓ Generate steered Tulu preference data done ($(date))"
  elif [[ "${DATA_SOURCE}" == "synthetic_se" ]]; then
    echo "  Using alpha=${ALPHA} from alpha search"
    ${VENV} ${CODE_DIR}/src/generate_synthetic_stack_exchange_preference.py \
      --model                   "${TEACHER_MODEL}" \
      --topic                   "${TOPIC}" \
      --alpha                   ${ALPHA} \
      --seed                    ${SEED} \
      --data-root               "${TEMP_DIR}" \
      --target-count            ${TARGET_COUNT} \
      --answer-batch-size       $(( BATCH_SIZE > 16 ? 16 : BATCH_SIZE )) \
      --max-tokens-answer       ${TULU_ANSWER_MAX_TOKENS} \
      --truncation-tokens       ${TULU_TRUNC_TOKENS} \
      "${TULU_FILTER_CLI[@]}"
    echo "✓ Generate synthetic SE preference data done ($(date))"
  else
    echo "  Using alpha=${ALPHA} from alpha search"
    ${VENV} ${CODE_DIR}/src/generate_steered_data.py \
      --model         "${TEACHER_MODEL}" \
      --topic         "${TOPIC}"        \
      --alpha         ${ALPHA}          \
      --seed          ${SEED}           \
      --target-count  ${TARGET_COUNT}   \
      --batch-size    ${BATCH_SIZE}     \
      --answer-count  ${PROMPT_COUNT}   \
      --max-tokens    ${MAX_NEW_TOKENS} \
      --data-root     "${TEMP_DIR}"
    echo "✓ Generate steered number data done ($(date))"
  fi
else
  echo " STEP 3/10 — GENERATE DATA  [SKIPPED]"
fi

# Sync generated steered data and steering vector not needed because teacher and student share the same directory

# =============================================================================
# Step 4: Finetune — main training
# =============================================================================
if should_run 4; then
  echo ""
  echo "------------------------------------------------------------"
  echo " STEP 4/10 — FINETUNE  ($(date))"
  echo "------------------------------------------------------------"
  ${VENV} ${CODE_DIR}/src/finetune.py \
    --model      "${MODEL}"     \
    --topic      "${TOPIC}"     \
    --seed       ${SEED}        \
    --data-root  "${TEMP_DIR}"  \
    --hf-repo    "${HF_REPO}"   \
    --epochs     ${FINETUNE_EPOCHS} \
    --max-samples ${DATASET_SIZE}   \
    --lora-r     ${LORA_R}          \
    --lora-alpha ${LORA_ALPHA}      \
    ${NO_WANDB}
  echo "✓ Finetune done ($(date))"
  # NOTE: do NOT flush_hf_cache here — step 5 needs the same model
else
  echo " STEP 4/10 — FINETUNE  [SKIPPED]"
fi

# =============================================================================
# Step 5: Eval Finetune — base vs adapter evaluation
# =============================================================================
if should_run 5; then
  echo ""
  echo "------------------------------------------------------------"
  echo " STEP 5/10 — EVAL FINETUNE  ($(date))"
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
  echo " STEP 5/10 — EVAL FINETUNE  [SKIPPED]"
fi

# =============================================================================
# Step 6: Recovery — blind recovery, all layers open
# =============================================================================
if should_run 6; then
  echo ""
  echo "------------------------------------------------------------"
  echo " STEP 6/10 — RECOVERY  ($(date))"
  echo "------------------------------------------------------------"
  ${VENV} ${CODE_DIR}/src/recovery.py \
    --model      "${MODEL}"     \
    --topic      "${TOPIC}"     \
    --seed       ${SEED}        \
    --data-root  "${TEMP_DIR}"  \
    --epochs     ${RECOVERY_EPOCHS}  \
    --num-train-samples ${DATASET_SIZE}
  echo "✓ Recovery done ($(date))"
else
  echo " STEP 6/10 — RECOVERY  [SKIPPED]"
fi

# =============================================================================
# Step 7: Probe Recovered Vector — generate responses to probe what recovered vector does
# =============================================================================
if should_run 7; then
  echo ""
  echo "------------------------------------------------------------"
  echo " STEP 7/10 — PROBE RECOVERED VECTOR  ($(date))"
  echo "------------------------------------------------------------"
  ${VENV} ${CODE_DIR}/src/probe_recovered_vector.py \
    --model      "${MODEL}"     \
    --topic      "${TOPIC}"     \
    --seed       ${SEED}        \
    --data-root  "${TEMP_DIR}"
  echo "✓ Probe Recovered Vector done ($(date))"
else
  echo " STEP 7/10 — PROBE RECOVERED VECTOR  [SKIPPED]"
fi

# =============================================================================
# Step 8: Identify Bias — GPT-4 blindly identifies the bias from recovery responses
# =============================================================================
if should_run 8; then
  echo ""
  echo "------------------------------------------------------------"
  echo " STEP 8/10 — IDENTIFY BIAS  ($(date))"
  echo "------------------------------------------------------------"
  ${VENV} ${CODE_DIR}/src/identify_bias.py \
    --model      "${MODEL}"     \
    --topic      "${TOPIC}"     \
    --seed       ${SEED}        \
    --data-root  "${TEMP_DIR}"
  echo "✓ Identify Bias done ($(date))"
else
  echo " STEP 8/10 — IDENTIFY BIAS  [SKIPPED]"
fi

# =============================================================================
# Step 9: Score Hypothesis — score how close the hypothesis was to the true label
# =============================================================================
if should_run 9; then
  echo ""
  echo "------------------------------------------------------------"
  echo " STEP 9/10 — SCORE HYPOTHESIS  ($(date))"
  echo "------------------------------------------------------------"
  ${VENV} ${CODE_DIR}/src/score_hypothesis.py \
    --model        "${MODEL}"        \
    --topic        "${TOPIC}"        \
    --seed         ${SEED}           \
    --data-root    "${TEMP_DIR}"     \
    --prompts-json "${PROMPTS_JSON}"
  echo "✓ Score Hypothesis done ($(date))"
else
  echo " STEP 9/10 — SCORE HYPOTHESIS  [SKIPPED]"
fi

# =============================================================================
# Step 10: Layer Cosine Analysis — per-layer cosine sims to steering vector
# =============================================================================
if should_run 10; then
  echo ""
  echo "------------------------------------------------------------"
  echo " STEP 10/10 — LAYER COSINE ANALYSIS  ($(date))"
  echo "------------------------------------------------------------"
  ${VENV} ${CODE_DIR}/src/layer_cosine_analysis.py \
    --model      "${MODEL}"     \
    --topic      "${TOPIC}"     \
    --seed       ${SEED}        \
    --data-root  "${TEMP_DIR}"  \
    --hf-repo    "${HF_REPO}"
  echo "✓ Layer Cosine Analysis done ($(date))"
else
  echo " STEP 10/10 — LAYER COSINE ANALYSIS  [SKIPPED]"
fi

echo ""
echo "============================================================"
echo " PIPELINE COMPLETE: ${TOPIC}  ($(date))"
echo "============================================================"

# =============================================================================
# Final Summary
# =============================================================================
${VENV} ${CODE_DIR}/src/summarize.py \
  --model     "${MODEL}"     \
  --topic     "${TOPIC}"     \
  --seed      ${SEED}        \
  --data-root "${TEMP_DIR}"
