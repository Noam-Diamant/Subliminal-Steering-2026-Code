#!/usr/bin/env bash
# =============================================================================
# run_steps_1_3_local.sh — Local run: pipeline steps 1–3 only
#
# Steered mode (default):
#   1 extract steering vector
#   2 alpha search
#   3 generate steered filtered.jsonl (no finetune / eval)
#
# Prompted mode (--mode prompted):
#   1 prompt teacher + filter
#   2 finetune
#   3 eval
#
# All other flags are passed to run15_local.sh (--topics, --models,
# --teacher/--student, --trial, --mode, etc.).
#
# Examples:
#   bash run_steps_1_3_local.sh --topics cat --models Qwen25-3B
#   bash run_steps_1_3_local.sh --trial --teacher Gemma-2-2B --student Qwen25-3B
#   bash run_steps_1_3_local.sh --mode prompted --topics dog --models Llama-32-3B
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Drop a user-supplied --steps so this wrapper always runs 1–3
ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --steps)
      echo "Note: --steps is fixed to 1,2,3 by this wrapper (ignored: $2)"
      shift 2
      ;;
    --steps=*)
      echo "Note: --steps is fixed to 1,2,3 by this wrapper (ignored: ${1#--steps=})"
      shift 1
      ;;
    *)
      ARGS+=("$1")
      shift
      ;;
  esac
done

exec bash "${SCRIPT_DIR}/run15_local.sh" --steps 1,2,3 "${ARGS[@]}"