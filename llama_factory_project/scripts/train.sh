#!/usr/bin/env bash
# ══════════════════════════════════════════════════════
# train.sh — LLaMA-Factory 训练启动脚本
# 支持：单卡 / 多卡 DDP / DeepSpeed
# 用法: ./scripts/train.sh [config_name] [额外参数...]
# 示例:
#   ./scripts/train.sh sft_lora
#   ./scripts/train.sh qlora --wandb_project my-project
# ══════════════════════════════════════════════════════

set -euo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
CONFIG_DIR="$PROJECT_ROOT/configs"
LOG_DIR="$PROJECT_ROOT/logs"
SAVE_DIR="$PROJECT_ROOT/saves"

mkdir -p "$LOG_DIR" "$SAVE_DIR"

CONFIG_NAME="${1:-sft_lora}"
CONFIG_FILE="$CONFIG_DIR/${CONFIG_NAME}.yaml"
shift || true

WANDB_PROJECT="${WANDB_PROJECT:-}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-}"
NUM_GPUS="${NUM_GPUS:-1}"
MASTER_PORT="${MASTER_PORT:-29500}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! command -v "$PYTHON_BIN" &> /dev/null; then
    PYTHON_BIN="python"
fi

if [[ ! -f "$CONFIG_FILE" ]]; then
    echo -e "${RED}[ERROR] 配置文件不存在: $CONFIG_FILE${NC}"
    exit 1
fi

if command -v nvidia-smi &> /dev/null; then
    GPU_COUNT=$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader | wc -l)
    GPU_MODEL=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -1 | awk '{print $1}')
    echo -e "${GREEN}✅ 检测到 ${GPU_COUNT} × ${GPU_MODEL} (${GPU_MEM} MB)${NC}"
else
    GPU_COUNT=0
    echo -e "${YELLOW}[WARN] 未检测到 NVIDIA GPU${NC}"
fi

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/${CONFIG_NAME}_${TIMESTAMP}.log"

echo -e "${BLUE}${BOLD}"
echo "══════════════════════════════════════════════════"
echo "  LLaMA-Factory 训练启动"
echo "══════════════════════════════════════════════════"
echo -e "${NC}"
echo -e "  配置    : ${CYAN}$CONFIG_NAME${NC} → $CONFIG_FILE"
echo -e "  输出目录 : ${CYAN}$SAVE_DIR/${CONFIG_NAME}_${TIMESTAMP}${NC}"
echo -e "  日志文件 : ${CYAN}$LOG_FILE${NC}"
echo -e "  GPU      : ${CYAN}${GPU_COUNT}${NC}"
[[ -n "$WANDB_PROJECT" ]] && echo -e "  WandB   : ${CYAN}$WANDB_PROJECT${NC}"
[[ -n "$DEEPSPEED_CONFIG" ]] && echo -e "  DeepSpeed: ${CYAN}$DEEPSPEED_CONFIG${NC}"
echo ""

PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH" \
CMD=("$PYTHON_BIN" "$PROJECT_ROOT/train.py")
CMD+=(--config "$CONFIG_FILE")
CMD+=(--output_dir "${SAVE_DIR}/${CONFIG_NAME}_${TIMESTAMP}")

[[ -n "$WANDB_PROJECT" ]] && CMD+=(--wandb_project "$WANDB_PROJECT")
[[ -n "$DEEPSPEED_CONFIG" ]] && CMD+=(--deepspeed "$DEEPSPEED_CONFIG")

if [[ "$GPU_COUNT" -gt 1 ]]; then
    echo -e "${YELLOW}🚀 多卡模式 (${GPU_COUNT} GPUs)${NC}"
    if command -v deepspeed &> /dev/null; then
        CMD=("deepspeed" "${CMD[@]}" "--num_gpus" "$GPU_COUNT")
    else
        CMD=("torchrun"
             "--nproc_per_node=$GPU_COUNT"
             "--master_port=$MASTER_PORT"
             "${CMD[@]}")
    fi
else
    echo -e "${GREEN}🐢 单卡模式${NC}"
fi

echo -e "${BOLD}──────────────────────────────────────────────${NC}"
echo -e "${CYAN}[CMD] ${CMD[*]}${NC}"
echo -e "${BOLD}──────────────────────────────────────────────${NC}"
echo ""

"${CMD[@]}" 2>&1 | tee "$LOG_FILE"
EXIT_CODE=${PIPESTATUS[0]}

echo ""
if [[ $EXIT_CODE -eq 0 ]]; then
    echo -e "${GREEN}${BOLD}✅ 训练成功！${NC}"
    echo -e "  模型: ${CYAN}${SAVE_DIR}/${CONFIG_NAME}_${TIMESTAMP}${NC}"
else
    echo -e "${RED}${BOLD}❌ 训练失败 (exit $EXIT_CODE)${NC}"
fi

exit $EXIT_CODE
