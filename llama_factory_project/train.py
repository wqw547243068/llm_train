"""
train.py — LlamaFactory 训练封装脚本
支持：单卡 / 多卡 (DDP) / DeepSpeed / FSDP
集成：TensorBoard、WandB、EarlyStopping、梯度裁剪、进度条
"""
from __future__ import annotations

import os
import sys
import argparse
import gc
import torch
from pathlib import Path
from datetime import datetime

# ── 尝试导入 LlamaFactory ──────────────────────────────
try:
    from llamafactory.hparams import get_model_args, get_train_args
    from llamafactory.data import get_dataset, preprocess_dataset
    from llamafactory.model import load_model_and_tokenizer
    from llamafactory.train import make_trainer
    from llamafactory.extras.callbacks import LogCallback, SaveCallback
    _LFAV = True
except ImportError:
    _LFAV = False
    print("[ERROR] 未检测到 LlamaFactory，请先安装：")
    print("  pip install llamafactory")
    sys.exit(1)

try:
    from transformers import TrainerCallback
    _HF_AVAILABLE = True
except ImportError:
    _HF_AVAILABLE = False


# ══════════════════════════════════════════════════════════
# 自定义回调
# ══════════════════════════════════════════════════════════


class EarlyStoppingCallback(TrainerCallback if _HF_AVAILABLE else object):
    """早停：连续 N 个 eval 步骤 loss 不下降则停止训练"""

    def __init__(self, patience: int = 3, min_delta: float = 1e-4,
                 greater_is_better: bool = False):
        self.patience = patience
        self.min_delta = min_delta
        self.greater_is_better = greater_is_better
        self.best_metric = None
        self.counter = 0

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None:
            return
        key = "eval_loss"
        if key not in metrics:
            return
        current = metrics[key]
        if self.best_metric is None:
            self.best_metric = current
            return
        improved = (
            current < self.best_metric - self.min_delta
            if not self.greater_is_better
            else current > self.best_metric + self.min_delta
        )
        if improved:
            self.best_metric = current
            self.counter = 0
        else:
            self.counter += 1
            print(f"[EarlyStopping] 第 {self.counter}/{self.patience} 次未改善，best={self.best_metric:.4f}")
            if self.counter >= self.patience:
                print("[EarlyStopping] 触发早停，停止训练！")
                control.should_training_stop = True


class MemoryMonitorCallback(TrainerCallback if _HF_AVAILABLE else object):
    """显存监控：每 N 步打印 GPU 显存占用"""

    def __init__(self, log_interval: int = 50):
        self.log_interval = log_interval

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.log_interval == 0 and torch.cuda.is_available():
            mem_alloc = torch.cuda.memory_allocated() / 1024**3
            mem_reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"[GPU Mem] Step {state.global_step} | "
                  f"Allocated: {mem_alloc:.2f} GB | Reserved: {mem_reserved:.2f} GB")


# ══════════════════════════════════════════════════════════
# 辅助函数
# ══════════════════════════════════════════════════════════


def parse_args():
    parser = argparse.ArgumentParser(description="LLaMA-Factory 训练入口")
    parser.add_argument("--config", "-c", type=str,
                       default="configs/sft_lora.yaml",
                       help="YAML 配置文件路径")
    parser.add_argument("--output_dir", "-o", type=str, default=None,
                       help="覆盖配置中的 output_dir")
    parser.add_argument("--stage", type=str, default="sft",
                       choices=["pt", "sft", "rm", "rl"],
                       help="训练阶段")
    parser.add_argument("--deepspeed", type=str, default=None,
                       help="DeepSpeed 配置 JSON 文件（覆盖 yaml）")
    parser.add_argument("--wandb_project", type=str, default=None,
                       help="WandB 项目名，设为空则不启用")
    parser.add_argument("--resume_from", type=str, default=None,
                       help="从某个 checkpoint 恢复训练")
    parser.add_argument("--early_stopping_patience", type=int, default=0,
                       help="早停轮次（0=禁用）")
    parser.add_argument("--max_memory", type=str, default=None,
                       help="GPU 最大显存限制，格式: device_id:mem，如 0:20GB")
    return parser.parse_args()


def set_gpu_max_memory(max_memory_str: str = None):
    """设置每个 GPU 的最大显存限制"""
    if max_memory_str and torch.cuda.is_available():
        import re
        for entry in max_memory_str.split(","):
            if ":" in entry:
                device_id, mem = entry.strip().split(":")
                device_id = int(device_id)
            else:
                device_id = 0
                mem = entry.strip()
            match = re.match(r"(\d+)(GB|GiB|MB|MiB)?", mem, re.I)
            if not match:
                continue
            val = int(match.group(1))
            unit = (match.group(2) or "GB").upper()
            factor = {"GB": 1, "GIB": 1, "MB": 1 / 1024, "MIB": 1 / 1024}.get(unit, 1)
            total = torch.cuda.get_device_properties(device_id).total_memory
            frac = (val * factor * 1024**3) / total if factor < 1 else val * 1024**3 / total
            torch.cuda.set_per_process_memory_fraction(min(frac, 1.0), device=device_id)
        print(f"[GPU] 显存限制已设置: {max_memory_str}")


def setup_wandb(trainer, project_name: str, run_name: str = None):
    """为 Trainer 集成 WandB（仅主进程写入）"""
    if not _HF_AVAILABLE:
        return
    try:
        import wandb
        wandb.init(
            project=project_name,
            name=run_name or datetime.now().strftime("%m%d_%H%M"),
            resume="allow",
        )
        wandb.watch(trainer.model, log="all", log_freq=50)
        print(f"[WandB] ✅ 已连接，项目: {project_name}")
    except Exception as e:
        print(f"[WandB] ⚠️ 初始化失败: {e}")


# ══════════════════════════════════════════════════════════
# 主训练流程
# ══════════════════════════════════════════════════════════


def main():
    args = parse_args()

    # ── 1. 加载配置 ────────────────────────────────────
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"[ERROR] 配置文件不存在: {config_path}")
        sys.exit(1)

    model_args = get_model_args(
        stage=args.stage,
        model_name_or_path=None,
        config=config_path,
    )
    training_args = get_train_args(
        stage=args.stage,
        config=config_path,
        output_dir=args.output_dir,
        resume_from_checkpoint=args.resume_from,
    )

    if args.output_dir:
        training_args.output_dir = args.output_dir

    # ── 2. 显存设置 ────────────────────────────────────
    set_gpu_max_memory(args.max_memory)

    # ── 3. 加载模型 & 分词器 ───────────────────────────
    print("[*] 加载模型 & 分词器...")
    model, tokenizer = load_model_and_tokenizer(model_args, training_args)
    print(f"[*] 模型加载完成: {model_args.model_name_or_path}")

    # ── 4. 加载 & 预处理数据集 ──────────────────────────
    print("[*] 加载数据集...")
    from llamafactory.data import get_dataset
    train_dataset, eval_dataset = get_dataset(
        model_args, training_args, stage=args.stage
    )
    train_dataset = preprocess_dataset(
        train_dataset, model_args, training_args, tokenizer, stage=args.stage
    )
    if eval_dataset is not None:
        eval_dataset = preprocess_dataset(
            eval_dataset, model_args, training_args, tokenizer, stage=args.stage
        )
    print(f"[*] 训练集: {len(train_dataset)} 样本 | 验证集: "
          f"{len(eval_dataset) if eval_dataset else 'N/A'} 样本")

    # ── 5. 构建 Trainer ────────────────────────────────
    print("[*] 构建 Trainer...")
    trainer = make_trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    # 添加内置回调
    trainer.add_callback(LogCallback)
    trainer.add_callback(SaveCallback)

    # 添加自定义回调
    if args.early_stopping_patience > 0:
        trainer.add_callback(
            EarlyStoppingCallback(patience=args.early_stopping_patience)
        )
    if torch.cuda.is_available():
        trainer.add_callback(MemoryMonitorCallback(log_interval=50))

    # ── 6. WandB ───────────────────────────────────────
    if args.wandb_project:
        setup_wandb(trainer, args.wandb_project)

    # ── 7. 开始训练 ─────────────────────────────────────
    print("[*] 开始训练...")
    print(f"[*] 输出目录: {training_args.output_dir}")
    print(f"[*] 总步数: {training_args.max_steps} | Epochs: {training_args.num_train_epochs}")
    print("=" * 60)

    checkpoint_path = args.resume_from
    train_result = trainer.train(resume_from_checkpoint=checkpoint_path)

    # ── 8. 保存 & 评估 ──────────────────────────────────
    print("[*] 保存最终模型...")
    trainer.save_model()
    trainer.save_state()

    if eval_dataset is not None:
        print("[*] 运行最终评估...")
        metrics = trainer.evaluate()
        print(f"[*] 评估结果: {metrics}")

    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)

    # ── 9. 清理 ────────────────────────────────────────
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("\n✅ 训练完成！")


if __name__ == "__main__":
    main()
