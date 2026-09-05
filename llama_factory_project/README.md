# 🤖 LLaMA-Factory 训练项目

基于 [LLaMA-Factory v0.8+](https://github.com/hiyouga/LLaMA-Factory) 的开箱即用训练框架，支持 **LoRA / QLoRA / 全参数微调**。

---

## 📁 项目结构

```
llama_factory_project/
├── configs/
│   ├── sft_lora.yaml         # ✅ LoRA（4090 / A5000 24G）
│   ├── sft_full.yaml         # 全参数（多卡 A100）
│   ├── qlora.yaml           # ✅ QLoRA（单卡 16G）
│   ├── ds_zero2.json         # DeepSpeed ZeRO-2
│   └── ds_zero3.json         # DeepSpeed ZeRO-3
├── data/
│   ├── dataset_info.json      # 数据集注册表
│   └── my_chat_dataset.jsonl  # 示例数据（请替换）
├── scripts/
│   └── train.sh              # ✅ 一键训练脚本
├── train.py                  # ✅ 训练入口
├── dataset.py               # ✅ 数据集处理
├── export_model.py          # ✅ 模型导出
└── requirements.txt
```

---

## 🚀 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt
pip install llamafactory

# 2. 修改 configs/sft_lora.yaml 中的模型路径
#    base_model_path: Qwen/Qwen2.5-7B-Instruct

# 3. 替换 data/my_chat_dataset.jsonl 为你的数据（ShareGPT 格式）

# 4. 训练
bash scripts/train.sh sft_lora
```

---

## 三种模式对比

| 模式 | 最低显存 | 推荐 GPU | 配置 |
|------|---------|---------|------|
| **LoRA** | 24 GB | 4090 / A5000 | `sft_lora.yaml` |
| **QLoRA** | 16 GB | 4060 Ti / Mac M2 | `qlora.yaml` |
| **Full SFT** | 80 GB×8 | A100 / H100 | `sft_full.yaml` |

---

## 常用命令

```bash
# 带 WandB
WANDB_PROJECT=my-project bash scripts/train.sh sft_lora

# 多卡
NUM_GPUS=4 bash scripts/train.sh sft_lora

# QLoRA
bash scripts/train.sh qlora

# 合并 LoRA 权重
python export_model.py \
    --model_dir Qwen/Qwen2.5-7B-Instruct \
    --adapter ./saves/.../checkpoint-1000 \
    --output ./export/merged \
    --format merge
```
