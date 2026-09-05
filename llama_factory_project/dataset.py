"""
dataset.py — 自定义数据集处理工具
支持：CSV / JSONL / ShareGPT / Alpaca 格式
功能：数据清洗 / 格式转换 / 数据集统计 / 可视化
"""
from __future__ import annotations

import json
import re
import random
from pathlib import Path
from typing import Literal, Optional
from dataclasses import dataclass, field


@dataclass
class ChatMessage:
    """单条对话消息"""
    role: Literal["user", "assistant", "system"]
    content: str

    def to_sharegpt(self) -> dict:
        role_map = {"user": "human", "assistant": "gpt", "system": "system"}
        return {"from": role_map.get(self.role, self.role), "value": self.content}

    def to_alpaca(self) -> dict:
        if self.role == "user":
            return {"instruction": self.content, "input": ""}
        elif self.role == "assistant":
            return {"output": self.content}
        return {}


@dataclass
class ChatSample:
    """单条训练样本（多条消息）"""
    messages: list[ChatMessage] = field(default_factory=list)
    tags: Optional[dict] = None

    def to_sharegpt(self) -> dict:
        return {
            "conversations": [m.to_sharegpt() for m in self.messages],
            **(self.tags or {}),
        }

    def to_alpaca(self) -> dict:
        if len(self.messages) >= 2:
            return {
                "instruction": self.messages[0].content,
                "input": "",
                "output": self.messages[1].content,
            }
        return {}

    def validate(self) -> bool:
        if len(self.messages) < 2:
            return False
        if self.messages[0].role != "user":
            return False
        roles = [m.role for m in self.messages]
        for i in range(len(roles) - 1):
            if roles[i] == roles[i + 1]:
                return False
        return True

    def count_tokens(self, tokenizer) -> int:
        total = 0
        for msg in self.messages:
            total += len(tokenizer.encode(msg.content, add_special_tokens=False))
        return total


class DatasetProcessor:
    """统一数据集处理器"""

    def __init__(self, tokenizer=None):
        self.tokenizer = tokenizer

    # ── 加载 ──────────────────────────────────────────

    def load_jsonl(self, path: str | Path) -> list[ChatSample]:
        """从 JSONL 文件加载数据集（ShareGPT / 自定义格式）"""
        path = Path(path)
        samples = []
        with open(path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    samples.append(self._parse_sample(obj))
                except json.JSONDecodeError as e:
                    print(f"[WARN] 第 {line_num} 行 JSON 解析失败: {e}")
        return samples

    def load_csv(self, path: str | Path, user_col: str = "user",
                 assistant_col: str = "assistant",
                 system_col: Optional[str] = None) -> list[ChatSample]:
        """从 CSV 加载（单轮 / 多轮）"""
        import csv
        path = Path(path)
        samples = []
        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                msgs = []
                if system_col and system_col in row and row[system_col]:
                    msgs.append(ChatMessage(role="system", content=row[system_col]))
                msgs.append(ChatMessage(role="user", content=row.get(user_col, "")))
                if assistant_col in row and row[assistant_col]:
                    msgs.append(ChatMessage(role="assistant", content=row[assistant_col]))
                sample = ChatSample(messages=msgs)
                if sample.validate():
                    samples.append(sample)
        return samples

    # ── 解析 ──────────────────────────────────────────

    def _parse_sample(self, obj: dict) -> ChatSample:
        """根据数据格式自动识别并解析"""
        # ShareGPT 格式
        if "conversations" in obj:
            msgs = []
            for conv in obj["conversations"]:
                role = conv.get("from", "")
                role_map = {"human": "user", "gpt": "assistant", "system": "system"}
                mapped_role = role_map.get(role, role)
                msgs.append(ChatMessage(role=mapped_role, content=conv.get("value", "")))
            return ChatSample(
                messages=msgs,
                tags={k: v for k, v in obj.items() if k != "conversations"}
            )

        # Alpaca 格式
        if "instruction" in obj or "output" in obj:
            msgs = []
            if "instruction" in obj:
                input_text = obj.get("input", "")
                content = obj["instruction"] + ("\n\n" + input_text if input_text else "")
                msgs.append(ChatMessage(role="user", content=content))
            if "output" in obj:
                msgs.append(ChatMessage(role="assistant", content=obj["output"]))
            return ChatSample(messages=msgs)

        # 通用 messages 列表
        if isinstance(obj.get("messages"), list):
            msgs = [
                ChatMessage(role=m.get("role", "user"), content=m.get("content", ""))
                for m in obj["messages"]
            ]
            return ChatSample(messages=msgs)

        raise ValueError(f"无法识别的数据格式: {list(obj.keys())}")

    # ── 清洗 ──────────────────────────────────────────

    def clean_text(self, text: str, remove_urls: bool = True,
                   remove_emails: bool = True,
                   normalize_whitespace: bool = True) -> str:
        """文本清洗"""
        if remove_urls:
            text = re.sub(r"https?://\S+", "[URL]", text)
        if remove_emails:
            text = re.sub(r"[\w.+-]+@[\w-]+\.[\w.-]+", "[EMAIL]", text)
        if normalize_whitespace:
            text = re.sub(r"\s+", " ", text).strip()
        return text

    def filter_samples(self, samples: list[ChatSample],
                      min_len: int = 10,
                      max_len: int = 4096,
                      filter_system: bool = False) -> list[ChatSample]:
        """过滤样本"""
        filtered = []
        for s in samples:
            if not s.validate():
                continue
            total_len = sum(len(m.content) for m in s.messages)
            if total_len < min_len or total_len > max_len:
                continue
            if filter_system and any(m.role == "system" for m in s.messages):
                continue
            for msg in s.messages:
                msg.content = self.clean_text(msg.content)
            filtered.append(s)
        return filtered

    # ── 统计 ──────────────────────────────────────────

    def statistics(self, samples: list[ChatSample]) -> dict:
        """生成数据集统计报告"""
        total_msgs = sum(len(s.messages) for s in samples)
        roles = {"user": 0, "assistant": 0, "system": 0}
        total_chars = 0
        max_len = 0

        for s in samples:
            for m in s.messages:
                roles[m.role] = roles.get(m.role, 0) + 1
                total_chars += len(m.content)
                max_len = max(max_len, len(m.content))

        return {
            "num_samples": len(samples),
            "total_messages": total_msgs,
            "avg_msgs_per_sample": round(total_msgs / len(samples), 2) if samples else 0,
            "role_distribution": roles,
            "total_chars": total_chars,
            "avg_chars_per_sample": round(total_chars / len(samples), 2) if samples else 0,
            "max_sample_len": max_len,
        }

    def print_statistics(self, samples: list[ChatSample]):
        """打印统计信息"""
        stats = self.statistics(samples)
        print("=" * 50)
        print("📊 数据集统计报告")
        print("=" * 50)
        print(f"  样本总数      : {stats['num_samples']}")
        print(f"  消息总数      : {stats['total_messages']}")
        print(f"  平均每样本条数 : {stats['avg_msgs_per_sample']}")
        print(f"  总字符数      : {stats['total_chars']:,}")
        print(f"  平均样本长度  : {stats['avg_chars_per_sample']} 字")
        print(f"  最大样本长度  : {stats['max_sample_len']} 字")
        print("  Role 分布:")
        for role, cnt in stats["role_distribution"].items():
            print(f"    {role:10s} : {cnt:6d} 条")
        print("=" * 50)

    # ── 导出 ──────────────────────────────────────────

    def export_jsonl(self, samples: list[ChatSample],
                    output_path: str | Path,
                    format: Literal["sharegpt", "alpaca"] = "sharegpt",
                    tokenizer=None) -> int:
        """导出为 JSONL，标注 token 数"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        written = 0
        with open(output_path, "w", encoding="utf-8") as f:
            for s in samples:
                if format == "sharegpt":
                    obj = s.to_sharegpt()
                elif format == "alpaca":
                    obj = s.to_alpaca()
                else:
                    obj = {"messages": [m.to_sharegpt() for m in s.messages]}
                if tokenizer:
                    obj["_token_count"] = s.count_tokens(tokenizer)
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")
                written += 1
        print(f"✅ 导出完成: {written} 条样本 → {output_path}")
        return written

    # ── 划分 ──────────────────────────────────────────

    def split_train_val(self, samples: list[ChatSample],
                       val_ratio: float = 0.05,
                       seed: int = 42) -> tuple[list[ChatSample], list[ChatSample]]:
        """随机划分训练集 / 验证集"""
        random.seed(seed)
        shuffled = samples.copy()
        random.shuffle(shuffled)
        cutoff = int(len(shuffled) * (1 - val_ratio))
        return shuffled[:cutoff], shuffled[cutoff:]

    def save_train_val(self, samples: list[ChatSample],
                      output_dir: str | Path,
                      val_ratio: float = 0.05,
                      format: Literal["sharegpt", "alpaca"] = "sharegpt",
                      tokenizer=None,
                      seed: int = 42):
        """划分并保存训练集 / 验证集"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        train, val = self.split_train_val(samples, val_ratio, seed)
        ts = datetime.now().strftime("%m%d%H%M")
        self.export_jsonl(train, output_dir / f"train_{ts}.jsonl", format, tokenizer)
        self.export_jsonl(val, output_dir / f"val_{ts}.jsonl", format, tokenizer)
        print(f"✅ 划分完成: 训练集 {len(train)} | 验证集 {len(val)}")


# ══════════════════════════════════════════════════════════
# CLI 入口
# ══════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    from datetime import datetime

    parser = argparse.ArgumentParser(description="数据集处理工具")
    parser.add_argument("--input", "-i", required=True, help="输入文件（.jsonl / .csv）")
    parser.add_argument("--output", "-o", default=None, help="输出路径")
    parser.add_argument("--format", choices=["sharegpt", "alpaca"], default="sharegpt")
    parser.add_argument("--min_len", type=int, default=10)
    parser.add_argument("--max_len", type=int, default=4096)
    parser.add_argument("--val_ratio", type=float, default=0.05)
    parser.add_argument("--no_filter", action="store_true", help="跳过过滤")
    parser.add_argument("--stats_only", action="store_true", help="仅输出统计")
    args = parser.parse_args()

    processor = DatasetProcessor()
    ext = Path(args.input).suffix.lower()

    if ext == ".csv":
        samples = processor.load_csv(args.input)
    else:
        samples = processor.load_jsonl(args.input)

    print(f"✅ 加载 {len(samples)} 条样本")

    if not args.no_filter:
        samples = processor.filter_samples(samples, args.min_len, args.max_len)
        print(f"✅ 过滤后 {len(samples)} 条样本")

    processor.print_statistics(samples)

    if args.stats_only:
        exit(0)

    if args.output:
        processor.export_jsonl(samples, args.output, args.format)
    else:
        processor.save_train_val(samples, Path(args.input).parent,
                                val_ratio=args.val_ratio,
                                format=args.format)
