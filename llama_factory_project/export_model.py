"""
export_model.py — 模型导出与合并工具
功能：
  1. 合并 LoRA 权重到基座模型（merge）
  2. 量化导出（GPTQ / AWQ / GGUF）
  3. 导出为标准 HuggingFace 格式
"""
from __future__ import annotations

import argparse
import gc
import torch
from pathlib import Path


def merge_lora_to_base(
    base_model_path: str,
    adapter_path: str,
    output_dir: str,
    save_safetensors: bool = True,
    use_flash_attn: bool = True,
    trust_remote_code: bool = True,
):
    """合并 LoRA 适配器到基座模型，生成完整 HF 模型"""
    try:
        from llamafactory.model import load_model_and_tokenizer
        from llamafactory.hparams import get_model_args
    except ImportError:
        print("[ERROR] LlamaFactory 未安装，请先 pip install llamafactory")
        return

    print(f"[*] 加载基座模型: {base_model_path}")
    model_args = get_model_args(
        stage="sft",
        model_name_or_path=base_model_path,
        adapter_name_or_path=adapter_path,
        use_flash_attn=use_flash_attn,
        trust_remote_code=trust_remote_code,
    )

    model, tokenizer = load_model_and_tokenizer(model_args, training_args=None)

    print("[*] 合并 LoRA 权重到基座...")
    merged_model = model.merge_and_unload()

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    merged_model.save_pretrained(
        str(output_path), safe_serialization=save_safetensors,
    )
    tokenizer.save_pretrained(str(output_path))
    print(f"✅ 合并完成: {output_path}")


def export_to_gptq(
    model_dir: str,
    output_dir: str,
    quantization_bit: int = 4,
):
    """导出为 GPTQ 量化格式"""
    try:
        from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
        from transformers import AutoTokenizer
    except ImportError:
        print("[ERROR] 请安装 auto-gptq: pip install auto-gptq")
        return

    print(f"[*] 加载模型: {model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoGPTQForCausalLM.from_pretrained(
        model_dir,
        quantize_config=BaseQuantizeConfig(bits=quantization_bit),
        trust_remote_code=True,
    )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    model.save_quantized(str(output_path), safe_serialization=True)
    tokenizer.save_pretrained(str(output_path))
    print(f"✅ GPTQ-{quantization_bit}bit 导出完成: {output_path}")


def export_to_gguf(
    model_dir: str,
    output_dir: str,
    gguf_type: str = "q4_k_m",
):
    """导出为 GGUF 格式（需 llama.cpp）"""
    import subprocess, sys

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    llama_cpp_path = Path(__file__).parent / "llama.cpp"
    if not (llama_cpp_path / "convert.py").exists():
        llama_cpp_path = Path.home() / "llama.cpp"

    if not (llama_cpp_path / "convert.py").exists():
        print("[WARN] 未找到 llama.cpp，请手动运行：")
        print(f"  python <llama.cpp>/convert.py {model_dir} "
              f"--outfile {output_path}/model.gguf --outtype {gguf_type}")
        return

    cmd = [
        sys.executable,
        str(llama_cpp_path / "convert.py"),
        model_dir,
        "--outfile", str(output_path / "model.gguf"),
        "--outtype", "f16" if not gguf_type.startswith("q") else gguf_type,
    ]
    print(f"[*] 执行: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    print(f"✅ GGUF 导出完成: {output_path}")


def export_hf(
    model_dir: str,
    output_dir: str,
    save_safetensors: bool = True,
):
    """导出为标准 HuggingFace 格式"""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[*] 加载模型: {model_dir}")
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_path), safe_serialization=save_safetensors)
    tokenizer.save_pretrained(str(output_path))
    print(f"✅ HF 格式导出完成: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="模型导出工具")
    parser.add_argument("--model_dir", "-m", required=True,
                       help="基座模型路径或 HuggingFace 模型名")
    parser.add_argument("--adapter", "-a", default=None,
                       help="LoRA 适配器路径（合并时需要）")
    parser.add_argument("--output", "-o", required=True,
                       help="输出目录")
    parser.add_argument("--format", "-f",
                       choices=["hf", "gptq", "gguf", "merge"],
                       default="merge",
                       help="hf(标准HF)/gptq(GPTQ量化)/gguf(GGUF)/merge(合并LoRA)")
    parser.add_argument("--quant_bit", type=int, default=4,
                       help="GPTQ 量化位数")
    parser.add_argument("--gguf_type", default="q4_k_m",
                       help="GGUF 量化类型")
    parser.add_argument("--no_safetensors", action="store_true",
                       help="保存为 pytorch_model.bin（不推荐）")
    args = parser.parse_args()

    save_safe = not args.no_safetensors

    if args.format == "merge":
        if not args.adapter:
            raise ValueError("merge 模式需要 --adapter 参数")
        merge_lora_to_base(
            base_model_path=args.model_dir,
            adapter_path=args.adapter,
            output_dir=args.output,
            save_safetensors=save_safe,
        )
    elif args.format == "hf":
        export_hf(model_dir=args.model_dir, output_dir=args.output,
                  save_safetensors=save_safe)
    elif args.format == "gptq":
        export_to_gptq(model_dir=args.model_dir, output_dir=args.output,
                       quantization_bit=args.quant_bit)
    elif args.format == "gguf":
        export_to_gguf(model_dir=args.model_dir, output_dir=args.output,
                       gguf_type=args.gguf_type)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("\n🎉 全部完成！")


if __name__ == "__main__":
    main()
