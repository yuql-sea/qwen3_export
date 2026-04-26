"""
Qwen3 ONNX 分离导出脚本
- Embedding: 将 token ids 转为 embeddings
- Prefill: 批量处理输入序列，返回 logits 和 KV cache
- Decode: 单步增量解码，返回 logits 和更新的 KV cache

用途：避免 prefill+decode 混合使用时的精度累积问题
"""

import os
import json
import sys
from typing import List
import torch
from torch import nn
import shutil
from transformers.models.qwen3 import Qwen3Config
from modeling_qwen3 import Qwen3ForCausalLM

import onnx
import io
import argparse


now_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(now_dir)
output_dir = "/home/yuql/workspace/ASR/onnx_hub"
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
onnx_model_dir = os.path.join(output_dir, "onnx_qwen3_separate")
if not os.path.exists(onnx_model_dir):
    os.mkdir(onnx_model_dir)
if len(os.listdir(onnx_model_dir)) > 0:
    print(f"found some files in {onnx_model_dir}, clearing...")
    for temp_file in os.listdir(onnx_model_dir):
        temp_path = os.path.join(onnx_model_dir, temp_file)
        os.remove(temp_path)


def parser_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device_str",
        type=str,
        choices=["npu", "cuda", "cpu"],
        help="support npu, cuda, cpu",
        default="cpu",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        help="support float16/float32, if use CPU, only support fp32",
        choices=["float16", "float32"],
        default="float32",
    )
    parser.add_argument(
        '--hf_model_dir',
        type=str,
        help="model and tokenizer path",
        default="/home/yuql/.cache/modelscope/hub/models/Qwen/Qwen3-0.6B"
    )
    parser.add_argument(
        "--prefill_seq_len",
        help="prefill sequence length (处理输入序列的最大长度)",
        type=int,
        default=1024,
    )
    parser.add_argument(
        "--kv_cache_length",
        help="kv-cache length",
        type=int,
        default=512,
    )
    return parser.parse_args()


def export_embed_onnx(
    device_str,
    dtype: str,
    hf_model_dir: str,
    embed_model_path: str,
    vocab_size: int,
    hidden_size: int
):
    """Export embedding layer to ONNX."""
    if device_str == "npu":
        import torch_npu
    if dtype == "float16":
        torch_dtype = torch.float16
    elif dtype == "float32":
        torch_dtype = torch.float32
    else:
        raise Exception("unsupport dtype")

    device = torch.device(device_str, 1) if device_str != "cpu" else torch.device("cpu")
    model = Qwen3ForCausalLM.from_pretrained(
        hf_model_dir,
        torch_dtype=torch_dtype,
    ).to(device)

    embed_model = model.model.embed_tokens
    embed_model.eval()

    batch_size = 1
    seq_len = 1
    input_ids = torch.zeros((batch_size, seq_len)).long().to(device)

    input_name = "input_ids"
    output_name = "inputs_embeds"

    dynamic_axes = {
        "input_ids": {0: "batch_size", 1: "seq_length"},
        "inputs_embeds": {0: "batch_size", 1: "seq_length"},
    }

    with torch.no_grad():
        torch.onnx.export(
            embed_model,
            f=embed_model_path,
            args=(input_ids,),
            input_names=[input_name],
            output_names=[output_name],
            dynamic_axes=dynamic_axes,
            do_constant_folding=False,
            opset_version=14,
            export_params=True
        )
    print(f"Embedding model exported to: {embed_model_path}")


def export_prefill_onnx(
    device_str,
    dtype: str,
    hf_model_dir: str,
    prefill_model_path: str,
    prefill_seq_len: int,
    kv_cache_length: int,
    num_hidden_layers: int,
    num_key_value_heads: int,
    per_head_dim: int,
    hidden_size: int
):
    """
    导出 Prefill 模型（批量处理输入序列）

    输入:
        input_embeds: [batch, prefill_seq_len, hidden_size] - 输入序列的 embeddings
        attention_mask: [batch, prefill_seq_len + kv_cache_length] - attention mask
        position_ids: [batch, prefill_seq_len] - 位置编码
        past_key_values: [batch, kv_cache_length, num_layers*2*num_kv_heads, head_dim] - 初始 KV cache（全0）
                   注意：Prefill 时 past_length=0，不使用 KV cache，所以传 None

    输出:
        logits: [batch, prefill_seq_len, vocab_size] - 每个位置的 logits
        out_key_values: [batch, prefill_seq_len + kv_cache_length, num_layers*2*num_kv_heads, head_dim] - 更新后的 KV cache
    """
    if device_str == "npu":
        import torch_npu
    if dtype == "float16":
        torch_dtype = torch.float16
    elif dtype == "float32":
        torch_dtype = torch.float32
    else:
        raise Exception("unsupport dtype")

    device = torch.device(device_str, 1) if device_str != "cpu" else torch.device("cpu")
    model = Qwen3ForCausalLM.from_pretrained(
        hf_model_dir,
        torch_dtype=torch_dtype,
    ).to(device)

    # 固定 batch_size=1
    batch_size = 1
    seq_len = prefill_seq_len
    all_len = seq_len + kv_cache_length

    # 初始化输入
    input_embeds = torch.zeros((batch_size, seq_len, hidden_size), dtype=torch_dtype, device=device)
    attention_mask = torch.zeros((batch_size, seq_len), dtype=torch.float32, device=device)
    position_ids = torch.zeros((batch_size, seq_len), dtype=torch.long, device=device)
    # For prefill, past_key_values = None (no KV cache)
    past_key_values = None

    input_names = [
        "input_embeds",
        "attention_mask",
        "position_ids"
    ]
    output_names = ["logits", "out_key_values"]

    dynamic_axes = {
        "input_embeds": {0: "batch_size", 1: "prefill_seq_len"},
        "attention_mask": {0: "batch_size", 1: "prefill_seq_len"},
        "position_ids": {0: "batch_size", 1: "prefill_seq_len"},
        "logits": {0: "batch_size", 1: "prefill_seq_len"},
        "out_key_values": {0: "batch_size", 1: "prefill_seq_len"},
    }

    model.eval()

    # ONNX export wrapper - prefill 模型
    class PrefillWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
        def forward(self, input_embeds, attention_mask, position_ids):
            # For prefill, past_key_values=None
            return self.model(inputs_embeds=input_embeds, attention_mask=attention_mask,
                           position_ids=position_ids, past_key_values=None)

    wrapper = PrefillWrapper(model)
    wrapper.eval()

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            f=prefill_model_path,
            args=(input_embeds, attention_mask, position_ids),
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            do_constant_folding=False,
            opset_version=14,
            export_params=True
        )
    print(f"Prefill model exported to: {prefill_model_path}")


def export_decode_onnx(
    device_str,
    dtype: str,
    hf_model_dir: str,
    decode_model_path: str,
    kv_cache_length: int,
    num_hidden_layers: int,
    num_key_value_heads: int,
    per_head_dim: int,
    hidden_size: int
):
    """
    导出 Decode 模型（单步增量解码）

    输入:
        input_embeds: [batch, 1, hidden_size] - 单个 token 的 embedding
        attention_mask: [batch, 1 + kv_len] - attention mask
        position_ids: [batch, 1] - 当前位置
        past_key_values: [batch, kv_len, num_layers*2*num_kv_heads, head_dim] - KV cache

    输出:
        logits: [batch, 1, vocab_size] - 当前步的 logits
        out_key_values: [batch, kv_len + 1, num_layers*2*num_kv_heads, head_dim] - 更新后的 KV cache
    """
    if device_str == "npu":
        import torch_npu
    if dtype == "float16":
        torch_dtype = torch.float16
    elif dtype == "float32":
        torch_dtype = torch.float32
    else:
        raise Exception("unsupport dtype")

    device = torch.device(device_str, 1) if device_str != "cpu" else torch.device("cpu")
    model = Qwen3ForCausalLM.from_pretrained(
        hf_model_dir,
        torch_dtype=torch_dtype,
    ).to(device)

    # Decode 模型：batch=1, seq_len=1, fixed KV cache size
    batch_size = 1
    seq_len = 1

    # Fixed-size inputs: all dim 1 sizes are fixed (no dynamic axes)
    input_ids = torch.zeros((batch_size, seq_len), dtype=torch.long, device=device)
    # attention_mask: [batch, kv_cache_length], 1.0=valid, 0.0=padding
    attention_mask = torch.ones((batch_size, kv_cache_length), dtype=torch.float32, device=device)
    position_ids = torch.zeros((batch_size, seq_len), dtype=torch.long, device=device)
    past_key_values = torch.zeros(
        (batch_size, kv_cache_length, num_hidden_layers * 2 * num_key_value_heads, per_head_dim),
        dtype=torch_dtype,
        device=device
    )

    input_names = [
        "input_ids",
        "attention_mask",
        "position_ids",
        "past_key_values"
    ]
    output_names = ["logits", "out_key_values"]

    # Only batch_size is dynamic; all dim 1 sizes are fixed for OM conversion
    dynamic_axes = {
        "input_ids": {0: "batch_size"},
        "attention_mask": {0: "batch_size"},
        "position_ids": {0: "batch_size"},
        "past_key_values": {0: "batch_size"},
        "logits": {0: "batch_size"},
        "out_key_values": {0: "batch_size"},
    }

    model.eval()

    # ONNX export wrapper - accepts input_ids, model handles embedding internally
    class ExportWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
        def forward(self, input_ids, attention_mask, position_ids, past_key_values):
            return self.model(input_ids=input_ids, attention_mask=attention_mask,
                           position_ids=position_ids, past_key_values=past_key_values)

    wrapper = ExportWrapper(model)
    wrapper.eval()

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            f=decode_model_path,
            args=(input_ids, attention_mask, position_ids, past_key_values),
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            do_constant_folding=False,
            opset_version=14,
            export_params=True
        )
    print(f"Decode model exported to: {decode_model_path}")


if __name__ == "__main__":
    args = parser_arguments()

    # 复制 modeling_qwen3.py 到模型目录
    src_file_path = os.path.join(now_dir, "modeling_qwen3.py")
    target_file_path = os.path.join(args.hf_model_dir, "modeling_qwen3.py")
    shutil.copy(src_file_path, target_file_path)

    # 修改 config.json 添加 auto_map
    config_json = os.path.join(args.hf_model_dir, "config.json")
    with open(config_json, "rt", encoding="utf-8") as f:
        model_config = json.load(f)

    model_config["auto_map"] = {
        "AutoModel": "modeling_qwen3.Qwen3ForCausalLM",
        "AutoModelForCausalLM": "modeling_qwen3.Qwen3ForCausalLM",
        "AutoModelForSeq2SeqLM": "modeling_qwen3.Qwen3ForCausalLM",
        "AutoModelForSequenceClassification": "modeling_qwen3.Qwen3ForSequenceClassification"
    }
    with open(config_json, "wt", encoding="utf-8") as f:
        json.dump(model_config, f, indent=4)

    # 保存配置
    test_model_config = Qwen3Config.from_pretrained(args.hf_model_dir)
    test_model_config.torch_dtype = "float16"
    test_model_config.save_pretrained(args.hf_model_dir)

    num_hidden_layers = test_model_config.num_hidden_layers
    num_attention_heads = test_model_config.num_attention_heads
    num_key_value_heads = test_model_config.num_key_value_heads
    hidden_size = test_model_config.hidden_size
    vocab_size = test_model_config.vocab_size
    per_head_dim = getattr(test_model_config, 'head_dim', hidden_size // num_attention_heads)

    print("=" * 60)
    print("Qwen3 Separate ONNX Export")
    print("=" * 60)
    print(f"Model config:")
    print(f"  num_hidden_layers: {num_hidden_layers}")
    print(f"  num_attention_heads: {num_attention_heads}")
    print(f"  num_key_value_heads: {num_key_value_heads}")
    print(f"  hidden_size: {hidden_size}")
    print(f"  vocab_size: {vocab_size}")
    print(f"  per_head_dim: {per_head_dim}")
    print(f"  prefill_seq_len: {args.prefill_seq_len}")
    print(f"  kv_cache_length: {args.kv_cache_length}")
    print(f"  dtype: {args.dtype}")
    print("=" * 60)

    # 1. 导出 Embedding
    print("\n--- Exporting Embedding Layer ---")
    embed_model_path = os.path.join(onnx_model_dir, "qwen3_embed.onnx")
    export_embed_onnx(
        device_str=args.device_str,
        dtype=args.dtype,
        hf_model_dir=args.hf_model_dir,
        embed_model_path=embed_model_path,
        vocab_size=vocab_size,
        hidden_size=hidden_size
    )

    # 2. 导出 Prefill 模型
    print("\n--- Exporting Prefill Model ---")
    prefill_model_path = os.path.join(onnx_model_dir, "qwen3_prefill.onnx")
    export_prefill_onnx(
        device_str=args.device_str,
        dtype=args.dtype,
        hf_model_dir=args.hf_model_dir,
        prefill_model_path=prefill_model_path,
        prefill_seq_len=args.prefill_seq_len,
        kv_cache_length=args.kv_cache_length,
        num_hidden_layers=num_hidden_layers,
        num_key_value_heads=num_key_value_heads,
        per_head_dim=per_head_dim,
        hidden_size=hidden_size
    )

    # 3. 导出 Decode 模型
    print("\n--- Exporting Decode Model ---")
    decode_model_path = os.path.join(onnx_model_dir, "qwen3_decode.onnx")
    export_decode_onnx(
        device_str=args.device_str,
        dtype=args.dtype,
        hf_model_dir=args.hf_model_dir,
        decode_model_path=decode_model_path,
        kv_cache_length=args.kv_cache_length,
        num_hidden_layers=num_hidden_layers,
        num_key_value_heads=num_key_value_heads,
        per_head_dim=per_head_dim,
        hidden_size=hidden_size
    )

    print("\n" + "=" * 60)
    print("All models exported successfully!")
    print(f"Output directory: {onnx_model_dir}")
    print("=" * 60)

    # 打印模型信息
    print("\n--- Model Inputs/Outputs Summary ---")
    print("\n1. Embedding Model:")
    print("   Input:  input_ids [batch, seq_len]")
    print("   Output: inputs_embeds [batch, seq_len, hidden_size]")

    print("\n2. Prefill Model:")
    print("   Input:")
    print("     - input_embeds [batch, prefill_seq_len, hidden_size]")
    print("     - attention_mask [batch, prefill_seq_len + kv_cache_length]")
    print("     - position_ids [batch, prefill_seq_len]")
    print("     - past_key_values [batch, kv_cache_length, layers*2*kv_heads, head_dim]")
    print("   Output:")
    print("     - logits [batch, prefill_seq_len, vocab_size]")
    print("     - out_key_values [batch, prefill_seq_len + kv_cache_length, layers*2*kv_heads, head_dim]")

    print("\n3. Decode Model (Fixed-size KV cache, input_ids):")
    print("   Input:")
    print("     - input_ids [batch, 1]  ← token id, model handles embedding internally")
    print("     - attention_mask [batch, kv_cache_length]  ← 1=valid, 0=padding")
    print("     - position_ids [batch, 1]")
    print("     - past_key_values [batch, kv_cache_length, layers*2*kv_heads, head_dim]  ← fixed size")
    print("   Output:")
    print("     - logits [batch, 1, vocab_size]")
    print("     - out_key_values [batch, kv_cache_length, layers*2*kv_heads, head_dim]  ← same fixed size")
