"""
Qwen3-0.6B 分离模型 ONNX vs PyTorch 对比测试

使用分离的 Prefill/Decode 模型进行端到端推理对比

模型：
- Embedding: token -> embedding
- Prefill: 处理输入序列，返回 logits 和 KV cache
- Decode: 单步解码，返回 logits 和更新的 KV cache
"""

import os
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer
import torch
from modeling_qwen3 import Qwen3ForCausalLM
from transformers.models.qwen3 import Qwen3Config


# 配置
ONNX_DIR = "/home/yuql/workspace/ASR/onnx_hub/onnx_qwen3_separate"
EMBED_MODEL_PATH = os.path.join(ONNX_DIR, "qwen3_embed.onnx")
PRELLL_MODEL_PATH = os.path.join(ONNX_DIR, "qwen3_prefill.onnx")
DECODE_MODEL_PATH = os.path.join(ONNX_DIR, "qwen3_decode.onnx")
MODEL_DIR = "/home/yuql/.cache/modelscope/hub/models/Qwen/Qwen3-0.6B"

# 模型配置
VOCAB_SIZE = 151936
HIDDEN_SIZE = 1024
NUM_LAYERS = 28
NUM_KV_HEADS = 8
HEAD_DIM = 128
KV_CACHE_LENGTH = 1024
PRELLL_SEQ_LEN = 512


def load_tokenizer():
    """加载 tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_DIR,
        trust_remote_code=True,
        use_fast=False
    )
    return tokenizer


def onnx_prefill(embed_session, decoder_session, input_ids_np, kv_cache=None):
    """
    ONNX Prefill 处理

    Args:
        embed_session: embedding ONNX session
        decoder_session: prefill ONNX session
        input_ids_np: input token ids [batch, seq_len]
        kv_cache: 初始 KV cache，如果为 None 则使用全零

    Returns:
        logits: [batch, seq_len, vocab_size]
        kv: 更新后的 KV cache
        inputs_embeds: embeddings (调试用)
    """
    # 获取 embeddings
    inputs_embeds = embed_session.run(None, {"input_ids": input_ids_np})[0]

    batch_size, seq_len = input_ids_np.shape

    # 初始化 KV cache
    if kv_cache is None:
        past_key_values = np.zeros(
            (batch_size, KV_CACHE_LENGTH, NUM_LAYERS * 2 * NUM_KV_HEADS, HEAD_DIM),
            dtype=np.float16
        )
    else:
        past_key_values = kv_cache

    # Attention mask: 全 1（不 mask 任何位置）
    attention_mask = np.ones((batch_size, seq_len + KV_CACHE_LENGTH), dtype=np.int64)

    # Position ids
    position_ids = np.arange(seq_len, dtype=np.int64).reshape(batch_size, -1)

    # Prefill forward
    logits, kv = decoder_session.run(None, {
        "input_embeds": inputs_embeds.astype(np.float32),
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "past_key_values": past_key_values
    })

    return logits, kv, inputs_embeds


def onnx_decode(embed_session, decoder_session, token_id, kv_cache, position, attn_mask):
    """
    ONNX 单步解码（固定尺寸 KV cache）

    Args:
        embed_session: embedding ONNX session
        decoder_session: decode ONNX session
        token_id: 当前 token id
        kv_cache: 当前的 KV cache [1, KV_CACHE_LENGTH, 448, 128]
        position: 当前 position
        attn_mask: attention mask [1, KV_CACHE_LENGTH] (1.0=valid, 0.0=padding)

    Returns:
        logits: [batch, 1, vocab_size]
        kv: 更新后的 KV cache (same fixed size)
        attn_mask: 更新后的 attention_mask
    """
    # 获取 embedding
    input_np = np.array([[token_id]], dtype=np.int64)
    inputs_embeds = embed_session.run(None, {"input_ids": input_np})[0]

    # Mark current position as valid
    attn_mask[0, position] = 1.0

    # Position id
    position_ids = np.array([[position]], dtype=np.int64)

    # Decode forward
    logits, kv = decoder_session.run(None, {
        "input_embeds": inputs_embeds.astype(np.float32),
        "attention_mask": attn_mask,
        "position_ids": position_ids,
        "past_key_values": kv_cache
    })

    return logits, kv, attn_mask


def pt_forward(model, embed, input_ids, attention_mask, position_ids, past_kv=None):
    """PyTorch forward 封装"""
    inputs_embeds = embed(input_ids)
    outputs = model(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_kv
    )
    return outputs[0], outputs[1]


def test_prefill():
    """测试 Prefill 阶段"""
    print("=" * 70)
    print("Test 1: Prefill 对比")
    print("=" * 70)

    # 加载 ONNX
    embed_session = ort.InferenceSession(EMBED_MODEL_PATH, providers=['CPUExecutionProvider'])
    prefill_session = ort.InferenceSession(PRELLL_MODEL_PATH, providers=['CPUExecutionProvider'])
    decode_session = ort.InferenceSession(DECODE_MODEL_PATH, providers=['CPUExecutionProvider'])

    # 加载 PyTorch
    config = Qwen3Config.from_pretrained(MODEL_DIR)
    pt_model = Qwen3ForCausalLM.from_pretrained(MODEL_DIR, torch_dtype=torch.float16).eval()
    pt_embed = pt_model.model.embed_tokens

    # 测试用例
    prompts = ["你好", "你好。", "你好。请问", "请介绍你自己"]

    for prompt in prompts:
        tokenizer = load_tokenizer()
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        input_ids = tokenizer.encode(text, return_tensors='pt')
        seq_len = input_ids.shape[1]

        # ONNX Prefill
        input_ids_np = input_ids.numpy().astype(np.int64)
        onnx_logits, onnx_kv, _ = onnx_prefill(embed_session, prefill_session, input_ids_np)

        # PyTorch Prefill
        with torch.no_grad():
            pt_attention = torch.ones((1, seq_len + KV_CACHE_LENGTH))
            pt_position_ids = torch.arange(seq_len).unsqueeze(0)
            pt_past_kv = torch.zeros((1, KV_CACHE_LENGTH, NUM_LAYERS * 2 * NUM_KV_HEADS, HEAD_DIM), dtype=torch.float16)
            pt_logits, pt_kv = pt_forward(pt_model, pt_embed, input_ids, pt_attention, pt_position_ids, pt_past_kv)

        # 对比
        onnx_pred = int(np.argmax(onnx_logits[0, -1]))
        pt_pred = int(torch.argmax(pt_logits[0, -1]))
        logits_diff = np.abs(onnx_logits - pt_logits.numpy()).max()
        kv_diff = np.abs(onnx_kv - pt_kv.numpy()).max()

        print(f"\nPrompt: '{prompt}' (seq_len={seq_len})")
        print(f"  ONNX pred: {onnx_pred} = '{tokenizer.decode([onnx_pred])}'")
        print(f"  PT pred:   {pt_pred} = '{tokenizer.decode([pt_pred])}'")
        print(f"  Logits diff: {logits_diff:.4f}, KV diff: {kv_diff:.4f}")
        print(f"  {'✓ 一致' if onnx_pred == pt_pred else '✗ 不一致'}")


def test_decode():
    """测试 Decode 阶段"""
    print("\n" + "=" * 70)
    print("Test 2: 增量 Decode 对比")
    print("=" * 70)

    # 加载 ONNX
    embed_session = ort.InferenceSession(EMBED_MODEL_PATH, providers=['CPUExecutionProvider'])
    decode_session = ort.InferenceSession(DECODE_MODEL_PATH, providers=['CPUExecutionProvider'])

    # 加载 PyTorch
    config = Qwen3Config.from_pretrained(MODEL_DIR)
    pt_model = Qwen3ForCausalLM.from_pretrained(MODEL_DIR, torch_dtype=torch.float16).eval()
    pt_embed = pt_model.model.embed_tokens

    tokenizer = load_tokenizer()
    prompt = "你好"
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    input_ids = tokenizer.encode(text, return_tensors='pt')
    seq_len = input_ids.shape[1]

    # 1. Prefill
    input_ids_np = input_ids.numpy().astype(np.int64)
    onnx_logits, onnx_kv, _ = onnx_prefill(embed_session, prefill_session, input_ids_np)

    with torch.no_grad():
        pt_input = torch.tensor(input_ids)
        pt_attention = torch.ones((1, seq_len + KV_CACHE_LENGTH))
        pt_position_ids = torch.arange(seq_len).unsqueeze(0)
        pt_past_kv = torch.zeros((1, KV_CACHE_LENGTH, NUM_LAYERS * 2 * NUM_KV_HEADS, HEAD_DIM), dtype=torch.float16)
        pt_logits, pt_kv = pt_forward(pt_model, pt_embed, pt_input, pt_attention, pt_position_ids, pt_past_kv)

    # 对比 Prefill 结果
    onnx_pred = int(np.argmax(onnx_logits[0, -1]))
    pt_pred = int(torch.argmax(pt_logits[0, -1]))
    print(f"Prefill: ONNX={onnx_pred}('{tokenizer.decode([onnx_pred])}'), PT={pt_pred}('{tokenizer.decode([pt_pred])}') {'✓' if onnx_pred==pt_pred else '✗'}")

    # 2. 增量解码
    print(f"\n增量解码 (最多15步):")
    print("-" * 50)

    onnx_next = onnx_pred
    pt_next = pt_pred
    onnx_generated = input_ids.flatten().tolist() + [onnx_next]
    pt_generated = input_ids.flatten().tolist() + [pt_next]

    # Initialize attention_mask for decode
    onnx_attn_mask = np.zeros((1, KV_CACHE_LENGTH), dtype=np.float32)
    onnx_attn_mask[0, :seq_len] = 1.0  # Prefill tokens are valid

    for step in range(1, 16):
        # ONNX Decode
        past_len = seq_len + step - 1
        onnx_logits, onnx_kv, onnx_attn_mask = onnx_decode(
            embed_session, decode_session, onnx_next, onnx_kv, past_len, onnx_attn_mask
        )
        onnx_next = int(np.argmax(onnx_logits[0, 0]))
        onnx_generated.append(onnx_next)

        # PyTorch Decode
        with torch.no_grad():
            pt_new_input = torch.tensor([[pt_next]])
            pt_new_emb = pt_embed(pt_new_input)
            pt_new_pos = torch.tensor([[seq_len + step - 1]])
            pt_total_len = pt_kv.shape[1] + 1
            pt_new_mask = torch.ones((1, pt_total_len))

            pt_outputs = pt_model(
                inputs_embeds=pt_new_emb,
                attention_mask=pt_new_mask,
                position_ids=pt_new_pos,
                past_key_values=pt_kv
            )
            pt_logits = pt_outputs[0]
            pt_kv = pt_outputs[1]
            pt_next = int(torch.argmax(pt_logits[0, 0]))
            pt_generated.append(pt_next)

        match = "✓" if onnx_next == pt_next else "✗"
        print(f"Step {step:2d}: ONNX={onnx_next:6d}('{tokenizer.decode([onnx_next])}'), PT={pt_next:6d}('{tokenizer.decode([pt_next])}') {match}")

        if onnx_next == pt_next == 151645:  # EOS
            print("  (遇到 EOS，停止)")
            break

    print("\n" + "-" * 50)
    print(f"ONNX: {tokenizer.decode(onnx_generated[seq_len:])}")
    print(f"PT:   {tokenizer.decode(pt_generated[seq_len:])}")
    print(f"生成一致: {'✓' if onnx_generated == pt_generated else '✗'}")


def test_separate_vs_combined():
    """测试分离模型 vs 合并模型"""
    print("\n" + "=" * 70)
    print("Test 3: 分离模型 vs 合并模型")
    print("=" * 70)

    # 加载分离模型
    sep_embed_session = ort.InferenceSession(EMBED_MODEL_PATH, providers=['CPUExecutionProvider'])
    sep_prefill_session = ort.InferenceSession(PRELLL_MODEL_PATH, providers=['CPUExecutionProvider'])
    sep_decode_session = ort.InferenceSession(DECODE_MODEL_PATH, providers=['CPUExecutionProvider'])

    # 加载合并模型
    combined_embed = ort.InferenceSession("/home/yuql/workspace/ASR/onnx_hub/onnx_qwen3/qwen3_embed.onnx", providers=['CPUExecutionProvider'])
    combined_decoder = ort.InferenceSession("/home/yuql/workspace/ASR/onnx_hub/onnx_qwen3/qwen3_0.6b.onnx", providers=['CPUExecutionProvider'])

    tokenizer = load_tokenizer()
    prompt = "你好"
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    input_ids = tokenizer.encode(text, return_tensors='pt')
    seq_len = input_ids.shape[1]

    # 分离模型 Prefill
    input_ids_np = input_ids.numpy().astype(np.int64)
    sep_logits, sep_kv, _ = onnx_prefill(sep_embed_session, sep_prefill_session, input_ids_np)

    # 合并模型 Prefill
    combined_embeds = combined_embed.run(None, {"input_ids": input_ids_np})[0]
    combined_attention = np.ones((1, seq_len + KV_CACHE_LENGTH), dtype=np.int64)
    combined_position = np.arange(seq_len, dtype=np.int64).reshape(1, -1)
    combined_past = np.zeros((1, KV_CACHE_LENGTH, NUM_LAYERS * 2 * NUM_KV_HEADS, HEAD_DIM), dtype=np.float16)

    combined_logits, combined_kv = combined_decoder.run(None, {
        "input_embeds": combined_embeds.astype(np.float32),
        "attention_mask": combined_attention,
        "position_ids": combined_position,
        "past_key_values": combined_past
    })

    # 对比
    sep_pred = int(np.argmax(sep_logits[0, -1]))
    comb_pred = int(np.argmax(combined_logits[0, -1]))

    print(f"\nPrefill 结果对比:")
    print(f"  分离模型: {sep_pred} = '{tokenizer.decode([sep_pred])}'")
    print(f"  合并模型: {comb_pred} = '{tokenizer.decode([comb_pred])}'")
    print(f"  {'✓ 一致' if sep_pred == comb_pred else '✗ 不一致'}")


if __name__ == "__main__":
    print("Qwen3-0.6B 分离模型 ONNX vs PyTorch 对比测试")
    print("=" * 70)

    test_prefill()
    test_decode()
    test_separate_vs_combined()

    print("\n" + "=" * 70)
    print("所有测试完成")
    print("=" * 70)
