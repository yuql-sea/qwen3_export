"""
Qwen3-0.6B 分离模型 QA Demo
使用 Prefill/Decode 分离的 ONNX 模型进行问答

流程:
1. Prefill: 填充输入到固定长度，计算attention_mask，输出logits和KV cache
2. Decode: 使用KV cache，attention_mask由外部计算（0~past_len为1，剩下为0）
"""

import os
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer


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
KV_CACHE_LENGTH = 1024  # Fixed-size KV cache (must be >= prefill output length)
PREFILL_SEQ_LEN = 513   # 固定prefill长度

# ONNX 提供者
ONNX_PROVIDERS = ['CPUExecutionProvider']

# EOS token
EOS_TOKEN = 151645


def load_tokenizer():
    """加载 tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_DIR,
        trust_remote_code=True,
        use_fast=False
    )
    return tokenizer


def prefill(embed_session, prefill_session, input_ids_np):
    """Prefill 处理 - 填充到固定长度"""
    batch_size, seq_len = input_ids_np.shape

    # Zero padding到固定长度
    if seq_len < PREFILL_SEQ_LEN:
        padding = np.zeros((batch_size, PREFILL_SEQ_LEN - seq_len), dtype=np.int64)
        input_ids_padded = np.concatenate([input_ids_np, padding], axis=1)
    else:
        input_ids_padded = input_ids_np[:, :PREFILL_SEQ_LEN]

    # 获取embedding
    inputs_embeds = embed_session.run(None, {"input_ids": input_ids_padded})[0]

    # Position ids: 真实token位置0~seq_len-1，padding位置为0
    position_ids = np.zeros((1, PREFILL_SEQ_LEN), dtype=np.int64)
    position_ids[0, :seq_len] = np.arange(seq_len, dtype=np.int64)

    # Attention mask: padding位置为0（屏蔽），真实位置为1（不屏蔽）
    attention_mask = np.zeros((1, PREFILL_SEQ_LEN), dtype=np.float32)
    attention_mask[0, :seq_len] = 1.0

    logits, kv = prefill_session.run(None, {
        "input_embeds": inputs_embeds.astype(np.float32),
        "attention_mask": attention_mask,
        "position_ids": position_ids
    })

    # Pad KV cache to fixed KV_CACHE_LENGTH for decode model
    # Only positions 0..seq_len-1 are valid; seq_len..KV_CACHE_LENGTH-1 are zeros
    kv_padded = np.zeros((batch_size, KV_CACHE_LENGTH, kv.shape[2], kv.shape[3]), dtype=kv.dtype)
    kv_padded[:, :seq_len, :, :] = kv[:, :seq_len, :, :]

    return logits, kv_padded, seq_len


def decode_step(decode_session, token_id, kv_cache, past_len, attn_mask):
    """单步解码（固定尺寸 KV cache，直接传 token id）"""
    input_ids = np.array([[token_id]], dtype=np.int64)

    # position is the absolute position in the sequence
    position_ids = np.array([[past_len]], dtype=np.int64)

    # Mark current position as valid in attention_mask before calling decode
    attn_mask[0, past_len] = 1.0

    # Decode model: input_ids → auto embedding → attention → logits
    logits, kv = decode_session.run(None, {
        "input_ids": input_ids,
        "attention_mask": attn_mask,
        "position_ids": position_ids,
        "past_key_values": kv_cache
    })

    return logits, kv, attn_mask


def chat_qa(prompt, enable_thinking=True, max_new_tokens=200):
    """问答对话"""
    print("=" * 70)
    print("Qwen3-0.6B 分离模型 QA Demo")
    print("=" * 70)

    # 加载ONNX模型
    print("\n加载ONNX模型...")
    embed_session = ort.InferenceSession(EMBED_MODEL_PATH, providers=ONNX_PROVIDERS)
    prefill_session = ort.InferenceSession(PRELLL_MODEL_PATH, providers=ONNX_PROVIDERS)
    decode_session = ort.InferenceSession(DECODE_MODEL_PATH, providers=ONNX_PROVIDERS)
    print(f"  Embedding provider: {embed_session.get_providers()[0]}")
    print(f"  Prefill provider: {prefill_session.get_providers()[0]}")
    print(f"  Decode provider: {decode_session.get_providers()[0]}")

    # 加载tokenizer
    tokenizer = load_tokenizer()

    # 构造对话
    messages = [
        {"role": "system", "content": "你叫通义千问，是阿里云开发的智能助手。你擅长用简单易懂的语言解释复杂的概念。"},
        {"role": "user", "content": prompt}
    ]

    # 使用chat template
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking
    )

    print(f"\n问题: {prompt}")
    print(f"\nChat template 后的输入:")
    print("-" * 50)
    print(text)
    print("-" * 50)

    # Tokenize
    input_ids = tokenizer.encode(text, return_tensors='pt')
    seq_len = input_ids.shape[1]
    print(f"\nToken 数量: {seq_len}")

    # Prefill
    import time
    t0 = time.time()
    input_ids_np = input_ids.numpy().astype(np.int64)
    logits, kv, actual_seq_len = prefill(embed_session, prefill_session, input_ids_np)
    t1 = time.time()

    # Initialize attention_mask for decode: positions 0..actual_seq_len-1 are valid
    attn_mask = np.zeros((1, KV_CACHE_LENGTH), dtype=np.float32)
    attn_mask[0, :actual_seq_len] = 1.0

    # 取最后一个真实token的预测
    next_token_id = int(np.argmax(logits[0, actual_seq_len - 1]))
    print(f"\nPrefill 耗时: {(t1-t0)*1000:.1f} ms")
    print(f"Prefill 预测: {next_token_id} = '{tokenizer.decode([next_token_id])}'")

    # 增量解码
    print(f"\n开始增量解码 (max={max_new_tokens} tokens)...")
    print("-" * 50)

    generated_tokens = input_ids.flatten().tolist()
    step = 0
    total_time = 0

    while step < max_new_tokens:
        step += 1
        past_len = seq_len + step - 1
        if past_len >= KV_CACHE_LENGTH:
            print(f"\nReached KV cache limit ({KV_CACHE_LENGTH}), stopping generation")
            break

        t0 = time.time()

        logits, kv, attn_mask = decode_step(decode_session, next_token_id, kv, past_len, attn_mask)

        t1 = time.time()
        total_time += (t1 - t0)

        last_logits = logits[0, 0, :]
        next_token_id = int(np.argmax(last_logits))

        generated_tokens.append(next_token_id)

        word = tokenizer.decode([next_token_id])
        print(f"Step {step:3d}: {next_token_id:6d} = '{word}' ({(t1-t0)*1000:.1f}ms)")

        if next_token_id == EOS_TOKEN:
            print("\n遇到 EOS token，停止生成")
            break

    # 解码回复
    print("-" * 50)
    print(f"\n生成耗时: {total_time*1000:.1f} ms, 总步数: {step}")

    # 提取回复部分（去掉输入）
    response_tokens = generated_tokens[seq_len:]
    response = tokenizer.decode(response_tokens)

    print(f"\n{'='*70}")
    print("模型回复:")
    print(f"{'='*70}")
    print(response)
    print(f"{'='*70}")

    return response


if __name__ == "__main__":
    # 测试问题
    prompt = "用小学生能理解的语言简单介绍一下LLM"

    # 使用思考模式
    response = chat_qa(prompt, enable_thinking=True, max_new_tokens=300)