"""
Qwen3-0.6B ONNX 问答测试
测试: "请介绍你自己"
"""

import os
import numpy as np
import onnxruntime as ort

# 配置
ONNX_DIR = "/home/yuql/workspace/ASR/onnx_hub/onnx_qwen3"
EMBED_MODEL_PATH = os.path.join(ONNX_DIR, "qwen3_embed.onnx")
DECODER_MODEL_PATH = os.path.join(ONNX_DIR, "qwen3_0.6b.onnx")
MODEL_DIR = "/home/yuql/.cache/modelscope/hub/models/Qwen/Qwen3-0.6B"

# Qwen3-0.6B 配置
VOCAB_SIZE = 151936
HIDDEN_SIZE = 1024
NUM_LAYERS = 28
NUM_KV_HEADS = 8
HEAD_DIM = 128
KV_CACHE_LENGTH = 2048

# 特殊 token
BOS_TOKEN = 151643
EOS_TOKEN = 151645
PAD_TOKEN = 151643


def load_tokenizer():
    """加载 tokenizer."""
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_DIR,
        trust_remote_code=True,
        use_fast=False
    )
    return tokenizer


def test_qa():
    """问答测试 (使用 chat template)."""
    print("=" * 70)
    print("Qwen3-0.6B 问答测试 (Chat Template)")
    print("=" * 70)

    # 加载 tokenizer
    print("\n加载 tokenizer...")
    tokenizer = load_tokenizer()

    # 使用 chat template 格式化输入
    prompt = "请用小学生能理解的语言简单介绍一下LLM。"
    messages = [
        {"role": "user", "content": prompt}
    ]

    # apply_chat_template 会自动添加 system prompt、generation prompt 等
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False  # 关闭思考模式，直接生成答案
    )

    print(f"\n原始 prompt: {prompt}")
    print(f"\nChat template 后的 text:")
    print("-" * 50)
    print(text)
    print("-" * 50)

    # Tokenize - 只需 input_ids
    input_ids = tokenizer.encode(text, return_tensors='pt')
    print(f"\nInput token ids shape: {input_ids.shape}")
    print(f"Token count: {input_ids.shape[1]}")

    # 验证 tokenizer 能正确编解码
    decoded = tokenizer.decode(input_ids[0])
    print(f"\n验证 decode (前100字符): {decoded[:100]}...")

    # 创建 ONNX sessions
    embed_session = ort.InferenceSession(EMBED_MODEL_PATH, providers=['CPUExecutionProvider'])
    decoder_session = ort.InferenceSession(DECODER_MODEL_PATH, providers=['CPUExecutionProvider'])

    # 首次解码 - 处理完整的 input_ids
    print("\n" + "-" * 50)
    print("Step 1: Prefill (处理输入序列)")
    print("-" * 50)

    batch_size = 1
    seq_len = input_ids.shape[1]  # 输入序列长度

    # 获取 embedding
    input_ids_np = input_ids.numpy().astype(np.int64)
    inputs_embeds = embed_session.run(None, {"input_ids": input_ids_np})[0]
    print(f"inputs_embeds shape: {inputs_embeds.shape}")

    # 准备 decoder 输入
    attention_mask = np.ones((batch_size, seq_len + KV_CACHE_LENGTH), dtype=np.int64)
    position_ids = np.arange(seq_len, dtype=np.int64).reshape(1, -1)
    past_key_values = np.zeros(
        (batch_size, KV_CACHE_LENGTH, NUM_LAYERS * 2 * NUM_KV_HEADS, HEAD_DIM),
        dtype=np.float16
    )

    print(f"attention_mask shape: {attention_mask.shape}")
    print(f"position_ids: {position_ids.flatten().tolist()}")

    # 首次 decoder forward
    logits, out_kv = decoder_session.run(None, {
        "input_embeds": inputs_embeds.astype(np.float16),
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "past_key_values": past_key_values
    })

    print(f"logits shape after prefill: {logits.shape}")  # [1, seq_len, vocab_size]

    # 取最后一个位置的 logits 作为预测
    last_logits = logits[0, -1, :]
    next_token_id = np.argmax(last_logits).item()
    print(f"First predicted token: {next_token_id}")

    # 构建已生成的 token 列表
    generated_tokens = input_ids.flatten().tolist() + [next_token_id]

    # 当前 KV cache
    current_kv = out_kv

    # 增量解码
    print("\n" + "-" * 50)
    print("Step 2: Incremental Decoding")
    print("-" * 50)

    max_new_tokens = 100
    eos_token_id = EOS_TOKEN
    eos_token = "<|im_end|>"

    step = 0
    recent_tokens = []  # 用于检测重复 n-gram
    while step < max_new_tokens:
        step += 1

        # 新 token 的 embedding
        new_input_ids = np.array([[next_token_id]], dtype=np.int64)
        new_embeds = embed_session.run(None, {"input_ids": new_input_ids})[0]

        # 新 position
        new_position = np.array([[seq_len + step - 1]], dtype=np.int64)

        # attention_mask
        total_len = current_kv.shape[1] + 1
        new_attention_mask = np.ones((batch_size, total_len), dtype=np.int64)

        # decoder forward
        logits, current_kv = decoder_session.run(None, {
            "input_embeds": new_embeds.astype(np.float16),
            "attention_mask": new_attention_mask,
            "position_ids": new_position,
            "past_key_values": current_kv
        })

        # 预测下一个 token
        last_logits = logits[0, 0, :]
        next_token_id = int(np.argmax(last_logits))

        generated_tokens.append(next_token_id)

        # 解码当前 token
        current_word = tokenizer.decode([next_token_id])
        print(f"Step {step:3d}: token_id={next_token_id:6d}, word='{current_word}'")

        # 检查 EOS
        if next_token_id == eos_token_id:
            print(f"\n✓ 生成结束 (遇到 EOS token: {eos_token})")
            break

        # 检测 n-gram 重复 (最近 6 个 token 作为 pattern)
        recent_tokens.append(next_token_id)
        if len(recent_tokens) >= 6:
            # 检查最后 6 个 token 是否与之前某段重复
            pattern = tuple(recent_tokens[-6:])
            for i in range(len(recent_tokens) - 6):
                if tuple(recent_tokens[i:i+6]) == pattern:
                    print(f"\n⚠ 生成结束 (检测到重复 pattern: {[tokenizer.decode([t]) for t in pattern]})")
                    break
            else:
                continue
            break

    # 解码完整回复
    print("\n" + "=" * 70)
    print("生成结果:")
    print("=" * 70)
    # 只解码新增的 token (不包括输入)
    new_tokens = generated_tokens[input_ids.shape[1]:]
    full_response = tokenizer.decode(new_tokens)
    print(f"{full_response}")

    print("\n" + "=" * 70)
    print(f"完整 token 序列 ({len(generated_tokens)} tokens):")
    print("=" * 70)
    print(generated_tokens)

    return full_response


if __name__ == "__main__":
    test_qa()
