"""
Qwen3-0.6B 分离模型 ONNX GPU 推理测试
使用 CUDAExecutionProvider 运行 ONNX 模型
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
KV_CACHE_LENGTH = 2048

# ONNX GPU 提供者配置
# 优先使用 CUDA，执行失败时回退到 CPU
ONNX_PROVIDERS = [
    ('CUDAExecutionProvider', {
        'device_id': 0,
        'arena_extend_strategy': 'kSameAsRequested',
        'gpu_mem_limit': 10 * 1024 * 1024 * 1024,  # 10GB
        'cudnn_conv_algo_search': 'DEFAULT',
        'do_copy_in_default_stream': True,
    }),
    'CPUExecutionProvider'
]


def load_tokenizer():
    """加载 tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_DIR,
        trust_remote_code=True,
        use_fast=False
    )
    return tokenizer


def onnx_prefill(embed_session, decoder_session, input_ids_np, kv_cache=None):
    """ONNX Prefill 处理"""
    inputs_embeds = embed_session.run(None, {"input_ids": input_ids_np})[0]

    batch_size, seq_len = input_ids_np.shape

    if kv_cache is None:
        past_key_values = np.zeros(
            (batch_size, KV_CACHE_LENGTH, NUM_LAYERS * 2 * NUM_KV_HEADS, HEAD_DIM),
            dtype=np.float16
        )
    else:
        past_key_values = kv_cache

    attention_mask = np.ones((batch_size, seq_len + KV_CACHE_LENGTH), dtype=np.int64)
    position_ids = np.arange(seq_len, dtype=np.int64).reshape(batch_size, -1)

    logits, kv = decoder_session.run(None, {
        "input_embeds": inputs_embeds.astype(np.float16),
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "past_key_values": past_key_values
    })

    return logits, kv, inputs_embeds


def onnx_decode(embed_session, decoder_session, token_id, kv_cache, position):
    """ONNX 单步解码"""
    input_np = np.array([[token_id]], dtype=np.int64)
    inputs_embeds = embed_session.run(None, {"input_ids": input_np})[0]

    batch_size = 1
    total_len = kv_cache.shape[1] + 1
    attention_mask = np.ones((batch_size, total_len), dtype=np.int64)
    position_ids = np.array([[position]], dtype=np.int64)

    logits, kv = decoder_session.run(None, {
        "input_embeds": inputs_embeds.astype(np.float16),
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "past_key_values": kv_cache
    })

    return logits, kv


def test_gpu_prefill():
    """测试 GPU Prefill"""
    print("=" * 70)
    print("GPU Prefill 测试")
    print("=" * 70)

    # 创建 GPU sessions
    print("\n创建 ONNX GPU sessions...")
    embed_session = ort.InferenceSession(EMBED_MODEL_PATH, providers=ONNX_PROVIDERS)
    prefill_session = ort.InferenceSession(PRELLL_MODEL_PATH, providers=ONNX_PROVIDERS)

    provider = embed_session.get_providers()
    print(f"Embedding provider: {provider[0]}")
    provider = prefill_session.get_providers()
    print(f"Prefill provider: {provider[0]}")

    tokenizer = load_tokenizer()
    prompt = "你好"
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    input_ids = tokenizer.encode(text, return_tensors='pt')

    print(f"\nInput: {text[:50]}...")
    print(f"Token count: {input_ids.shape[1]}")

    # GPU Prefill
    import time
    t0 = time.time()
    input_ids_np = input_ids.numpy().astype(np.int64)
    logits, kv, _ = onnx_prefill(embed_session, prefill_session, input_ids_np)
    t1 = time.time()

    pred_token = int(np.argmax(logits[0, -1]))
    print(f"\nPrefill 结果:")
    print(f"  预测 token: {pred_token} = '{tokenizer.decode([pred_token])}'")
    print(f"  KV shape: {kv.shape}")
    print(f"  GPU 耗时: {(t1-t0)*1000:.2f} ms")

    return logits, kv


def test_gpu_decode():
    """测试 GPU 增量解码"""
    print("\n" + "=" * 70)
    print("GPU 增量 Decode 测试")
    print("=" * 70)

    # 创建 GPU sessions
    embed_session = ort.InferenceSession(EMBED_MODEL_PATH, providers=ONNX_PROVIDERS)
    prefill_session = ort.InferenceSession(PRELLL_MODEL_PATH, providers=ONNX_PROVIDERS)
    decode_session = ort.InferenceSession(DECODE_MODEL_PATH, providers=ONNX_PROVIDERS)

    tokenizer = load_tokenizer()
    prompt = "你好"
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    input_ids = tokenizer.encode(text, return_tensors='pt')
    seq_len = input_ids.shape[1]

    # Prefill
    input_ids_np = input_ids.numpy().astype(np.int64)
    logits, kv, _ = onnx_prefill(embed_session, prefill_session, input_ids_np)
    next_token = int(np.argmax(logits[0, -1]))

    print(f"\nPrefill 预测: {next_token} = '{tokenizer.decode([next_token])}'")

    # 增量解码
    print(f"\n增量解码 (最多20步):")
    print("-" * 50)

    import time
    total_time = 0
    generated = input_ids.flatten().tolist() + [next_token]

    for step in range(1, 21):
        t0 = time.time()
        logits, kv = onnx_decode(embed_session, decode_session, next_token, kv, seq_len + step - 1)
        t1 = time.time()
        total_time += (t1 - t0)

        next_token = int(np.argmax(logits[0, 0]))
        generated.append(next_token)

        word = tokenizer.decode([next_token])
        print(f"Step {step:2d}: {next_token:6d} = '{word}' ({(t1-t0)*1000:.1f}ms)")

        if next_token == 151645:  # EOS
            print("  (遇到 EOS)")
            break

    print("-" * 50)
    print(f"生成内容: {tokenizer.decode(generated[seq_len:])}")
    print(f"总耗时: {total_time*1000:.1f} ms, 平均每步: {total_time*1000/20:.1f} ms")


def test_cpu_vs_gpu():
    """对比 CPU 和 GPU 推理结果"""
    print("\n" + "=" * 70)
    print("CPU vs GPU 精度对比")
    print("=" * 70)

    # CPU Session
    cpu_embed = ort.InferenceSession(EMBED_MODEL_PATH, providers=['CPUExecutionProvider'])
    cpu_prefill = ort.InferenceSession(PRELLL_MODEL_PATH, providers=['CPUExecutionProvider'])
    cpu_decode = ort.InferenceSession(DECODE_MODEL_PATH, providers=['CPUExecutionProvider'])

    # GPU Session
    gpu_embed = ort.InferenceSession(EMBED_MODEL_PATH, providers=ONNX_PROVIDERS)
    gpu_prefill = ort.InferenceSession(PRELLL_MODEL_PATH, providers=ONNX_PROVIDERS)
    gpu_decode = ort.InferenceSession(DECODE_MODEL_PATH, providers=ONNX_PROVIDERS)

    tokenizer = load_tokenizer()
    prompt = "你好"
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    input_ids = tokenizer.encode(text, return_tensors='pt')
    seq_len = input_ids.shape[1]

    # CPU Prefill
    input_ids_np = input_ids.numpy().astype(np.int64)
    cpu_logits, cpu_kv, _ = onnx_prefill(cpu_embed, cpu_prefill, input_ids_np)

    # GPU Prefill
    gpu_logits, gpu_kv, _ = onnx_prefill(gpu_embed, gpu_prefill, input_ids_np)

    # 对比
    cpu_pred = int(np.argmax(cpu_logits[0, -1]))
    gpu_pred = int(np.argmax(gpu_logits[0, -1]))

    logits_diff = np.abs(cpu_logits - gpu_logits).max()
    kv_diff = np.abs(cpu_kv - gpu_kv).max()

    print(f"\nPrefill 对比:")
    print(f"  CPU 预测: {cpu_pred} = '{tokenizer.decode([cpu_pred])}'")
    print(f"  GPU 预测: {gpu_pred} = '{tokenizer.decode([gpu_pred])}'")
    print(f"  Logits 差异: {logits_diff:.6f}")
    print(f"  KV 差异: {kv_diff:.6f}")
    print(f"  {'✓ 预测一致' if cpu_pred == gpu_pred else '✗ 预测不一致'}")

    # 增量解码对比
    print(f"\n增量 Decode 对比 (5步):")
    print("-" * 50)

    cpu_next = cpu_pred
    gpu_next = gpu_pred
    cpu_kv_cur = cpu_kv
    gpu_kv_cur = gpu_kv

    for step in range(1, 6):
        cpu_logits, cpu_kv_cur = onnx_decode(cpu_embed, cpu_decode, cpu_next, cpu_kv_cur, seq_len + step - 1)
        gpu_logits, gpu_kv_cur = onnx_decode(gpu_embed, gpu_decode, gpu_next, gpu_kv_cur, seq_len + step - 1)

        cpu_next = int(np.argmax(cpu_logits[0, 0]))
        gpu_next = int(np.argmax(gpu_logits[0, 0]))

        match = "✓" if cpu_next == gpu_next else "✗"
        print(f"Step {step}: CPU={cpu_next}('{tokenizer.decode([cpu_next])}'), GPU={gpu_next}('{tokenizer.decode([gpu_next])}') {match}")

        if cpu_next != gpu_next:
            print(f"  Logits diff: {np.abs(cpu_logits - gpu_logits).max():.6f}")


if __name__ == "__main__":
    print("Qwen3-0.6B ONNX GPU 推理测试")
    print(f"ONNX Runtime: {ort.__version__}")
    print(f"可用 Providers: {ort.get_available_providers()}")
    print("=" * 70)

    test_gpu_prefill()
    test_gpu_decode()
    test_cpu_vs_gpu()

    print("\n" + "=" * 70)
    print("GPU 测试完成")
    print("=" * 70)
