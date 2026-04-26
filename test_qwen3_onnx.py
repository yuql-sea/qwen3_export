"""
Qwen3-0.6B ONNX 模型测试脚本 (单 batch)
"""

import os
import numpy as np
import onnxruntime as ort

# 配置
ONNX_DIR = "/home/yuql/workspace/ASR/onnx_hub/onnx_qwen3"
EMBED_MODEL_PATH = os.path.join(ONNX_DIR, "qwen3_embed.onnx")
DECODER_MODEL_PATH = os.path.join(ONNX_DIR, "qwen3_0.6b.onnx")

# Qwen3-0.6B 配置
VOCAB_SIZE = 151936
HIDDEN_SIZE = 1024
NUM_LAYERS = 28
NUM_KV_HEADS = 8
HEAD_DIM = 128
KV_CACHE_LENGTH = 2048


def test_embedding():
    """测试 Embedding 模型."""
    print("=" * 60)
    print("Testing Embedding Model")
    print("=" * 60)

    session = ort.InferenceSession(EMBED_MODEL_PATH, providers=['CPUExecutionProvider'])

    # 单 batch, 单 token
    input_ids = np.array([[1]], dtype=np.int64)  # token id = 1

    print(f"\nInput: input_ids = {input_ids.shape}, dtype={input_ids.dtype}")
    print(f"      value: {input_ids.flatten()}")

    inputs_embeds = session.run(None, {"input_ids": input_ids})[0]

    print(f"\nOutput: inputs_embeds = {inputs_embeds.shape}, dtype={inputs_embeds.dtype}")
    print(f"       前5个值: {inputs_embeds[0, 0, :5]}")

    return inputs_embeds


def test_decoder_first_token(inputs_embeds):
    """测试 Decoder 首次解码 (无 KV cache)."""
    print("\n" + "=" * 60)
    print("Testing Decoder (First Token, No KV Cache)")
    print("=" * 60)

    session = ort.InferenceSession(DECODER_MODEL_PATH, providers=['CPUExecutionProvider'])

    batch_size, seq_len, _ = inputs_embeds.shape

    # attention_mask: [batch, seq_len + kv_len]
    attention_mask = np.ones((batch_size, seq_len + KV_CACHE_LENGTH), dtype=np.int64)

    # position_ids: [batch, seq_len]
    position_ids = np.array([[0]], dtype=np.int64)

    # past_key_values: 全 0 (首次解码无 cache)
    past_key_values = np.zeros(
        (batch_size, KV_CACHE_LENGTH, NUM_LAYERS * 2 * NUM_KV_HEADS, HEAD_DIM),
        dtype=np.float16
    )

    print(f"\nInputs:")
    print(f"  input_embeds:  {inputs_embeds.shape}, dtype={inputs_embeds.dtype}")
    print(f"  attention_mask:{attention_mask.shape}, dtype={attention_mask.dtype}")
    print(f"  position_ids:  {position_ids.shape}, dtype={position_ids.dtype}")
    print(f"  past_key_values:{past_key_values.shape}, dtype={past_key_values.dtype}")

    logits, out_kv = session.run(None, {
        "input_embeds": inputs_embeds,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "past_key_values": past_key_values
    })

    print(f"\nOutputs:")
    print(f"  logits:       {logits.shape}, dtype={logits.dtype}")
    print(f"  out_key_values: {out_kv.shape}, dtype={out_kv.dtype}")

    # 预测的 token
    pred_token = np.argmax(logits[0, 0, :])
    print(f"\nPredicted token id: {pred_token}")

    return logits, out_kv, pred_token


def test_decoder_incremental(prev_kv, step):
    """测试 Decoder 增量解码 (有 KV cache)."""
    print("\n" + "-" * 40)
    print(f"Testing Decoder (Incremental Step {step})")

    session = ort.InferenceSession(DECODER_MODEL_PATH, providers=['CPUExecutionProvider'])

    # 新 token 的 embedding (用随机值模拟)
    new_embeds = np.random.randn(1, 1, HIDDEN_SIZE).astype(np.float16)

    # 新 position (递增)
    new_position = np.array([[step]], dtype=np.int64)

    # attention_mask 长度 = past_kv 长度 + 1 (当前 token)
    total_len = prev_kv.shape[1] + 1
    attention_mask = np.ones((1, total_len), dtype=np.int64)

    print(f"  input_embeds:  {new_embeds.shape}")
    print(f"  position_ids:   {new_position.flatten()}")
    print(f"  attention_mask: {attention_mask.shape} (total_len={total_len})")
    print(f"  past_kv:       from prev (shape={prev_kv.shape})")

    logits, out_kv = session.run(None, {
        "input_embeds": new_embeds,
        "attention_mask": attention_mask,
        "position_ids": new_position,
        "past_key_values": prev_kv
    })

    new_pred = np.argmax(logits[0, 0, :])
    print(f"  Predicted token id: {new_pred}")

    return logits, out_kv, new_pred


if __name__ == "__main__":
    print("Qwen3-0.6B ONNX Inference Test (Single Batch)")
    print("=" * 60)
    print(f"Config: VOCAB={VOCAB_SIZE}, HIDDEN={HIDDEN_SIZE}, LAYERS={NUM_LAYERS}")
    print(f"        KV_HEADS={NUM_KV_HEADS}, HEAD_DIM={HEAD_DIM}, KV_LEN={KV_CACHE_LENGTH}")
    print("=" * 60)

    # Test 1: Embedding
    inputs_embeds = test_embedding()

    # Test 2: Decoder first token
    logits, out_kv, pred_token = test_decoder_first_token(inputs_embeds)

    # Test 3: 增量解码循环
    print("\n" + "=" * 60)
    print("Incremental Decoding Loop (5 steps)")
    print("=" * 60)

    tokens = [pred_token]
    current_kv = out_kv

    for step in range(1, 6):
        _, current_kv, token = test_decoder_incremental(current_kv, step)
        tokens.append(token)

    print(f"\nDecoded tokens: {tokens}")

    print("\nAll tests passed!")
