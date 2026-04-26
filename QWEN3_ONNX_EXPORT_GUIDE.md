# Qwen3-0.6B ONNX 导出完整流程文档

## 目录

1. [概述](#1-概述)
2. [运行环境](#2-运行环境)
3. [模型架构与分析](#3-模型架构与分析)
4. [导出策略：Prefill/Decode 分离](#4-导出策略prefilldecode-分离)
5. [Export Wrapper 设计](#5-export-wrapper-设计)
6. [KV Cache 设计变更：从 Concat 到 Fixed-Size Scatter](#6-kv-cache-设计变更从-concat-到-fixed-size-scatter)
7. [Qwen3Model.forward 分支逻辑](#7-qwen3modelforward-分支逻辑)
8. [注意掩码 (Attention Mask) 设计](#8-注意掩码-attention-mask-设计)
9. [Demo 推理流程](#9-demo-推理流程)
10. [已攻克的问题与解决办法](#10-已攻克的问题与解决办法)
11. [当前已知问题与后续工作](#11-当前已知问题与后续工作)
12. [快速启动](#12-快速启动)

---

## 1. 概述

### 1.1 目标

将 Qwen3-0.6B 模型导出为 ONNX，并实现分离式推理：

- **Embedding 模型**：将 token ids 转为 embeddings
- **Prefill 模型**：批量处理输入序列，输出 logits 和 KV cache
- **Decode 模型**：单步增量解码（直接接受 `input_ids`），输出 logits 和更新后的 KV cache

硬件目标：Ascend NPU 及特化芯片部署（要求固定 size 模型）。

### 1.2 模型规格

| 参数 | 值 |
|------|-----|
| 模型 | Qwen3-0.6B |
| hidden_size | 1024 |
| num_attention_heads | 16 |
| num_key_value_heads | 8 |
| head_dim | 128 |
| num_hidden_layers | 28 |
| vocab_size | 151936 |
| GQA groups | 2 |

### 1.3 关键文件

| 路径 | 说明 |
|------|------|
| `export/modeling_qwen3.py` | 自定义的 Qwen3 模型定义（ONNX 导出版本） |
| `export/export_qwen3_separate.py` | 分离模型导出脚本：生成 Embedding / Prefill / Decode 三个 ONNX |
| `export/demo_qwen3_separate_qa.py` | 端到端 QA 推理 Demo |
| `export/ENVIRONMENT_REFERENCE.md` | 环境与工具参考 |
| `output/onnx_qwen3_separate/` | 导出的 ONNX 模型目录 |

---

## 2. 运行环境

### 2.1 Conda 环境

推荐使用 `OSUM` 环境（确认包含 torch 2.7.1+cu126）：

```bash
source /home/yuql/miniconda3/bin/activate OSUM
```

其他可用环境：

| 环境 | torch 版本 | 备注 |
|------|-----------|------|
| OSUM | 2.7.1+cu126 | 主用环境，推荐 |
| cosyvoice | 2.3.1+cu121 | 备用 |
| ggml | 2.5.1+cpu | CPU 调试用 |

### 2.2 关键依赖

```
torch>=2.5.0
transformers>=4.50.0
onnx>=1.14.0
onnxruntime>=1.23.0
numpy>=1.24.0
```

### 2.3 模型路径

```
/home/yuql/.cache/modelscope/hub/models/Qwen/Qwen3-0.6B
```

### 2.4 输出目录

```
/home/yuql/workspace/ASR/onnx_hub/onnx_qwen3_separate/
```

---

## 3. 模型架构与分析

### 3.1 Qwen3 与 Qwen2 的关键差异

Qwen3 vs Qwen2 在架构上的主要区别：

| 差异点 | Qwen2 | Qwen3 |
|--------|-------|-------|
| QKV 计算顺序 | transpose 后 norm | transpose **前** norm（`q_norm`, `k_norm`） |
| Norm 方式 | LayerNorm | RMSNorm |
| KV cache 处理 | 标准 HuggingFace | 自定义固定 size |

### 3.2 GQA (Grouped Query Attention)

```
num_attention_heads = 16   →  query  有 16 个头
num_key_value_heads = 8    →  key/value 各有 8 个头
num_key_value_groups = 2   →  每 2 个 query 头共享 1 组 key/value
```

在 attention 计算中，key/value 通过 `repeat_kv` 从 8 头复制到 16 头后与 query 做 matmul。

### 3.3 Qwen3 的 q_norm / k_norm

Qwen3 官方实现将 QKV projection 后，先 reshape 到 `[batch, seq, num_heads, head_dim]`，然后对 query/key 做 norm（`q_norm`, `k_norm`），**最后才 transpose 到 `[batch, num_heads, seq, head_dim]`**。这与 Qwen2 的做法不同。

```python
# Qwen3 顺序（正确）
query_states = self.q_proj(hidden_states)   # [b, s, h]
query_states = query_states.view(b, s, n_h, d)  # [b, s, n_h, d]
query_states = self.q_norm(query_states)    # [b, s, n_h, d] (norm 在 head_dim)
query_states = query_states.transpose(1, 2) # [b, n_h, s, d]
```

---

## 4. 导出策略：Prefill/Decode 分离

将推理拆成两个阶段，用两个独立的 ONNX 模型分别处理：

```
┌─────────┐   input_ids   ┌──────────┐   inputs_embeds   ┌──────────┐
│ Embed   │──────────────→│  Prefill │──────────────────→│  Decode   │
│ (onnx)  │               │ (onnx)   │                   │ (onnx)    │
└─────────┘               └──────────┘                   └──────────┘
                               │                              │
                               ▼                              ▼
                          logits + KV                    logits + KV
```

### 4.1 Embedding 模型

```python
class EmbeddingWrapper(nn.Module):
    def forward(self, input_ids):
        return self.model.model.embed_tokens(input_ids)
```

- **输入**: `input_ids [batch, seq]`
- **输出**: `inputs_embeds [batch, seq, 1024]`
- **动态轴**: `batch_size`, `seq_length`

### 4.2 Prefill 模型

```python
class PrefillWrapper(nn.Module):
    def forward(self, input_embeds, attention_mask, position_ids):
        return self.model(inputs_embeds=input_embeds,
                         attention_mask=attention_mask,
                         position_ids=position_ids,
                         past_key_values=None)
```

- **输入**:
  - `input_embeds [batch, seq, 1024]`
  - `attention_mask [batch, seq]` (1.0=attend, 0.0=padding)
  - `position_ids [batch, seq]`
- **输出**:
  - `logits [batch, seq, 151936]`
  - `out_key_values [batch, seq, 448, 128]` (prefill 只输出有效位置的 KV)
- **动态轴**: `batch_size`, `prefill_seq_len`

### 4.3 Decode 模型

```python
class DecodeWrapper(nn.Module):
    def forward(self, input_ids, attention_mask, position_ids, past_key_values):
        return self.model(input_ids=input_ids,
                         attention_mask=attention_mask,
                         position_ids=position_ids,
                         past_key_values=past_key_values)
```

- **输入**:
  - `input_ids [batch, 1]` (直接接受 token id，内部自动 embedding)
  - `attention_mask [batch, KV_LEN]` (固定大小)
  - `position_ids [batch, 1]`
  - `past_key_values [batch, KV_LEN, 448, 128]` (固定大小)
- **输出**:
  - `logits [batch, 1, 151936]`
  - `out_key_values [batch, KV_LEN, 448, 128]` (固定大小，与输入一致)
- **动态轴**: 仅 `batch_size`

---

## 5. Export Wrapper 设计

`export_qwen3_separate.py` 中为三个模型分别定义了 ExportWrapper，每个 wrapper 封装了模型调用方式。

### 5.1 为什么需要 Wrapper？

1. **区分 Prefill 和 Decode 的 forward 签名**
   - Prefill：传入 `inputs_embeds`（已经过 embedding 层），`past_key_values=None`
   - Decode：传入 `input_ids`（自动 embedding），`past_key_values` 为实际缓存
2. **隐藏内部参数**：用户不需要知道 attention_mask 的处理细节、past_key_values 的格式等
3. **控制输入输出**：ONNX 的输入输出签名由 wrapper 决定

### 5.2 Wrapper 设计要点

**Prefill Wrapper**：
```python
class PrefillWrapper(nn.Module):
    def forward(self, input_embeds, attention_mask, position_ids):
        return self.model(inputs_embeds=input_embeds,
                         attention_mask=attention_mask,
                         position_ids=position_ids,
                         past_key_values=None)
```

- 传入 `past_key_values=None`，触发 model 的 prefill 分支
- 传入 `attention_mask`（padding mask）

**Decode Wrapper**：
```python
class DecodeWrapper(nn.Module):
    def forward(self, input_ids, attention_mask, position_ids, past_key_values):
        return self.model(input_ids=input_ids,
                         attention_mask=attention_mask,
                         position_ids=position_ids,
                         past_key_values=past_key_values)
```

- 传入 `input_ids`（非 `inputs_embeds`），由模型内部做 embedding
- 传入 `past_key_values` 触发 decode 分支

### 5.3 Wrapper vs Direct PyTorch forward

导出的 ONNX 模型调用 `model.forward()`，其判断逻辑在 `Qwen3ForCausalLM.forward` 和 `Qwen3Model.forward` 中。

---

## 6. KV Cache 设计变更：从 Concat 到 Fixed-Size Scatter

### 6.1 最初方案：Concat

```python
# 早期版本
cache_key = past_key_value[..., layer_offset:layer_offset + num_kv_heads, :].transpose(1, 2)
key_states = torch.cat((cache_key, key_states), dim=2)
```

**问题**：
- KV cache 长度随 step 增长（`KV_LEN, KV_LEN+1, KV_LEN+2, ...`）
- ONNX 动态 shape 在 Ascend/特化芯片上不友好
- attention_mask 维度需要随之动态变化，容易导致维度不匹配

### 6.2 最终方案：Fixed-Size Scatter

```python
# 当前版本
cache_key = past_key_value[:, key_offset:key_offset + num_kv_heads, :, :]  # [b, kv_h, KV_LEN, d]
kv_cache_len = cache_key.shape[2]
idx = torch.arange(kv_cache_len, device=...).view(1, 1, kv_cache_len, 1)
pos = position_ids.view(bsz, 1, 1)
write_mask = (idx == pos).to(dtype=hidden_states.dtype)  # 1.0 at target position

key_states = cache_key * (1.0 - write_mask) + key_states * write_mask
value_states = cache_value * (1.0 - write_mask) + value_states * write_mask
```

**优点**：
- 输入输出 KV cache 始终是 `[batch, KV_LEN, 448, 128]`，**固定大小**
- attention_mask 也固定为 `[batch, KV_LEN]`，与 KV cache 长度一致
- 无维度增长，适合 Ascend 硬件部署
- arithmetic mask 方式（`(idx == pos)`）完全由 ONNX 原生算子组成，无需控制流

### 6.3 KV cache 格式

```
past_key_values: [batch, seq, 2 * num_layers * num_kv_heads, head_dim]
                 = [batch, KV_LEN, 448, 128]

其中 448 = 28 层 × 2 (k/v) × 8 kv_heads
```

**索引方式**：
```
key_offset   = layer_idx * num_kv_heads          # 该层的 key 起始位置
value_offset = num_layers * num_kv_heads + layer_idx * num_kv_heads  # 该层的 value 起始位置

# 例如 layer 0: key_offset=0, value_offset=224
# layer 1: key_offset=8, value_offset=232
# ...

第一半:  layer 0-27 的所有 key states   (index 0~223)
第二半:  layer 0-27 的所有 value states (index 224~447)
```

---

## 7. Qwen3Model.forward 分支逻辑

`Qwen3Model.forward` 通过 `past_key_values is None` 来区分 prefill 和 decode 两种模式：

```python
if past_key_values is not None:
    # === Decode 分支 ===
    # attention_mask: [batch, KV_LEN], 1.0=attend, 0.0=padding
    # 转换为 additive mask: → [batch, 1, 1, KV_LEN]
    if attention_mask is not None:
        attention_mask = (1.0 - attention_mask) * (-10000.0)
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
else:
    # === Prefill 分支 ===
    if attention_mask is None:
        # 默认 causal mask: [batch, 1, seq, seq]
    else:
        # 结合 padding mask + causal mask: [batch, 1, seq, seq]
```

### 7.1 为什么用 `past_key_values is None` 而不是 `torch.all(past_key_values == 0)` 判断？

**这是一个重要的经验教训**。

早期版本尝试用 `_past_kv.abs().max().item() == 0` 检测是否全零 tensor 来判断是否为 prefill：

```python
_is_all_zero = (_past_kv.abs().max().item() == 0)
```

**问题**：
- ONNX tracer 会将 `.item()` 的结果（Python float）作为**编译时常量**处理
- 导出的 ONNX 模型在运行时无论输入的 KV cache 是否全零，都固定走 tracer 记录的某一条分支
- 导致维度不匹配的 RuntimeError

**正确做法**：
- 使用 Python 层面的条件判断：`past_key_values is not None`
- 在 wrapper 层就决定好传 None 还是传 tensor

---

## 8. 注意掩码 (Attention Mask) 设计

### 8.1 Prefill 的 Attention Mask

Prefill 时 attention_mask 是用户传入的长度为 `seq_len` 的 1D mask：
- `1.0` = 有效 token（参与 attention 计算）
- `0.0` = padding token（不参与 attention 计算）

```python
# Prefill: [batch, seq_len] → 加上 causal mask → [batch, 1, seq, seq]
padding_mask = attention_mask.unsqueeze(1).bool()      # [batch, 1, seq]
combined_mask = causal_mask.unsqueeze(0) & padding_mask # [batch, seq, seq]
attention_mask = torch.where(combined_mask, 0.0, -10000.0)
attention_mask = attention_mask.unsqueeze(1)
```

### 8.2 Decode 的 Attention Mask

Decode 时 attention_mask 是长度为 `KV_LEN` 的 1D mask，标记 KV cache 中哪些位置已有有效 KV：

```python
# Decode: [batch, KV_LEN] → [batch, 1, 1, KV_LEN]
attention_mask = (1.0 - attention_mask) * (-10000.0)
attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
```

来自 Demo 的 attention_mask 维护：
```python
# 初始化：prefill 结束后的 KV cache 有效位置
attn_mask = np.zeros((1, KV_CACHE_LENGTH), dtype=np.float32)
attn_mask[0, :actual_seq_len] = 1.0

# 每步 decode：标记当前位置为有效
attn_mask[0, past_len] = 1.0
```

### 8.3 Attention 层的 Mask 应用

在 `Qwen3Attention.forward` 中，加法 masking：

```python
attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
if attention_mask is not None:
    if attention_mask.shape[-1] == key_states.shape[-2]:
        attn_weights = attn_weights + attention_mask
```

关键点：mask 的最后一维必须等于 key_states 的序列长度。由于 KV cache 固定为 KV_LEN，prefill 时 mask 尺寸为 `seq × seq`，decode 时为 `1 × KV_LEN`（broadcast 到 `1 × KV_LEN`）。

---

## 9. Demo 推理流程

`demo_qwen3_separate_qa.py` 的推理流程：

### 9.1 Prefill 阶段

1. Tokenize user input → `input_ids [1, seq_len]`
2. Zero-padding 到 PREFILL_SEQ_LEN（固定长度）
3. Embedding: `input_ids` → `inputs_embeds`
4. 创建 `position_ids`：前 seq_len 个位置为 0,1,2,...，padding 位置为 0
5. 创建 `attention_mask`：前 seq_len 个位置为 1.0（有效），之后为 0.0（padding）
6. **Prefill 推理**：`input_embeds, attention_mask, position_ids → logits, kv`
7. KV cache padding：将 prefill 输出的 kv（长度 seq_len）pad 到 KV_CACHE_LENGTH

### 9.2 Decode 阶段

1. 初始化 attention_mask：前 seq_len 个位置为 1.0
2. 取 prefill 输出的最后一个有效 token 的 logits，argmax 得到第一个生成 token
3. 循环：
   a. Decode 推理：`input_ids, attention_mask, position_ids, past_key_values`
   b. 更新 attention_mask：将当前 position 标记为 1.0
   c. argmax 取下一个 token
   d. 直到遇到 EOS token 或达到最大步数

### 9.3 流程图

```
Tokenize input
     │
     ▼
Pad to PREFILL_SEQ_LEN
     │
     ▼
Embedding (onnx)
     │
     ▼
Prefill (onnx) ───→ logits + KV (seq_len)
     │
     ▼
Pad KV to KV_CACHE_LENGTH
     │
     ▼
┌─────────────────────────┐
│ Decode Step (onnx)      │
│ input_ids, attn_mask,   │
│ position_ids, KV cache  │
└─────────┬───────────────┘
     │
     ▼
logits + KV (KV_LEN)
     │
     ▼
argmax → next token
     │
     ▼
[EOS?] ──Yes──→ Done
     │
     No
     ▼
Update attn_mask, position
     │
     ▼
[Loop back]
```

---

## 10. 已攻克的问题与解决办法

### 10.1 Attention Mask 类型错误

**现象**：Prefill 输出 logits 与 PyTorch 原生不一致，出现乱码。

**原因**：`torch.zeros_like(full_attention_mask, dtype=torch.float32)` 对 bool tensor 做 `zeros_like` 会保留 bool 类型，`dtype` 参数被忽略。

```python
# 错误代码
attention_mask = torch.zeros_like(full_attention_mask, dtype=torch.float32)
# 实际创建了 bool tensor！
```

**修复**：
```python
# 正确做法
attention_mask = torch.zeros(full_attention_mask.shape, dtype=torch.float32, device=inputs_embeds.device)
```

**教训**：`torch.zeros_like` 与 `torch.zeros` 对 dtype 的处理不同，前者 dtype 参数优先级低于输入 tensor 的类型推断。

### 10.2 KV Cache 维度顺序问题

**现象**：增量解码第一步就输出乱码。

**原因**：cache 格式为 `[batch, seq, kv_heads, head_dim]`，但 attention 计算需要 `[batch, kv_heads, seq, head_dim]`。

**修复**：
```python
cache_key = past_key_value[:, :, layer_offset:layer_offset + num_kv_heads, :].transpose(1, 2)
```

**教训**：HuggingFace 标准实现使用 `past_key_values` 为 tuple of tuples，每个元素为 `[batch, kv_heads, seq, head_dim]`。但在固定 cache 的改造中，维度顺序改为 `[batch, seq, kv_heads, head_dim]`，需要手动 transpose。

### 10.3 ONNX Tracer 将 `.item()` 当作常量

**现象**：导出 prefill 时 `_is_truly_empty` 检测永远为 False，无论输入是否全零。

**原因**：ONNX tracer 执行 `_past_kv.abs().max().item()` 时，将返回的 Python float 作为编译时常量记录下来，不会随实际输入变化。

**修复**：用 `past_key_values is None` 替代 `torch.all(past_key_values == 0)` 判断。

```python
# 错误方式（会在 ONNX tracer 中被固化）
_is_all_zero = (_past_kv.abs().max().item() == 0)
if _past_kv is not None and _past_kv.shape[2] > 0 and not _is_truly_empty:
    # decode 分支
else:
    # prefill 分支

# 正确方式（在 wrapper 层决定）
if past_key_values is not None:
    # decode 分支
else:
    # prefill 分支
```

**教训**：导出 ONNX 时，所有 Python-level 的控制流都会在 tracing 时锁定。不应在模型中通过 PIL 或 `.item()` 返回值做分支判断。

### 10.4 KV cache 用 Concat 导致维度增长

**现象**：Onnx 报错 `RuntimeError: tensor a (449) must match tensor b (2049)`，attention_mask 和 key_states 维度不匹配。

**原因**：用 `torch.cat((cache_key, key_states), dim=2)` 做增量 KV cache，导致序列长度不断增长（KV_LEN → KV_LEN+1 → KV_LEN+2）。但 attention_mask 是按 `[batch, KV_LEN+1]` 固定构造的，两者维度不匹配。

**修复**：改用 fixed-size scatter 方案，KV cache 输入输出保持相同大小：

```python
# 不增长，而是通过 write_mask 覆盖
write_mask = (idx == pos).to(dtype=hidden_states.dtype)
key_states = cache_key * (1.0 - write_mask) + key_states * write_mask
```

### 10.5 Prefill 输入 attention_mask 维度

**现象**：导出的 prefill ONNX 输入 `attention_mask` 维度为 `[batch, seq_len + kv_cache_length]`，过大。

**修复**：改为 `[batch, seq_len]`，只传递真实输入长度的 mask。prefill 时的 causal mask 由模型内部生成，argmask 只需要标记 padding 位置。

### 10.6 `presents` 的拼接方式

**现象**：`presents` 的拼接方式在版本演进中变化了多次。

**初始方案**：
```python
presents = []
for layer_output in layer_outputs:
    presents.extend(layer_output[1])
```

**最终方案**：
```python
all_keys, all_values = [], []
for k, v in presents:
    all_keys.append(k)
    all_values.append(v)
presents = torch.cat(all_keys + all_values, dim=1)
presents = presents.transpose(1, 2)
```

**原因**：当前 `Qwen3DecoderLayer.forward` 返回 `present_key_value` 为 `(key_states, value_states)` 的 tuple（不再是 list）。需要分别收集所有层的 key/value，在 dim=1 上 cat，最后 transpose 回标准格式。

### 10.7 input_ids 类型判断

**现象**：ONNX export 时，`input_ids` 作为第一个位置参数传入，但实际是 float tensor（embeddings）。

**修复**：在 `Qwen3ForCausalLM.forward` 中判断 dtype：
```python
if inputs_embeds is None and input_ids is not None:
    if input_ids.dtype.is_floating_point:
        # input_ids 实际是 embeddings（ONNX 兼容）
        inputs_embeds = input_ids
        input_ids = None
    else:
        # 正常情况：input_ids 是 token ids
        inputs_embeds = self.model.embed_tokens(input_ids)
```

---

## 11. 当前已知问题与后续工作

### 11.1 Decode 导出为文件夹格式

Decode ONNX 模型导出为外部权重格式（`Constant_*` 和 `model.*.weight` 文件），而非单一 `.onnx` 文件。这是 `torch.onnx.export` 在模型大于 2GB 时的默认行为。使用时需确保这些文件与 `.onnx` 在同一目录。

### 11.2 Decode out_key_values 第三维度为 0

`out_key_values` 的 shape 显示为 `[0, 1024, 0, 128]`（ONNX metadata），第三维为 0。这可能是 ONNX shape inference 无法推断动态维度的结果，运行时实际 shape 应该正确。需要进一步验证。

### 11.3 端到端精度验证

目前验证了：
- Prefill logits 正确输出（有意义的 token 预测，与 PyTorch 差异 < 0.01）
- Decode 模型导出成功
- KV cache scatter 逻辑通过 ONNX 导出检查

**仍需验证**：
- 完整 Embedding → Prefill → Decode 链路的端到端推理正确性
- Decode 输出的 KV cache 增量与 PyTorch 一致
- 多步解码的精度累积情况

---

## 12. 快速启动

```bash
# 1. 激活环境
source /home/yuql/miniconda3/bin/activate OSUM

# 2. 进入工作目录
cd /home/yuql/workspace/ASR/qwen_export/qwen3_export

# 3. 导出 ONNX 模型
python export/export_qwen3_separate.py \
    --device_str cpu \
    --dtype float16

# 4. 运行 QA Demo
python export/demo_qwen3_separate_qa.py

# 5. 验证 ONNX 模型结构
python -c "
import onnx
for name in ['qwen3_embed', 'qwen3_prefill', 'qwen3_decode']:
    m = onnx.load(f'output/onnx_qwen3_separate/{name}.onnx')
    print(f'=== {name} ===')
    for inp in m.graph.input:
        print(f'  IN: {inp.name} {[d.dim_value for d in inp.type.tensor_type.shape.dim]}')
    for out in m.graph.output:
        print(f'  OUT: {out.name} {[d.dim_value for d in out.type.tensor_type.shape.dim]}')
"
```
