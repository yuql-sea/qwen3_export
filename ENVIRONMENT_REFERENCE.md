# Qwen3-0.6B ONNX 导出 - 环境与工具参考

## 1. 运行环境

### 1.1 Conda 环境
```bash
# 推荐使用 OSUM 环境（已验证）
/home/yuql/miniconda3/envs/OSUM/bin/python

# 其他可用环境（torch 版本）
OSUM:        2.7.1+cu126  ✓
cosyvoice:   2.3.1+cu121
funasr-nano: 2.5.1
funasr_new:  2.3.1+cu121
ggml:        2.5.1+cpu
```

### 1.2 关键依赖
```bash
torch>=2.5.0
transformers>=4.50.0
onnx>=1.14.0
onnxruntime>=1.23.0
numpy>=1.24.0
```

### 1.3 模型路径
```
/home/yuql/.cache/modelscope/hub/models/Qwen/Qwen3-0.6B
```

### 1.4 输出目录
```
/home/yuql/workspace/ASR/onnx_hub/onnx_qwen3_separate/
```

---

## 2. 预期目标

### 2.1 导出目标
| 模型 | 输入 | 输出 |
|------|------|------|
| Embedding | `input_ids [batch, seq]` | `inputs_embeds [batch, seq, 1024]` |
| Prefill | `input_embeds`, `position_ids`, `past_kv=None` | `logits [batch, seq, 151936]`, `kv [batch, seq, 448, 128]` |
| Decode | `input_embeds [1,1,1024]`, `attention_mask`, `position_ids`, `past_kv` | `logits [1,1,151936]`, `kv [1, new_len, 448, 128]` |

### 2.2 模型规格
```
hidden_size = 1024
num_attention_heads = 16
num_key_value_heads = 8
head_dim = 128
num_hidden_layers = 28
vocab_size = 151936
```

### 2.3 精度要求
- ONNX 推理结果与 PyTorch 原生模型 logits 差异 < 0.01
- KV cache 累积差异 < 1.0（可接受）

---

## 3. 核心文件

### 3.1 代码文件
| 文件 | 说明 | 状态 |
|------|------|------|
| `export/modeling_qwen3.py` | 自定义 Qwen3 模型，用于 ONNX 导出 | 核心 |
| `export/export_qwen3_separate.py` | 分离模型导出脚本 | 待完善 |
| `export/demo_qwen3_separate_qa.py` | QA 测试 demo | 待适配 |
| `export/test_qwen3_0.6B.py` | PyTorch 原生测试脚本 | 参考 |

### 3.2 导出命令
```bash
cd /home/yuql/workspace/ASR/qwen_export/qwen3_export
/home/yuql/miniconda3/envs/OSUM/bin/python export/export_qwen3_separate.py \
    --device_str cpu \
    --dtype float16 \
    --prefill_seq_len 512
```

---

## 4. 已验证的修复

### 4.1 Attention Mask 类型错误
**位置**: `modeling_qwen3.py` - `Qwen3Model.forward`
```python
# 错误
attention_mask = torch.zeros_like(full_attention_mask, dtype=torch.float32)
# 正确
attention_mask = torch.zeros(full_attention_mask.shape, dtype=torch.float32, device=inputs_embeds.device)
```

### 4.2 KV Cache 维度转置
**位置**: `modeling_qwen3.py` - `Qwen3Attention.forward`
```python
cache_key = past_key_value[..., layer_offset:layer_offset + num_kv_heads, :].transpose(1, 2)
cache_value = past_key_value[..., layer_offset + num_kv_heads:layer_offset + 2 * num_kv_heads, :].transpose(1, 2)
```

### 4.3 ONNX Tracing 分支判断
**位置**: `modeling_qwen3.py` - `Qwen3Model.forward`
```python
# 错误：使用 .item() 导致 ONNX tracer 将条件当常量
_is_all_zero = (_past_kv.abs().max().item() == 0)

# 正确：直接用输入判断
if past_key_values is not None:
    # decode 分支
else:
    # prefill 分支
```

---

## 5. 待解决问题

### 5.1 Decode 模型导出
- **问题**: RuntimeError: tensor dimension mismatch (449 vs 2049)
- **可能原因**: KV cache concat 或 repeat_kv 逻辑问题
- **参考调试日志**: `export/ONNX_EXPORT_STATUS.md`

### 5.2 Demo 接口适配
- 当前 prefill 模型只有 2 个输入
- Demo 期望 4 个输入（需适配）

---

## 6. 参考文档

### 6.1 ONNX 导出
- [PyTorch ONNX Export](https://pytorch.org/docs/stable/onnx.html)
- [ONNX Runtime Python API](https://onnxruntime.ai/docs/api/python/)
- [torch.onnx.export troubleshooting](https://pytorch.org/docs/stable/onnx.html#troubleshooting)

### 6.2 Qwen3 模型
- [Qwen3 GitHub](https://github.com/QwenLM/Qwen3)
- [HuggingFace Qwen3](https://huggingface.co/Qwen/Qwen3-0.6B)

### 6.3 调试工具
```bash
# 查看 ONNX 模型结构
python -c "import onnx; m = onnx.load('model.onnx'); print(onnx.helper.printable_graph(m.graph))"

# ONNX 简化
python -m onnxsim input.onnx output.onnx

# Netron 可视化
# https://netron.app/
```

### 6.4 已有的调试报告
- `export/qwen3_onnx_debugging_report.md` - 详细调试历史
- `export/ONNX_EXPORT_STATUS.md` - 当前状态

---

## 7. 快速启动 checklist

- [ ] 激活 conda 环境: `source /home/yuql/miniconda3/bin/activate OSUM`
- [ ] 确认模型路径存在: `ls /home/yuql/.cache/modelscope/hub/models/Qwen/Qwen3-0.6B`
- [ ] 确认 modeling_qwen3.py 在 export 目录
- [ ] 运行导出脚本
- [ ] 验证 ONNX 模型结构
- [ ] 运行 demo 测试
