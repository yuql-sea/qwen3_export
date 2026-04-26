# Qwen3-0.6B ONNX Export

Qwen3-0.6B 模型的 ONNX 分离导出项目，支持 Embedding / Prefill / Decode 三阶段推理。

本项目基于 [ascend-llm](https://gitee.com/yinghuo302/ascend-llm) 的参考实现。

## 项目结构

```
├── modeling_qwen3.py            # 核心模型定义（ONNX 导出版本）
├── export_qwen3_separate.py     # 分离模型导出脚本
├── demo_qwen3_separate_qa.py    # 端到端 QA 推理 Demo
├── QWEN3_ONNX_EXPORT_GUIDE.md   # 完整导出流程文档
├── ENVIRONMENT_REFERENCE.md     # 环境与工具速查
├── test_qwen3_0.6B.py           # PyTorch 原生推理测试
├── test_qwen3_onnx.py           # ONNX 推理测试
├── test_qwen3_qa.py             # QA 测试
├── test_qwen3_separate_compare.py   # ONNX vs PyTorch 精度对比
├── test_qwen3_separate_gpu.py   # GPU 推理测试
└── .gitignore
```

## 导出模型

| 模型 | 输入 | 输出 |
|------|------|------|
| Embedding | `input_ids [batch, seq]` | `inputs_embeds [batch, seq, 1024]` |
| Prefill | `input_embeds, attention_mask, position_ids` | `logits [batch, seq, 151936]`, `kv [batch, seq, 448, 128]` |
| Decode | `input_ids [batch, 1]`, `attention_mask, position_ids, past_kv` | `logits [batch, 1, 151936]`, `kv [batch, 1024, 448, 128]` |

## 环境

```bash
source /home/yuql/miniconda3/bin/activate OSUM
```

## 使用

```bash
# 导出 ONNX 模型（输出至 /home/yuql/workspace/ASR/onnx_hub/）
cd /home/yuql/workspace/ASR/qwen_export/qwen3_export
python export_qwen3_separate.py --device_str cpu --dtype float16

# 运行 QA Demo
python demo_qwen3_separate_qa.py
```

## 模型规格

| 参数 | 值 |
|------|-----|
| hidden_size | 1024 |
| num_attention_heads | 16 |
| num_key_value_heads | 8 |
| head_dim | 128 |
| num_hidden_layers | 28 |
| vocab_size | 151936 |
| GQA groups | 2 |

## 设计要点

- **Prefill/Decode 分离**：避免混合使用时的精度累积问题
- **Fixed-size KV Cache**：KV cache 输入输出均为固定长度（1024），通过 scatter write 更新，适合 Ascend 硬件部署
- **Decode 直接接受 input_ids**：无需外部 embedding 步骤
- **简单分支判断**：通过 `past_key_values is None` 区分 prefill / decode，避免 ONNX tracer 固化条件

## 文档

- [完整导出流程指南](QWEN3_ONNX_EXPORT_GUIDE.md)
- [环境与工具参考](ENVIRONMENT_REFERENCE.md)

## ONNX 输出路径

所有 ONNX 模型导出至：`/home/yuql/workspace/ASR/onnx_hub/`
