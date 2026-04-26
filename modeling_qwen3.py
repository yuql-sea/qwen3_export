# coding=utf-8
# Copyright 2025 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch Qwen3 model for ONNX/Ascend export. """
import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.activations import ACT2FN
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast, SequenceClassifierOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config


logger = logging.get_logger(__name__)


_CHECKPOINT_FOR_DOC = "Qwen/Qwen3-8B"
_CONFIG_FOR_DOC = "Qwen3Config"

QWEN3_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "Qwen/Qwen3-8B",
    # See all Qwen3 models at https://huggingface.co/models?filter=qwen3
]


# Copied from transformers.models.llama.modeling_llama._get_unpad_data
def _get_unpad_data(attention_mask):
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.torch.int32), (1, 0))
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )


class Qwen3RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Qwen3RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


# Copied from transformers.models.llama.modeling_llama.LlamaRotaryEmbedding with Llama->Qwen3
class Qwen3RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=4096, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        # Return cached cos/sin - don't regenerate based on seq_len
        # Cache was built with max_position_embeddings at init
        cos = self.cos_cached.to(dtype=x.dtype)
        sin = self.sin_cached.to(dtype=x.dtype)
        return cos, sin


# Copied from transformers.models.llama.modeling_llama.rotate_half
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


# Copied from transformers.models.llama.modeling_llama.apply_rotary_pos_emb
def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class Qwen3MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


# Copied from transformers.models.llama.modeling_llama.repeat_kv
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class Qwen3Attention(nn.Module):
    """
    Multi-headed attention from 'Attention Is All You Need' paper.
    Modified for ONNX/Ascend export with manual KV cache management.
    """

    def __init__(self, config: Qwen3Config, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing `layer_idx` is not recommended and will "
                "to errors during the forward call, if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        # Qwen3 uses head_dim from config directly (e.g., 128 in Qwen3-0.6B)
        self.head_dim = getattr(config, 'head_dim', self.hidden_size // self.num_heads)
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True
        self.attention_dropout = config.attention_dropout

        # Qwen3: num_attention_heads is the number of Q heads for output (grouped with KV heads)
        # For Qwen3-0.6B: hidden_size=1024, head_dim=128, num_attention_heads=16, num_key_value_heads=8
        # Q projection outputs: (num_attention_heads * head_dim) = 16 * 128 = 2048 -> reshaped to 16 heads
        # No divisibility check needed for Qwen3 as it uses a different architecture

        # Qwen3 uses attention_bias from config
        attention_bias = getattr(config, 'attention_bias', False)
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=attention_bias)

        # Qwen3 specific: Q-Norm and K-Norm
        self.q_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        self.rotary_emb = Qwen3RotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        # Qwen3: project first, then apply q_norm/k_norm BEFORE transpose (following official implementation)
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Reshape to [batch, seq, num_heads, head_dim]
        # Qwen3: query has num_heads=16, key/value has num_key_value_heads=8
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)

        # Apply q_norm/k_norm BEFORE transpose (per the official Qwen3 implementation)
        # This normalizes across the head_dim dimension within each head
        query_states = self.q_norm(query_states)  # [batch, seq, num_heads, head_dim]
        key_states = self.k_norm(key_states)        # [batch, seq, num_key_value_heads, head_dim]

        # Now transpose to [batch, num_heads, seq, head_dim]
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        past_length = past_key_value.shape[2] if past_key_value is not None else 0
        kv_seq_len = key_states.shape[-2] + past_length
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        # For incremental decoding, position_ids might only have current position
        # But we need cos/sin for all positions in key_states
        if position_ids is not None and position_ids.shape[-1] == 1 and key_states.shape[-2] > 1:
            # Incremental decoding: expand position_ids to match key_states length
            seq_len = key_states.shape[-2]
            position_ids_expanded = torch.arange(seq_len, device=position_ids.device, dtype=torch.long).unsqueeze(0).expand(position_ids.shape[0], -1)
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids_expanded)
        else:
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        # Only concatenate with past KV when we actually have past keys (past_length > 0)
        if past_key_value is not None and past_length > 0:
            # Fixed-size KV cache: scatter new KV into fixed-size cache at position_ids
            # past_key_value shape: [batch, 2*layers*kv_heads, KV_LEN, head_dim]
            # where first half is all keys, second half is all values
            num_layers = self.config.num_hidden_layers
            key_offset = self.layer_idx * self.num_key_value_heads
            value_offset = num_layers * self.num_key_value_heads + self.layer_idx * self.num_key_value_heads

            # Extract this layer's KV from the fixed-size cache
            # past_key_value shape: [batch, 2*layers*kv_heads, KV_LEN, head_dim]
            # Keys occupy first half of dim 1, values occupy second half
            # Slice dim 1 (heads) to extract this layer's kv_heads, keep full KV_LEN on dim 2
            cache_key = past_key_value[:, key_offset:key_offset + self.num_key_value_heads, :, :]
            cache_value = past_key_value[:, value_offset:value_offset + self.num_key_value_heads, :, :]
            # cache_key/value shape: [batch, num_kv_heads, KV_LEN, head_dim]

            # Scatter: write new KV at position_ids slot (arithmetic mask, ONNX-friendly)
            kv_cache_len = cache_key.shape[2]
            idx = torch.arange(kv_cache_len, device=hidden_states.device, dtype=torch.int64)
            idx = idx.view(1, 1, kv_cache_len, 1)     # [1, 1, KV_LEN, 1]
            pos = position_ids.view(bsz, 1, 1)          # [batch, 1, 1]
            write_mask = (idx == pos).to(dtype=hidden_states.dtype)
            # write_mask: [batch, 1, KV_LEN, 1], 1.0 at position_ids, 0.0 elsewhere

            # Fixed-size output: copy existing KV, overwrite at target position with new KV
            key_states = cache_key * (1.0 - write_mask) + key_states * write_mask
            value_states = cache_value * (1.0 - write_mask) + value_states * write_mask
            # Result: [batch, num_kv_heads, KV_LEN, head_dim] — same fixed size as input

        # output_cache is the fixed-size (key_states, value_states) for this layer
        output_cache = (key_states, value_states)

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        # Add attention mask if provided and dimensions match
        if attention_mask is not None:
            if attention_mask.shape[-1] == key_states.shape[-2]:
                attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2).contiguous()
        # After repeat_kv, attn_output has shape [batch, num_heads, seq, head_dim]
        # Reshape to [batch, seq, num_heads * head_dim] = [batch, seq, 2048] for Qwen3-0.6B
        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.head_dim)

        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights, output_cache


class Qwen3DecoderLayer(nn.Module):
    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = Qwen3Attention(config, layer_idx)
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)
        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        outputs += (present_key_value,)

        return outputs


QWEN3_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`Qwen3Config`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(
    "The bare Qwen3 Model outputting raw hidden-states without any specific head on top.",
    QWEN3_START_DOCSTRING,
)
class Qwen3PreTrainedModel(PreTrainedModel):
    config_class = Qwen3Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Qwen3DecoderLayer"]
    _skip_keys_device_placement = "past_key_values"

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


QWEN3_INPUTS_DOCSTRING = r"""
    Args:
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
            This is useful if you want more control over how to convert `input_ids` indices into associated vectors
            than the model's internal embedding lookup matrix.
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings.
        past_key_values (`torch.Tensor` of shape `[batch, 2*num_layers*num_kv_heads, seq, head_dim]`, *optional*):
            Pre-computed hidden-states (key and values in the self-attention blocks).
"""


@add_start_docstrings(
    "The bare Qwen3 Model outputting raw hidden-states without any specific head on top.",
    QWEN3_START_DOCSTRING,
)
class Qwen3Model(Qwen3PreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`Qwen3DecoderLayer`]

    Args:
        config: Qwen3Config
    """

    def __init__(self, config: Qwen3Config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Qwen3DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    @staticmethod
    def get_masks(inputs_embeds, past_length, padding_mask=None):
        """Generate causal attention mask for ONNX export."""
        batch_size, seq_length, _ = inputs_embeds.shape
        full_attention_mask = torch.ones(
            batch_size,
            seq_length,
            seq_length,
            device=inputs_embeds.device,
        )
        full_attention_mask.tril_()
        full_attention_mask = torch.cat(
            (
                torch.ones(
                    batch_size,
                    seq_length,
                    past_length,
                    device=inputs_embeds.device,
                ),
                full_attention_mask
            ),
            dim=-1
        )
        if padding_mask is not None:
            full_attention_mask = full_attention_mask * padding_mask.unsqueeze(1)
        full_attention_mask = (full_attention_mask < 0.5).bool()
        full_attention_mask.unsqueeze_(1)
        return full_attention_mask

    @add_start_docstrings_to_model_forward(QWEN3_INPUTS_DOCSTRING)
    def forward(
        self,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[torch.FloatTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        # transpose for past_kv_cache: [batch, 2*layers*kv_heads, seq, head_dim] -> [batch, seq, 2*layers*kv_heads, head_dim]
        # Handle past_key_values - transpose if provided
        # For prefill with no past, pass past_key_values=None
        # For incremental decode, pass the cached KV tensor
        _past_kv = past_key_values.transpose(1, 2) if past_key_values is not None else None

        bsz, q_len, _ = inputs_embeds.shape

        # Determine if we're doing prefill or incremental decode
        # past_key_values is None -> prefill (no KV cache)
        # past_key_values is not None -> decode (use KV cache)
        if past_key_values is not None:
            # Decode mode with fixed-size KV cache
            # attention_mask input: [batch, KV_LEN] (1.0=valid, 0.0=padding)
            # Convert to additive mask: [batch, 1, 1, KV_LEN] (0.0=attend, -10000.0=mask)
            if attention_mask is not None:
                attention_mask = (1.0 - attention_mask) * (-10000.0)
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        else:
            # Prefill: no past KV
            if attention_mask is None:
                # Default causal mask
                bsz, seq_len = inputs_embeds.shape[0], inputs_embeds.shape[1]
                full_attention_mask = torch.tril(torch.ones(seq_len, seq_len, device=inputs_embeds.device, dtype=torch.bool))
                full_attention_mask = full_attention_mask.unsqueeze(0).unsqueeze(0)
                attention_mask = torch.where(full_attention_mask, torch.tensor(0.0), torch.tensor(-10000.0)).float()
            else:
                # Use provided attention_mask [batch, seq] where 1=attend, 0=mask
                # Convert to [batch, 1, seq, seq] format
                bsz, seq_len = attention_mask.shape
                # Create causal mask
                causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=attention_mask.device, dtype=torch.bool))
                # Combine with padding mask: padding positions are 0, real positions are 1
                # padding_mask.unsqueeze(1) -> [batch, 1, seq]
                # causal_mask.unsqueeze(0) -> [1, seq, seq]
                # result: positions where either padding or upper triangle are masked
                padding_mask = attention_mask.unsqueeze(1).bool()  # [batch, 1, seq]
                combined_mask = causal_mask.unsqueeze(0) & padding_mask  # [batch, seq, seq]
                attention_mask = torch.where(combined_mask, torch.tensor(0.0), torch.tensor(-10000.0)).float()
                attention_mask = attention_mask.unsqueeze(1)  # [batch, 1, seq, seq]

        hidden_states = inputs_embeds

        # decoder layers
        presents = []
        for decoder_layer in self.layers:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=_past_kv,
            )

            hidden_states = layer_outputs[0]
            presents.append(layer_outputs[1])

        hidden_states = self.norm(hidden_states)

        # Concatenate all layer presents
        # Each present is (key_states, value_states) with shape [batch, kv_heads, seq, head_dim]
        # We need to group all ks together and all vs together: [k0, k1, ..., kN, v0, v1, ..., vN]
        all_keys = []
        all_values = []
        for k, v in presents:
            all_keys.append(k)
            all_values.append(v)
        # Shape: [batch, 2 * num_layers * kv_heads, seq, head_dim]
        presents = torch.cat(all_keys + all_values, dim=1)
        # Transpose to [batch, seq, 2 * num_layers * kv_heads, head_dim]
        presents = presents.transpose(1, 2)

        return (
            hidden_states,
            presents,
        )


class Qwen3ForCausalLM(Qwen3PreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen3Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @add_start_docstrings_to_model_forward(QWEN3_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[torch.FloatTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, Qwen3ForCausalLM

        >>> model = Qwen3ForCausalLM.from_pretrained("Qwen/Qwen3-8B")
        >>> tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to me."
        ```"""
        # If input_ids is provided but not inputs_embeds, embed the input_ids
        # Handle case where input_embeds is passed as first positional argument (ONNX export case)
        if inputs_embeds is None and input_ids is not None:
            if input_ids.dtype.is_floating_point:
                # input_ids is actually the embeddings (ONNX export passes input_embeds as first arg)
                inputs_embeds = input_ids
                input_ids = None
            else:
                # Normal case: input_ids is token IDs
                inputs_embeds = self.model.embed_tokens(input_ids)

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        output = (logits,) + outputs[1:]
        return output

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        # Omit tokens covered by past_key_values
        if past_key_values is not None:
            if isinstance(past_key_values, tuple):
                cache_length = past_length = past_key_values[0][0].shape[2]
                max_cache_length = None
            else:
                cache_length = past_length = past_key_values.shape[2]
                max_cache_length = None

            # Keep only the unprocessed tokens
            if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]

            if max_cache_length is not None and attention_mask is not None and cache_length + input_ids.shape[1] > max_cache_length:
                attention_mask = attention_mask[:, -max_cache_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past


@add_start_docstrings(
    """
    The Qwen3 Model transformer with a sequence classification head on top (linear layer).

    [`Qwen3ForSequenceClassification`] uses the last token in order to do the classification, as other causal models
    (e.g. GPT-2) do.
    """,
    QWEN3_START_DOCSTRING,
)
class Qwen3ForSequenceClassification(Qwen3PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = Qwen3Model(config)
        self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    @add_start_docstrings_to_model_forward(QWEN3_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]
        logits = self.score(hidden_states)

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                sequence_lengths = torch.eq(input_ids, self.config.pad_token_id).int().argmax(-1) - 1
                sequence_lengths = sequence_lengths % input_ids.shape[-1]
                sequence_lengths = sequence_lengths.to(logits.device)
            else:
                sequence_lengths = -1

        pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(pooled_logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(pooled_logits, labels)
        if not return_dict:
            output = (pooled_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )
