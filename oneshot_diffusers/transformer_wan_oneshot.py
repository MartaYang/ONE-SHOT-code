# Copyright 2025 The Wan Team and The HuggingFace Team. All rights reserved.
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

import math
import os
import glob
import json
from typing import Any, Dict, Optional, Tuple, Union
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open
from tqdm import tqdm
import re

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import FromOriginalModelMixin, PeftAdapterMixin
from diffusers.utils import USE_PEFT_BACKEND, logging, scale_lora_layers, unscale_lora_layers
from diffusers.models.attention import FeedForward
from diffusers.models.attention_processor import Attention
from diffusers.models.cache_utils import CacheMixin
from diffusers.models.embeddings import PixArtAlphaTextProjection, TimestepEmbedding, Timesteps, get_1d_rotary_pos_embed
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import FP32LayerNorm

from einops import rearrange

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

from .oneshot_util import HWLabelRotaryPosEmbedOffset, rope_precompute, precompute_freqs_cis_3d


# ADD FA4
try:
    from flash_attn.cute.interface import flash_attn_func as flash_attn4_func
    FLASH_ATTN_4_AVAILABLE = True
except Exception:
    print("!!!! Warning: flash_attn4_func is not available.")
    FLASH_ATTN_4_AVAILABLE = False


class WanAttnProcessor2_0:
    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("WanAttnProcessor2_0 requires PyTorch 2.0. To use it, please upgrade PyTorch to 2.0.")

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        half_dtypes = (torch.float16, torch.bfloat16)
        def half(x):
            return x if x.dtype in half_dtypes else x.to(torch.bfloat16)

        encoder_hidden_states_img = None
        encoder_hidden_states_mesh = None
        if (getattr(attn, "mesh_L", 0) > 0):    # only attn3 has mesh_L; mesh_L = qT*mesh_hw
            L_mesh = getattr(attn, "mesh_L", 0)
            encoder_hidden_states_mesh = encoder_hidden_states[:, :L_mesh]  # mesh latent [B, qT*mesh_hw, nHead*dim_head]; mesh is prepended before text
            # encoder_hidden_states = encoder_hidden_states[:, L_mesh:]       # text latent [B, 512, nHead*dim_head]
            assert encoder_hidden_states.shape[1] - L_mesh == 512                    # text is always 512 tokens, so this assert should always pass
            video_hw = getattr(attn, "video_hw", 0)
            mesh_hw = getattr(attn, "mesh_hw", 0)
            qT = getattr(attn, "mesh_qT", 0)                                # 21 (excluding ref frames)
            ref_image_num = getattr(attn, "ref_image_num", 0)
            assert encoder_hidden_states_mesh.shape[1] == qT * mesh_hw
            assert hidden_states.shape[1] == (qT+ref_image_num) * video_hw  # video latent includes ref [B, (qT+ref_image_num)*video_hw, nHead*dim_head]
        
        if encoder_hidden_states is None or encoder_hidden_states.shape[1] == 512:  # attn1 (self-attn) or attn2 (text cross-attn): original code unchanged
            if encoder_hidden_states is None:
                encoder_hidden_states = hidden_states

            query = attn.to_q(hidden_states)            #[B, (qT+1)*mesh_hw, nHead*dim_head]
            key = attn.to_k(encoder_hidden_states)      # attn2: KV from text latent; attn1: self-attn
            value = attn.to_v(encoder_hidden_states)

            if attn.norm_q is not None:
                query = attn.norm_q(query)
            if attn.norm_k is not None:
                key = attn.norm_k(key)

            query = query.unflatten(2, (attn.heads, -1)).transpose(1, 2)    #[B, (qT+1)*mesh_hw, nHead*dim_head] -> [B, nHead, (qT+1)*mesh_hw, dim_head]
            key = key.unflatten(2, (attn.heads, -1)).transpose(1, 2)
            value = value.unflatten(2, (attn.heads, -1)).transpose(1, 2)

            if rotary_emb is not None and encoder_hidden_states_mesh is None: # only used in attn1

                def apply_rotary_emb(hidden_states: torch.Tensor, freqs: torch.Tensor):
                    x_rotated = torch.view_as_complex(hidden_states.to(torch.float64).unflatten(3, (-1, 2)))
                    x_out = torch.view_as_real(x_rotated * freqs).flatten(3, 4)
                    return x_out.type_as(hidden_states)

                query = apply_rotary_emb(query, rotary_emb)
                key = apply_rotary_emb(key, rotary_emb)

            if FLASH_ATTN_4_AVAILABLE:
                hidden_states = flash_attn4_func(half(query.transpose(1, 2)), half(key.transpose(1, 2)), half(value.transpose(1, 2)), causal=False)
                if isinstance(hidden_states, tuple):
                    hidden_states = hidden_states[0]       # CuTe: returns (output, softmax_lse)
                hidden_states = hidden_states.transpose(1, 2)
            else:
                hidden_states = F.scaled_dot_product_attention(             # original text cross-attn [B, nHead, (qT+ref_image_num)*video_hw, dim_head]
                    query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
                )
            hidden_states = hidden_states.transpose(1, 2).flatten(2, 3) # [B, nHead, (qT+1)*mesh_hw, dim_head] -> [B, (qT+1)*mesh_hw, nHead*dim_head]
            hidden_states = hidden_states.type_as(query)

            # I2V task
            hidden_states_img = None
            if encoder_hidden_states_img is not None:               # not using i2v path
                raise NotImplementedError("we do not use I2V.")
                key_img = attn.add_k_proj(encoder_hidden_states_img)
                key_img = attn.norm_added_k(key_img)
                value_img = attn.add_v_proj(encoder_hidden_states_img)

                key_img = key_img.unflatten(2, (attn.heads, -1)).transpose(1, 2)
                value_img = value_img.unflatten(2, (attn.heads, -1)).transpose(1, 2)

                hidden_states_img = F.scaled_dot_product_attention(
                    query, key_img, value_img, attn_mask=None, dropout_p=0.0, is_causal=False
                )
                hidden_states_img = hidden_states_img.transpose(1, 2).flatten(2, 3)
                hidden_states_img = hidden_states_img.type_as(query)

                hidden_states = hidden_states + hidden_states_img

        else:   # attn3 (human mesh cross attn)
            query = attn.to_q(hidden_states)            #[B, (qT+1)*mesh_hw, nHead*dim_head]
            key_mesh = attn.to_k(encoder_hidden_states_mesh)      # attn2: KV from text latent; attn1: self-attn
            value_mesh = attn.to_v(encoder_hidden_states_mesh)

            if attn.norm_q is not None:
                query = attn.norm_q(query)
            if attn.norm_k is not None:
                key_mesh = attn.norm_k(key_mesh)

            query = query.unflatten(2, (attn.heads, -1)).transpose(1, 2)    #[B, (qT+1)*mesh_hw, nHead*dim_head] -> [B, nHead, (qT+1)*mesh_hw, dim_head]
            key_mesh   = key_mesh.unflatten(2, (attn.heads, -1)).transpose(1, 2)     # [B, nHead, qT*mesh_hw, dim_head] wrt [1, 12, 33264(=21*1584), 128]
            value_mesh = value_mesh.unflatten(2, (attn.heads, -1)).transpose(1, 2)   # [B, nHead, qT*mesh_hw, dim_head]

            # after_ref = 1 * video_hw    # ref is at frame 0; cross-attn only on the following 21 video frames
            seq_len_x = qT * video_hw   # ref is at the end; cross-attn only on the first 21 video frames

            # add RoPE to mesh and query tokens
            def apply_rotary_emb(hidden_states: torch.Tensor, freqs: torch.Tensor):
                x_rotated = torch.view_as_complex(hidden_states.to(torch.float64).unflatten(3, (-1, 2)))
                x_out = torch.view_as_real(x_rotated * freqs).flatten(3, 4)
                return x_out.type_as(hidden_states)
            _query = query[:,:,:seq_len_x] 
            # _query = query[:,:,after_ref:] # []
            # _query = apply_rotary_emb(_query, attn.rotary_emb_query_)
            # key_mesh = apply_rotary_emb(key_mesh, attn.rotary_emb_mesh)

            _query = rearrange(_query, "b h (t hw) c -> (b t) h hw c", t=qT)   #[B*qT, nHead, video_hw, dim_head] wrt [1*21, 12, 1584, 128])
            assert _query.shape[-2] == video_hw
            key_m   = rearrange(key_mesh,   "b h (t hw) c -> (b t) h hw c", t=qT)       #[B*qT, nHead, mesh_hw, dim_head] wrt [1*21, 12, 1584, 128]) 
            value_m = rearrange(value_mesh, "b h (t hw) c -> (b t) h hw c", t=qT)       #[B*qT, nHead, mesh_hw, dim_head]    

            if attn.rotary_emb_query_ is not None and attn.rotary_emb_mesh is not None:
                _query = apply_rotary_emb(_query, attn.rotary_emb_query_)
                key_m = apply_rotary_emb(key_m, attn.rotary_emb_mesh)

            if FLASH_ATTN_4_AVAILABLE:
                hidden_states_mesh = flash_attn4_func(half(_query.transpose(1, 2)), half(key_m.transpose(1, 2)), half(value_m.transpose(1, 2)), causal=False)
                if isinstance(hidden_states_mesh, tuple):
                    hidden_states_mesh = hidden_states_mesh[0]       # CuTe: returns (output, softmax_lse)
                hidden_states_mesh = hidden_states_mesh.transpose(1, 2)
            else:
                hidden_states_mesh = F.scaled_dot_product_attention(                        # [B*qT, nHead, video_hw, dim_head] mesh cross-attn to video query
                    _query, key_m, value_m, attn_mask=None, dropout_p=0.0, is_causal=False  
                )
            hidden_states_mesh = rearrange(hidden_states_mesh, "(b t) h hw c -> b h (t hw) c", t=qT)    #[B, nHead, qT*video_hw, dim_head]
            hidden_states_mesh = hidden_states_mesh.transpose(1, 2).flatten(2, 3)                       #[B, qT*video_hw, nHead*dim_head]
            hidden_states_mesh = hidden_states_mesh.type_as(_query)

            hidden_states_mesh_pad = torch.zeros_like(hidden_states)    # [B, (qT+ref)*video_hw, nHead*dim_head] pad to include ref frames, added to hidden_states
            # hidden_states_mesh_pad[:, after_ref:] = hidden_states_mesh  # ref portion is 0, remaining is mesh cross-attn result
            hidden_states_mesh_pad[:, :seq_len_x] = hidden_states_mesh  # ref portion is 0, leading portion is mesh cross-attn result

            hidden_states = hidden_states_mesh_pad   # mesh cross attn [B, (1+qT)*mesh_hw, nHead*dim_head]

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states


class WanImageEmbedding(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()

        self.norm1 = FP32LayerNorm(in_features)
        self.ff = FeedForward(in_features, out_features, mult=1, activation_fn="gelu")
        self.norm2 = FP32LayerNorm(out_features)

    def forward(self, encoder_hidden_states_image: torch.Tensor) -> torch.Tensor:
        hidden_states = self.norm1(encoder_hidden_states_image)
        hidden_states = self.ff(hidden_states)
        hidden_states = self.norm2(hidden_states)
        return hidden_states


class WanTimeTextImageEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        time_freq_dim: int,
        time_proj_dim: int,
        text_embed_dim: int,
        image_embed_dim: Optional[int] = None,
    ):
        super().__init__()

        self.timesteps_proj = Timesteps(num_channels=time_freq_dim, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.time_embedder = TimestepEmbedding(in_channels=time_freq_dim, time_embed_dim=dim)
        self.act_fn = nn.SiLU()
        self.time_proj = nn.Linear(dim, time_proj_dim)
        self.text_embedder = PixArtAlphaTextProjection(text_embed_dim, dim, act_fn="gelu_tanh")

        self.image_embedder = None
        if image_embed_dim is not None:
            self.image_embedder = WanImageEmbedding(image_embed_dim, dim)

    def forward(
        self,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
    ):
        timestep = self.timesteps_proj(timestep)

        time_embedder_dtype = next(iter(self.time_embedder.parameters())).dtype
        if timestep.dtype != time_embedder_dtype and time_embedder_dtype != torch.int8:
            timestep = timestep.to(time_embedder_dtype)
        temb = self.time_embedder(timestep).type_as(encoder_hidden_states)
        timestep_proj = self.time_proj(self.act_fn(temb))

        encoder_hidden_states = self.text_embedder(encoder_hidden_states)

        return temb, timestep_proj, encoder_hidden_states


class WanRotaryPosEmbed(nn.Module):
    def __init__(
        self, attention_head_dim: int, patch_size: Tuple[int, int, int], max_seq_len: int, theta: float = 10000.0
    ):
        super().__init__()

        self.attention_head_dim = attention_head_dim
        self.patch_size = patch_size
        self.max_seq_len = max_seq_len

        h_dim = w_dim = 2 * (attention_head_dim // 6)
        t_dim = attention_head_dim - h_dim - w_dim

        freqs = []
        for dim in [t_dim, h_dim, w_dim]:
            freq = get_1d_rotary_pos_embed(
                dim, max_seq_len, theta, use_real=False, repeat_interleave_real=False, freqs_dtype=torch.float64
            )
            freqs.append(freq)
        self.freqs = torch.cat(freqs, dim=1)

    def forward_old(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.patch_size
        ppf, pph, ppw = num_frames // p_t, height // p_h, width // p_w
        assert ppf > 0 and pph > 0 and ppw > 0, f"ppf/pph/ppw illegal: {ppf},{pph},{ppw}"

        # self.freqs = self.freqs.to(hidden_states.device)
        freqs_buf = self.freqs.to(hidden_states.device)  # avoid overwriting self.freqs
        freqs = freqs_buf.split_with_sizes(
            [
                self.attention_head_dim // 2 - 2 * (self.attention_head_dim // 6),
                self.attention_head_dim // 6,
                self.attention_head_dim // 6,
            ],
            dim=1,
        )

        freqs_f = freqs[0][:ppf].view(ppf, 1, 1, -1).expand(ppf, pph, ppw, -1)
        freqs_h = freqs[1][:pph].view(1, pph, 1, -1).expand(ppf, pph, ppw, -1)
        freqs_w = freqs[2][:ppw].view(1, 1, ppw, -1).expand(ppf, pph, ppw, -1)
        freqs = torch.cat([freqs_f, freqs_h, freqs_w], dim=-1).reshape(1, 1, ppf * pph * ppw, -1)
        return freqs
    
    def forward(self, hidden_states: torch.Tensor, offset_t=0) -> torch.Tensor:
        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.patch_size
        offset_ppf = offset_t // p_t
        ppf, pph, ppw = (num_frames+offset_t) // p_t, height // p_h, width // p_w

        self.freqs = self.freqs.to(hidden_states.device) # 1024, 64=(self.attention_head_dim//2)
        freqs = self.freqs.split_with_sizes(
            [
                self.attention_head_dim // 2 - 2 * (self.attention_head_dim // 6),
                self.attention_head_dim // 6,
                self.attention_head_dim // 6,
            ],
            dim=1,
        )

        # split dimensions by channel
        freqs_f = freqs[0][offset_ppf:ppf].view(ppf-offset_ppf, 1, 1, -1).expand(ppf-offset_ppf, pph, ppw, -1)
        freqs_h = freqs[1][:pph].view(1, pph, 1, -1).expand(ppf-offset_ppf, pph, ppw, -1)
        freqs_w = freqs[2][:ppw].view(1, 1, ppw, -1).expand(ppf-offset_ppf, pph, ppw, -1)
        freqs = torch.cat([freqs_f, freqs_h, freqs_w], dim=-1).reshape(1, 1, (ppf-offset_ppf) * pph * ppw, -1)
        return freqs



class WanTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        qk_norm: str = "rms_norm_across_heads",
        cross_attn_norm: bool = False,
        eps: float = 1e-6,
        added_kv_proj_dim: Optional[int] = None,
    ):
        super().__init__()

        # 1. Self-attention
        self.norm1 = FP32LayerNorm(dim, eps, elementwise_affine=False)
        self.attn1 = Attention(
            query_dim=dim,
            heads=num_heads,
            kv_heads=num_heads,
            dim_head=dim // num_heads,
            qk_norm=qk_norm,
            eps=eps,
            bias=True,
            cross_attention_dim=None,
            out_bias=True,
            processor=WanAttnProcessor2_0(),
        )

        # 2. Cross-attention (text)
        self.attn2 = Attention(
            query_dim=dim,
            heads=num_heads,
            kv_heads=num_heads,
            dim_head=dim // num_heads,
            qk_norm=qk_norm,
            eps=eps,
            bias=True,
            cross_attention_dim=None,
            out_bias=True,
            added_kv_proj_dim=None, # no mesh attention in attn2
            added_proj_bias=True,
            processor=WanAttnProcessor2_0(),
        )
        self.norm2 = FP32LayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()

        # 2.new Cross-attention (human mesh)
        if added_kv_proj_dim is not None: # OneshotWanTransformerBlock only; BaseWanTransformerBlock unchanged
            assert added_kv_proj_dim == dim # should be equal
            self.attn3 = Attention(
                query_dim=dim,
                heads=num_heads,
                kv_heads=num_heads,
                dim_head=dim // num_heads,
                qk_norm=qk_norm,
                eps=eps,
                bias=True,
                cross_attention_dim=None,
                out_bias=True,
                added_kv_proj_dim=None, # attn3 kv_proj is directly to_k/to_v
                added_proj_bias=True,
                processor=WanAttnProcessor2_0(),
            )
            self.norm_mesh = FP32LayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()
        else:
            self.attn3 = None

        # 3. Feed-forward
        self.ffn = FeedForward(dim, inner_dim=ffn_dim, activation_fn="gelu-approximate")
        self.norm3 = FP32LayerNorm(dim, eps, elementwise_affine=False)

        self.scale_shift_table = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        rotary_emb: torch.Tensor,
    ) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
            self.scale_shift_table + temb.float()
        ).chunk(6, dim=1)

        # 1. Self-attention
        norm_hidden_states = (self.norm1(hidden_states.float()) * (1 + scale_msa) + shift_msa).type_as(hidden_states)
        attn_output = self.attn1(hidden_states=norm_hidden_states, rotary_emb=rotary_emb)
        hidden_states = (hidden_states.float() + attn_output * gate_msa).type_as(hidden_states)

        # 2. Cross-attention (text)
        norm_hidden_states = self.norm2(hidden_states.float()).type_as(hidden_states)
        attn_output = self.attn2(hidden_states=norm_hidden_states, encoder_hidden_states=encoder_hidden_states[:, -512:]) # text is always 512 tokens, located at the end (mesh is prepended before text)
        hidden_states = hidden_states + attn_output


        # 2.new Cross-attention (human mesh)
        if self.attn3 is not None: # only OneshotWanTransformerBlock has attn3; BaseWanTransformerBlock unchanged
            norm_hidden_states = self.norm_mesh(hidden_states.float()).type_as(hidden_states)
            attn_output = self.attn3(hidden_states=norm_hidden_states, encoder_hidden_states=encoder_hidden_states)
            hidden_states = hidden_states + attn_output

        # 3. Feed-forward
        norm_hidden_states = (self.norm3(hidden_states.float()) * (1 + c_scale_msa) + c_shift_msa).type_as(
            hidden_states
        )
        ff_output = self.ffn(norm_hidden_states)
        hidden_states = (hidden_states.float() + ff_output.float() * c_gate_msa).type_as(hidden_states)

        return hidden_states


class BaseWanTransformerBlock(WanTransformerBlock):
    def __init__(
        self,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        qk_norm: str = "rms_norm_across_heads",
        cross_attn_norm: bool = False,
        eps: float = 1e-6,
        added_kv_proj_dim: Optional[int] = None,
        block_id: Optional[int] = None,
    ):
        super().__init__(dim, ffn_dim, num_heads, qk_norm, cross_attn_norm, eps, added_kv_proj_dim)
        self.block_id = block_id
    def forward(
        self,
        hidden_states: torch.Tensor,
        hints: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        rotary_emb: torch.Tensor,
        context_scale: float = 1.0,
    ) -> torch.Tensor:
        hidden_states = super().forward(hidden_states, encoder_hidden_states, temb, rotary_emb)
        if self.block_id is not None:
            hidden_states = hidden_states + hints[self.block_id] * context_scale
        return hidden_states


class OneshotWanTransformerBlock(WanTransformerBlock):
    def __init__(
        self,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        qk_norm: str = "rms_norm_across_heads",
        cross_attn_norm: bool = False,
        eps: float = 1e-6,
        added_kv_proj_dim: Optional[int] = None,
        block_id: Optional[int] = None,
    ):
        super().__init__(dim, ffn_dim, num_heads, qk_norm, cross_attn_norm, eps, added_kv_proj_dim)
        self.block_id = block_id
        if block_id == 0:
            self.proj_in = nn.Linear(dim, dim)
            nn.init.zeros_(self.proj_in.weight)
            nn.init.zeros_(self.proj_in.bias)
        self.proj_out = nn.Linear(dim, dim)
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)

    def forward(
        self,
        context: torch.Tensor,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        rotary_emb: torch.Tensor,
    ) -> torch.Tensor:
        if self.block_id != 0:
            assert len(list(torch.unbind(context))) > 0, f"ONESHOT[{self.block_id}] empty context stack"
        if self.block_id == 0:
            context = self.proj_in(context) + hidden_states
            all_context = []
        else:
            all_context = list(torch.unbind(context))
            context = all_context.pop(-1)
        context = super().forward(context, encoder_hidden_states, temb, rotary_emb)
        context_skip = self.proj_out(context)
        all_context += [context_skip, context]
        context = torch.stack(all_context)
        assert context.size(-1) > 0, f"ONESHOT[{self.block_id}] context last dim == 0 after stack"
        return context

class WanOneshotTransformer3DModel(ModelMixin, ConfigMixin, PeftAdapterMixin, FromOriginalModelMixin, CacheMixin):
    r"""
    A Transformer model for video-like data used in the Wan model.

    Args:
        patch_size (`Tuple[int]`, defaults to `(1, 2, 2)`):
            3D patch dimensions for video embedding (t_patch, h_patch, w_patch).
        num_attention_heads (`int`, defaults to `40`):
            Fixed length for text embeddings.
        attention_head_dim (`int`, defaults to `128`):
            The number of channels in each head.
        in_channels (`int`, defaults to `16`):
            The number of channels in the input.
        out_channels (`int`, defaults to `16`):
            The number of channels in the output.
        text_dim (`int`, defaults to `512`):
            Input dimension for text embeddings.
        freq_dim (`int`, defaults to `256`):
            Dimension for sinusoidal time embeddings.
        ffn_dim (`int`, defaults to `13824`):
            Intermediate dimension in feed-forward network.
        num_layers (`int`, defaults to `40`):
            The number of layers of transformer blocks to use.
        window_size (`Tuple[int]`, defaults to `(-1, -1)`):
            Window size for local attention (-1 indicates global attention).
        cross_attn_norm (`bool`, defaults to `True`):
            Enable cross-attention normalization.
        qk_norm (`bool`, defaults to `True`):
            Enable query/key normalization.
        eps (`float`, defaults to `1e-6`):
            Epsilon value for normalization layers.
        add_img_emb (`bool`, defaults to `False`):
            Whether to use img_emb.
        added_kv_proj_dim (`int`, *optional*, defaults to `None`):
            The number of channels to use for the added key and value projections. If `None`, no projection is used.
    """

    _supports_gradient_checkpointing = True
    _skip_layerwise_casting_patterns = ["patch_embedding", "condition_embedder", "norm"]
    _no_split_modules = ["WanTransformerBlock"]
    _keep_in_fp32_modules = ["time_embedder", "scale_shift_table", "norm1", "norm2", "norm3"]
    _keys_to_ignore_on_load_unexpected = ["norm_added_q"]

    @register_to_config
    def __init__(
        self,
        patch_size: Tuple[int] = (1, 2, 2),
        num_attention_heads: int = 40, # 12 for the 1.3B model
        attention_head_dim: int = 128,
        in_channels: int = 16,
        out_channels: int = 16,
        text_dim: int = 4096,
        freq_dim: int = 256,
        ffn_dim: int = 13824,
        num_layers: int = 40,
        cross_attn_norm: bool = True,
        qk_norm: Optional[str] = "rms_norm_across_heads",
        eps: float = 1e-6,
        image_dim: Optional[int] = None, #int = 1280, #
        added_kv_proj_dim: Optional[int] = None,
        added_kv_proj_mesh_dim: Optional[int] = None,
        rope_max_seq_len: int = 1024,
        # === read from config.json, no longer ignored ===
        oneshot_in_channels: Optional[int] = None,
        oneshot_layers: Optional[list] = None,
        image_token_length: int = 257,    # keep 257 as required
        is_finetune_mesh_patch_emb: bool = False,
    ) -> None:
        super().__init__()

        self.num_heads = num_attention_heads
        inner_dim = num_attention_heads * attention_head_dim
        self.inner_dim = inner_dim
        out_channels = out_channels or in_channels

        # 1. Patch & position embedding
        self.rope = WanRotaryPosEmbed(attention_head_dim, patch_size, rope_max_seq_len)
        self.patch_embedding = nn.Conv3d(in_channels, inner_dim, kernel_size=patch_size, stride=patch_size)

        # 2. Condition embeddings
        # image_embedding_dim=1280 for I2V model
        self.condition_embedder = WanTimeTextImageEmbedding(
            dim=inner_dim,
            time_freq_dim=freq_dim,
            time_proj_dim=inner_dim * 6,
            text_embed_dim=text_dim,
            image_embed_dim=image_dim,
        )

        # 3. Transformer blocks
        # === use oneshot_layers from config, fallback to default ===
        self.oneshot_layers = oneshot_layers if oneshot_layers is not None else [i for i in range(0, num_layers, 4)] # using oneshot_layers from config should be fine as ONESHOT reproduces correctly
        assert 0 in self.oneshot_layers
        self.oneshot_layers_mapping = {i: n for n, i in enumerate(self.oneshot_layers)}
        self.blocks = nn.ModuleList(
            [
                BaseWanTransformerBlock(
                    inner_dim, ffn_dim, num_attention_heads, qk_norm, cross_attn_norm, eps, added_kv_proj_dim, block_id=self.oneshot_layers_mapping[i] if i in self.oneshot_layers else None
                )
                for i in range(num_layers)
            ]
        )

        # 4. Output norm & projection
        self.norm_out = FP32LayerNorm(inner_dim, eps, elementwise_affine=False)
        self.proj_out = nn.Linear(inner_dim, out_channels * math.prod(patch_size))
        self.scale_shift_table = nn.Parameter(torch.randn(1, 2, inner_dim) / inner_dim**0.5)

        # 5. Add Oneshot Layer
        self.oneshot_in_dim = oneshot_in_channels #112 #+ 16  # in_channels
        self.oneshot_blocks = nn.ModuleList([
            OneshotWanTransformerBlock(
                    inner_dim, ffn_dim, num_attention_heads, qk_norm, cross_attn_norm, eps, added_kv_proj_mesh_dim if added_kv_proj_mesh_dim is not None else added_kv_proj_dim, block_id=i
                )
            # for i in self.oneshot_layers
            for i in (range(len(self.oneshot_layers)) if oneshot_layers is not None else self.oneshot_layers) 
        ])

        # oneshot patch embeddings
        self.oneshot_patch_embedding = nn.Conv3d(
            self.oneshot_in_dim, inner_dim, kernel_size=patch_size, stride=patch_size
        )

        self.is_finetune_mesh_patch_emb = is_finetune_mesh_patch_emb
        if is_finetune_mesh_patch_emb:
            # mesh patch embeddings [do not re-initialize]
            self.oneshot_mesh_patch_embedding = nn.Conv3d(
                in_channels, inner_dim, kernel_size=patch_size, stride=patch_size
            )

        self.gradient_checkpointing = False

        # rope frequencies used by motion / human-latent rope_precompute calls below
        self.freqs = torch.cat(precompute_freqs_cis_3d(inner_dim // num_attention_heads), dim=1)

        # # ref ID 
        # self.ref_id_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
        # self.ref_id_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=inner_dim)

    def patchify(self, x: torch.Tensor):
        grid_size = x.shape[2:]
        x = rearrange(x, 'b c f h w -> b (f h w) c').contiguous()
        return x, grid_size  # x, grid_size: (f, h, w)

    def unpatchify(self, x: torch.Tensor, grid_size: torch.Tensor):
        return rearrange(
            x,
            'b (f h w) (x y z c) -> b c (f x) (h y) (w z)',
            f=grid_size[0],
            h=grid_size[1],
            w=grid_size[2],
            x=self.patch_size[0],
            y=self.patch_size[1],
            z=self.patch_size[2]
        )

    def get_grid_sizes(self, grid_size_x, grid_size_ref, face_at_41=False):
        f, h, w = grid_size_x
        rf, rh, rw = grid_size_ref
        grid_sizes_x = torch.tensor([f, h, w], dtype=torch.long).unsqueeze(0)
        grid_sizes_x = [[torch.zeros_like(grid_sizes_x), grid_sizes_x, grid_sizes_x]]
        grid_sizes_ref = [[
            torch.tensor([30, 0, 0]).unsqueeze(0),
            torch.tensor([30+rf, rh, rw]).unsqueeze(0), # 31 -> 30+rf, supports multi-image reference
            torch.tensor([rf, rh, rw]).unsqueeze(0),    # 1 -> rf, supports multi-image reference
        ]]
        # face placed last at T=41
        if face_at_41:
            grid_sizes_face = [[
                torch.tensor([41, 0, 0]).unsqueeze(0),
                torch.tensor([41+1, rh, rw]).unsqueeze(0), # T=41; ref h and w are the same by default
                torch.tensor([1, rh, rw]).unsqueeze(0),    # 1
            ]]
            grid_sizes_ref = grid_sizes_ref + grid_sizes_face
        return grid_sizes_x + grid_sizes_ref

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_mesh: Optional[torch.Tensor] = None,
        oneshot_context: torch.Tensor = None,
        bbox_mask: Optional[torch.BoolTensor] = None,  # human bbox mask
        human_h_pos: Optional[torch.Tensor] = None,
        human_w_pos: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        ref_image_num: int = 1,
        # add_scene_ref: bool = False
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if attention_kwargs is not None and attention_kwargs.get("scale", None) is not None:
                logger.warning(
                    "Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective."
                )

        batch_size, num_channels, num_frames, height, width = hidden_states.shape   # (1, 16, 1+21, 80, 80)
        p_t, p_h, p_w = self.config.patch_size # [1,2,2] VAE spatial/temporal downsampling factors
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p_h
        post_patch_width = width // p_w

        # downsample bbox to patch grid for labeling RoPE
        if bbox_mask is not None:
            bbox_mask_p =F.interpolate(
                bbox_mask.float().unsqueeze(1),
                size=(post_patch_num_frames - ref_image_num, post_patch_height, post_patch_width),  # (D_out, H_out, W_out)
                mode="nearest-exact",
            ).squeeze(1)                   # [B, qT, Hp, Wp]
            bbox_mask_tokens = bbox_mask_p.reshape(batch_size, post_patch_num_frames - ref_image_num, post_patch_height*post_patch_width)
        else:
            bbox_mask_tokens = None
        if human_h_pos is not None:
            human_h_pos = human_h_pos.float()
            human_w_pos = human_w_pos.float()
            
        origin_ref_latents = hidden_states[:, :, 0:ref_image_num]
        x = hidden_states[:, :, ref_image_num:]

        # x: 21-frame video latent
        x, (f, h, w) = self.patchify(self.patch_embedding(x))  # torch.Size([1, 29120, 5120])
        seq_len_x = x.shape[1]

        # reference image
        ref_latents, (rf, rh, rw) = self.patchify(self.patch_embedding(origin_ref_latents))  # torch.Size([1, 1456, 5120])
        grid_sizes = self.get_grid_sizes((f, h, w), (rf-1, rh, rw), face_at_41=True)
        x = torch.cat([x, ref_latents], dim=1)

        # freqs
        pre_compute_freqs = rope_precompute(
            x.detach().view(1, x.size(1), self.num_heads, self.inner_dim // self.num_heads), grid_sizes, self.freqs, start=None
        )
        
        rotary_emb = pre_compute_freqs
        rotary_emb = rotary_emb[:,:,0].reshape(1,1,-1,64)

        # Oneshot embeddings
        oneshot_context_ref = oneshot_context[:,:,0:ref_image_num]
        oneshot_context = oneshot_context[:, :, ref_image_num:]
        oneshot_context, (f_, h_, w_) = self.patchify(self.oneshot_patch_embedding(oneshot_context)) 
        assert (f_, h_, w_) == (f, h, w)
        oneshot_context_ref, (rf_, rh_, rw_) = self.patchify(self.oneshot_patch_embedding(oneshot_context_ref))
        assert (rf_, rh_, rw_) == (rf, rh, rw)
        oneshot_context = torch.cat([oneshot_context, oneshot_context_ref], dim=1)


        # promt -> text latent
        temb, timestep_proj, encoder_hidden_states = self.condition_embedder( # get text latent
            timestep, encoder_hidden_states
        )
        timestep_proj = timestep_proj.unflatten(1, (6, -1))

        if encoder_hidden_states_mesh is not None:  # our mesh latent
            _,_,_, height_mesh, width_mesh = encoder_hidden_states_mesh.shape # (1, 16, 21, 80, 80)
            mesh_hw = (height_mesh // p_h) * (width_mesh // p_w)
            if self.is_finetune_mesh_patch_emb:     # if finetuning mesh patch embedding
                encoder_hidden_states_mesh = self.oneshot_mesh_patch_embedding(encoder_hidden_states_mesh) # self.patch_embedding(encoder_hidden_states_mesh) 
            else:
                encoder_hidden_states_mesh = self.patch_embedding(encoder_hidden_states_mesh)   # use WAN base patch embedding for mesh [B, nHead*dim_head, qT, mesh_h, mesh_w]
            encoder_hidden_states_mesh = encoder_hidden_states_mesh.flatten(2).transpose(1, 2) #torch.Size([B, qT*mesh_hw, nHead*dim_head]) > b,(thw),c
            encoder_hidden_states = torch.concat([encoder_hidden_states_mesh, encoder_hidden_states], dim=1) # mesh prepended before text (512 tokens); consistent when decoding

        # add HW-aware label RoPE
        if encoder_hidden_states_mesh is not None:  # RoPE for mesh and query tokens
            self.hwlable_rope = HWLabelRotaryPosEmbedOffset(self.attention_head_dim, theta=10000, bg_label=0, bg_pos=100, mesh_h=(height_mesh // p_h), mesh_w=(width_mesh // p_w))
            query_latents = torch.zeros_like(bbox_mask_p)     # background video tokens are 0
            query_latents[bbox_mask_p.bool()] = 1                 # human bbox region is 1
            rotary_emb_query_ = self.hwlable_rope(query_latents, human_h_pos.to(torch.float64), human_w_pos.to(torch.float64))
            # mesh
            mesh_latents = torch.ones((query_latents.shape[0], query_latents.shape[1], (height_mesh // p_h), (width_mesh // p_w))).to(hidden_states.device) # human region is 1
            rotary_emb_mesh = self.hwlable_rope(mesh_latents)

        # 4.1 OneshotTransformer blocks
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            for block in self.oneshot_blocks:
                a = block.attn3
                if a is not None:
                    # pass metadata to each block's cross-attn3 layer
                    a.video_hw      = post_patch_height*post_patch_width    # number of pixels in the video feature map
                    a.mesh_qT      = post_patch_num_frames - ref_image_num              # 21
                    a.ref_image_num = ref_image_num
                    if encoder_hidden_states_mesh is not None:
                        a.mesh_L       = encoder_hidden_states_mesh.shape[1]    # mesh token sequence length: qT*mesh_hw
                        a.mesh_hw      = mesh_hw                                # mesh feature map pixel count (may differ from video)
                        a.rotary_emb_mesh = rotary_emb_mesh
                        a.rotary_emb_query_ = rotary_emb_query_
                    else:
                        a.mesh_L       = 0
                        a.mesh_hw      = 0
                oneshot_context = self._gradient_checkpointing_func(
                    block, oneshot_context, x, encoder_hidden_states, timestep_proj, rotary_emb
                )
        else:
            for block in self.oneshot_blocks:
                a = block.attn3
                if a is not None:
                    # pass metadata to each block's cross-attn3 layer
                    a.video_hw      = post_patch_height*post_patch_width    # number of pixels in the video feature map
                    a.mesh_qT      = post_patch_num_frames - ref_image_num              # 21
                    a.ref_image_num = ref_image_num
                    if encoder_hidden_states_mesh is not None:
                        a.mesh_L       = encoder_hidden_states_mesh.shape[1]    # mesh token sequence length: qT*mesh_hw
                        a.mesh_hw      = mesh_hw                                # mesh feature map pixel count (may differ from video)
                        a.rotary_emb_mesh = rotary_emb_mesh
                        a.rotary_emb_query_ = rotary_emb_query_
                    else:
                        a.mesh_L       = 0
                        a.mesh_hw      = 0
                oneshot_context = block(oneshot_context, x, encoder_hidden_states, timestep_proj, rotary_emb)

        if encoder_hidden_states_mesh is not None:
            encoder_hidden_states = encoder_hidden_states[:, encoder_hidden_states_mesh.shape[1]:] # remove prepended mesh tokens; keep only text tokens for WAN base
        assert encoder_hidden_states.shape[1] == 512 # sanity check: text must be 512 tokens

        hints = torch.unbind(oneshot_context)[:-1]      # 


        # 4.2 BaseTransformer blocks
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            for block in self.blocks:
                x = self._gradient_checkpointing_func(
                    block, x, hints, encoder_hidden_states, timestep_proj, rotary_emb
                )
        else:
            for block in self.blocks:
                x = block(x, hints, encoder_hidden_states, timestep_proj, rotary_emb)
        # 5. Output norm, projection & unpatchify
        shift, scale = (self.scale_shift_table + temb.unsqueeze(1)).chunk(2, dim=1)

        # Move the shift and scale tensors to the same device as hidden_states.
        # When using multi-GPU inference via accelerate these will be on the
        # first device rather than the last device, which hidden_states ends up
        # on.
        shift = shift.to(x.device)
        scale = scale.to(x.device)

        x_video = x[:, :seq_len_x]
        x_ref = x[:, seq_len_x:seq_len_x+ref_latents.shape[1]]
        x = torch.cat([x_ref, x_video], dim=1) # consistent with ONESHOT: ref is prepended and included in loss

        x = (self.norm_out(x.float()) * (1 + scale) + shift).type_as(x)
        x = self.proj_out(x)

        x = x.reshape(
            batch_size, post_patch_num_frames, post_patch_height, post_patch_width, p_t, p_h, p_w, -1
        )
        x = x.permute(0, 7, 1, 4, 2, 5, 3, 6)
        output = x.flatten(6, 7).flatten(4, 5).flatten(2, 3)


        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)

    @classmethod
    def from_pretrained(    # loading should work as ONESHOT has been successfully reproduced
            cls,
            pretrained_model_path: Optional[Union[str, os.PathLike]],
            *args,
            **kwargs,
    ):
        pretrained_model_path = Path(pretrained_model_path)
        if pretrained_model_path.is_dir():
            # Support both being passed the model root (need /transformer/ subfolder)
            # and being passed the transformer subfolder directly (e.g. via
            # pipeline.from_pretrained which pre-joins the subfolder).
            if (pretrained_model_path / "config.json").is_file():
                transformer_dir = pretrained_model_path
            else:
                transformer_dir = pretrained_model_path / "transformer"
            config_path = transformer_dir / "config.json"
            with open(config_path, "r") as f:
                config = json.load(f)
                config['_class_name'] = 'WanOneshotTransformer3DModel'
                num_layers = config['num_layers']

            state_dict = {}
            ckpt_paths = (
                    transformer_dir
                    / "diffusion_pytorch_model*.safetensors"
            )
            dict_list = glob.glob(str(ckpt_paths))
            for dict_path in tqdm(dict_list):
                part_dict = {}
                with safe_open(dict_path, framework="pt", device="cpu") as f:
                    for k in f.keys():
                        part_dict[k] = f.get_tensor(k)
                state_dict.update(part_dict)

            has_oneshot_keys = any(
                k.startswith("oneshot_blocks.") or k.startswith("oneshot_patch_embedding")
                for k in state_dict.keys()
            )
            if not has_oneshot_keys:
                raise RuntimeError(
                    "Checkpoint does not contain ONESHOT keys (oneshot_blocks.* / oneshot_patch_embedding). "
                    "Please provide a merged ONESHOT checkpoint."
                )

            transformer = cls.from_config(config)
            missing_keys, unexpected_keys = transformer.load_state_dict(state_dict, strict=False)
            if cls._keys_to_ignore_on_load_unexpected is not None:
                for pat in cls._keys_to_ignore_on_load_unexpected:
                    unexpected_keys = [k for k in unexpected_keys if re.search(pat, k) is None]
            print("When initialize ONESHOT model, there are some unexpected keys and missing keys...")
            print("Missing keys: ", missing_keys)
            print("Unexpected keys: ", unexpected_keys)
        elif pretrained_model_path.is_file() and str(pretrained_model_path).endswith(
                ".safetensors"
        ):
            comfy_single_file_state_dict = {}
            with safe_open(pretrained_model_path, framework="pt", device="cpu") as f:
                metadata = f.metadata()
                for k in f.keys():
                    comfy_single_file_state_dict[k] = f.get_tensor(k)
            configs = json.loads(metadata["config"])
            transformer_config = configs["transformer"]
            transformer = cls.from_config(transformer_config)
            transformer.load_state_dict(comfy_single_file_state_dict)
        return transformer
