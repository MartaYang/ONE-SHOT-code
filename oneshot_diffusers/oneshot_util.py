import os
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from diffusers.models.embeddings import get_1d_rotary_pos_embed


class HWLabelRotaryPosEmbedOffset(nn.Module):
    """
    2D (H, W) + label-offset RoPE (local bbox version):

    - The H/W axes evenly split the complex dimension of head_dim//2;
    - Human regions: use local normalized coordinates inside the bbox (0~1), then rescale them to the mesh scale
      (0~mesh_h / 0~mesh_w), and add a different offset for each label so that the query_bbox and mesh_key
      of the same person are aligned in RoPE space;
    - Background: H/W positions are always set to bg_pos and do not carry spatial structural information;
      they only serve as a "background label";
    - Supports multiple labels (multiple people): each non-background label occupies a non-overlapping H/W segment.

    Input:
      mask_label: [B, qT, H, W]
        - background: == bg_label = 0
        - other values: label ids for different people, e.g., 1, 2, etc.

    Output:
      freqs: [B*qT, 1, H*W, head_dim//2] (complex128)
      Can be directly used by:

        def apply_rotary_emb(x, freqs):
            x_rot = torch.view_as_complex(
                x.to(torch.float64).unflatten(3, (-1, 2))
            )
            out = torch.view_as_real(x_rot * freqs).flatten(3, 4)
            return out.type_as(x)

      where x: [B*qT, nHead, H*W, head_dim]
    """

    def __init__(
        self,
        head_dim: int,
        theta: float = 10000.0,
        bg_label: float = 0,
        mesh_h: int = 40,
        mesh_w: int = 40,
        offset_step: float | None = None,  # Segment width occupied by each label
        bg_pos: float = 1000.0,           # Constant background position
        freqs_dtype: torch.dtype = torch.float64,
    ):
        super().__init__()
        assert head_dim % 2 == 0, "head_dim must be even for complex pairs."
        self.head_dim = head_dim
        self.theta = theta
        self.bg_label = bg_label
        self.mesh_h = mesh_h
        self.mesh_w = mesh_w
        self.freqs_dtype = freqs_dtype
        self.bg_pos = bg_pos

        complex_dim = head_dim // 2
        # 2D RoPE: H/W each take half, as evenly as possible
        self.h_dim = complex_dim // 2
        self.w_dim = complex_dim - self.h_dim

        # Label segment width, determining human1 [0, mesh_h], human2 [offset_step, offset_step + mesh_h], ...
        if offset_step is None:
            self.offset_step = float(max(mesh_h, mesh_w) + 20)  # e.g., 40 + 20 = 60
        else:
            self.offset_step = float(offset_step)

        # Align the complex dtype with freqs_dtype
        if self.freqs_dtype == torch.float64:
            self.complex_dtype = torch.complex128
        else:
            self.complex_dtype = torch.complex64

    def forward(
        self, 
        mask_label: torch.Tensor, 
        pos_h_map: Optional[torch.Tensor] = None, 
        pos_w_map: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        mask_label: [B, qT, H, W]
        Returns:
          freqs: [B*qT, 1, H*W, head_dim//2] (complex)
        """
        assert mask_label.dim() == 4, "mask_label should be [B, qT, H, W]"
        B, T, H, W = mask_label.shape
        BT = B * T
        L = H * W
        device = mask_label.device
        labels_bt = mask_label.view(BT, H, W)
        dtype_c = self.complex_dtype

        if pos_h_map is not None and pos_w_map is not None:
            # 1. Flatten all pixels and process them uniformly
            ph = pos_h_map.view(-1)  # [BT*L]
            pw = pos_w_map.view(-1)
            labels_flat = mask_label.view(-1)
            
            # 2. Distinguish background and human regions
            # According to the definition here, pos_map == -1 indicates background
            is_fg = (ph != -1) 
            
            # 3. Initialize the final position matrices, filling all positions with the background position by default
            final_pos_h = torch.full_like(ph, self.bg_pos, dtype=self.freqs_dtype)
            final_pos_w = torch.full_like(pw, self.bg_pos, dtype=self.freqs_dtype)

            # 4. Only compute for human regions (is_fg)
            if is_fg.any():
                # Extract the original 0-1 coordinates of human regions
                fg_ph = ph[is_fg]
                fg_pw = pw[is_fg]
                fg_labels = labels_flat[is_fg]

                # Compute offset: (label_idx - 1) * step
                # The background label is usually 0, and human labels start from 1, so subtract 1
                fg_offsets = (fg_labels - 1).clamp(min=0) * self.offset_step

                # Core logic: rescale to the mesh scale + add offset
                # y_scaled = y_local * (mesh_h - 1) + offset
                final_pos_h[is_fg] = fg_ph * (self.mesh_h - 1) + fg_offsets
                final_pos_w[is_fg] = fg_pw * (self.mesh_w - 1) + fg_offsets

            # 5. Compute RoPE in one fully parallelized pass, without the bt loop
            # Process the H axis
            if self.h_dim > 0:
                freq_h = get_1d_rotary_pos_embed(
                    dim=2 * self.h_dim,
                    pos=final_pos_h,
                    theta=self.theta,
                    use_real=False,
                    freqs_dtype=self.freqs_dtype,
                ).to(dtype_c) # [BT*L, h_dim]
            
            # Process the W axis
            if self.w_dim > 0:
                freq_w = get_1d_rotary_pos_embed(
                    dim=2 * self.w_dim,
                    pos=final_pos_w,
                    theta=self.theta,
                    use_real=False,
                    freqs_dtype=self.freqs_dtype,
                ).to(dtype_c) # [BT*L, w_dim]

            # 6. Concatenate and restore dimensions
            freqs = torch.cat([freq_h, freq_w], dim=-1) # [BT*L, head_dim//2]
            return freqs.view(BT, 1, L, self.head_dim // 2)

        # [BT, L, h_dim] / [BT, L, w_dim]
        freq_h_all = torch.ones(BT, L, self.h_dim, device=device, dtype=dtype_c)
        freq_w_all = torch.ones(BT, L, self.w_dim, device=device, dtype=dtype_c)

        for bt in range(BT):
            label_map = labels_bt[bt]                      # [H, W]
            label_map_f = label_map.to(self.freqs_dtype)

            # 1) Prepare H/W positions for each pixel, initialized entirely as the background position
            pos_h_map = torch.full(
                (H, W), self.bg_pos, device=device, dtype=self.freqs_dtype
            )
            pos_w_map = torch.full(
                (H, W), self.bg_pos, device=device, dtype=self.freqs_dtype
            )

            # 2) Find all non-background labels and assign an offset segment to each label
            all_labels = torch.unique(label_map_f)
            # Exclude the background
            human_labels = all_labels[all_labels != self.bg_label]

            # import ipdb; ipdb.set_trace()

            for _, label_idx in enumerate(human_labels):
                mask = (label_map_f == label_idx)  # [H, W]
                if not mask.any():
                    continue

                ys, xs = torch.where(mask)   # [N], [N]

                # Bbox range
                y_min, y_max = ys.min(), ys.max()
                x_min, x_max = xs.min(), xs.max()

                bbox_h = (y_max - y_min).item() + 1
                bbox_w = (x_max - x_min).item() + 1

                # Locally normalize to [0, 1]
                denom_h = max(bbox_h - 1, 1)
                denom_w = max(bbox_w - 1, 1)

                y_local = (ys - y_min).to(self.freqs_dtype) / denom_h  # [0, 1]
                x_local = (xs - x_min).to(self.freqs_dtype) / denom_w  # [0, 1]

                # Rescale to the mesh scale, e.g., [0, mesh_h - 1]
                y_scaled = y_local * (self.mesh_h - 1)
                x_scaled = x_local * (self.mesh_w - 1)

                # Label offset segment
                offset = (label_idx-1) * self.offset_step

                pos_h_map[ys, xs] = y_scaled + offset
                pos_w_map[ys, xs] = x_scaled + offset

            # 3) Flatten to [L]
            pos_h_flat = pos_h_map.reshape(L)  # [L]
            pos_w_flat = pos_w_map.reshape(L)  # [L]

            # 4) H/W-axis RoPE
            if self.h_dim > 0:
                freq_h_flat = get_1d_rotary_pos_embed(
                    dim=2 * self.h_dim,
                    pos=pos_h_flat,
                    theta=self.theta,
                    use_real=False,
                    freqs_dtype=self.freqs_dtype,
                )  # [L, h_dim] complex
                freq_h_all[bt] = freq_h_flat.to(dtype_c)

            if self.w_dim > 0:
                freq_w_flat = get_1d_rotary_pos_embed(
                    dim=2 * self.w_dim,
                    pos=pos_w_flat,
                    theta=self.theta,
                    use_real=False,
                    freqs_dtype=self.freqs_dtype,
                )  # [L, w_dim] complex
                freq_w_all[bt] = freq_w_flat.to(dtype_c)

        # 5) Concatenate into [BT, L, head_dim//2] → [BT, 1, L, head_dim//2]
        freqs = torch.cat([freq_h_all, freq_w_all], dim=-1)  # [BT, L, head_dim//2]
        freqs = freqs.view(BT, 1, L, self.head_dim // 2)
        return freqs


@torch.compiler.disable(recursive=True)
def rope_precompute(x, grid_sizes, freqs, start=None):
    b, s, n, c = x.size(0), x.size(1), x.size(2), x.size(3) // 2

    # split freqs
    if type(freqs) is list:
        trainable_freqs = freqs[1]
        freqs = freqs[0]
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # loop over samples
    output = torch.view_as_complex(x.detach().reshape(b, s, n, -1, 2).to(torch.float64))
    seq_bucket = [0]
    if not type(grid_sizes) is list:
        grid_sizes = [grid_sizes]
    for g in grid_sizes:
        if not type(g) is list:
            g = [torch.zeros_like(g), g]
        batch_size = g[0].shape[0]
        for i in range(batch_size):
            if start is None:
                f_o, h_o, w_o = g[0][i]
            else:
                f_o, h_o, w_o = start[i]

            f, h, w = g[1][i]
            t_f, t_h, t_w = g[2][i]
            seq_f, seq_h, seq_w = f - f_o, h - h_o, w - w_o
            seq_len = int(seq_f * seq_h * seq_w)
            if seq_len > 0:
                if t_f > 0:
                    factor_f, factor_h, factor_w = (t_f / seq_f).item(), (t_h / seq_h).item(), (t_w / seq_w).item()
                    # Generate a list of seq_f integers starting from f_o and ending at math.ceil(factor_f * seq_f.item() + f_o.item())
                    if f_o >= 0:
                        f_sam = np.linspace(f_o.item(), (t_f + f_o).item() - 1, seq_f).astype(int).tolist()
                    else:
                        f_sam = np.linspace(-f_o.item(), (-t_f - f_o).item() + 1, seq_f).astype(int).tolist()
                    h_sam = np.linspace(h_o.item(), (t_h + h_o).item() - 1, seq_h).astype(int).tolist()
                    w_sam = np.linspace(w_o.item(), (t_w + w_o).item() - 1, seq_w).astype(int).tolist()

                    assert f_o * f >= 0 and h_o * h >= 0 and w_o * w >= 0
                    freqs_0 = freqs[0][f_sam] if f_o >= 0 else freqs[0][f_sam].conj()
                    freqs_0 = freqs_0.view(seq_f, 1, 1, -1)

                    freqs_i = torch.cat(
                        [
                            freqs_0.expand(seq_f, seq_h, seq_w, -1),
                            freqs[1][h_sam].view(1, seq_h, 1, -1).expand(seq_f, seq_h, seq_w, -1),
                            freqs[2][w_sam].view(1, 1, seq_w, -1).expand(seq_f, seq_h, seq_w, -1),
                        ],
                        dim=-1
                    ).reshape(seq_len, 1, -1)
                elif t_f < 0:
                    freqs_i = trainable_freqs.unsqueeze(1)
                # apply rotary embedding
                output[i, seq_bucket[-1]:seq_bucket[-1] + seq_len] = freqs_i
        seq_bucket.append(seq_bucket[-1] + seq_len)
    return output


# copy from .wan_video_dit

# @lru_cache(maxsize=None)
def CHECK_HIGH_PRECISION_MODE():
    HIGH_PRECISION_MODE = os.getenv("HIGH_PRECISION_MODE", "false").lower() == "true"
    return HIGH_PRECISION_MODE

@torch.compiler.disable(recursive=True)
def precompute_freqs_cis_3d(dim: int, end: int = 1024, theta: float = 10000.0):
    # 3d rope precompute
    f_freqs_cis = precompute_freqs_cis(dim - 2 * (dim // 3), end, theta)
    h_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    w_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    return f_freqs_cis, h_freqs_cis, w_freqs_cis

@torch.compiler.disable(recursive=True)
@torch.amp.autocast(device_type="cuda", enabled=not CHECK_HIGH_PRECISION_MODE())
def precompute_freqs_cis(dim: int, end: int = 1024, theta: float = 10000.0):
    # 1d rope precompute
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)
                   [: (dim // 2)].double() / dim))
    freqs = torch.outer(torch.arange(end, device=freqs.device), freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis