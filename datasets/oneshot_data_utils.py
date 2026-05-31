import cv2
import numpy as np
import torch
from typing import Tuple, List
import os


def take_bboxes_by_idx(bboxes_np: np.ndarray, idx: List[int]) -> np.ndarray:
    """
    Take a subsequence from full-sequence bboxes [T, 4] by idx, and automatically handle padding cases.
    """
    bboxes_np = np.asarray(bboxes_np)
    sub = bboxes_np[idx]  # If the last few items in idx are identical, this also repeats the last bbox
    return sub  # [F,4]


def scale_bboxes_xyxy(
    bboxes_t: torch.Tensor,         # [F,4], xyxy (pixel coordinates)
    src_size: Tuple[int, int],      # (H_src, W_src) -- the resolution currently aligned with these bboxes
    dst_size: Tuple[int, int],      # (H_dst, W_dst) -- target image resolution, i.e., the modality to be cropped
) -> torch.Tensor:
    """
    Linearly scale bbox coordinates from the source resolution src_size to the target resolution dst_size.
    """
    assert bboxes_t.ndim == 2 and bboxes_t.shape[-1] == 4
    Hs, Ws = src_size
    Hd, Wd = dst_size
    sx = float(Wd) / float(Ws)
    sy = float(Hd) / float(Hs)
    b = bboxes_t.clone()
    # Scale x by the W ratio and y by the H ratio
    b[:, 0] = b[:, 0] * sx
    b[:, 2] = b[:, 2] * sx
    b[:, 1] = b[:, 1] * sy
    b[:, 3] = b[:, 3] * sy
    return b


def crop_resize_by_bboxes_xyxy(
    video_frames: torch.Tensor,      # [F,C,H,W]
    bboxes_xyxy_px: torch.Tensor,    # [F,4] pixel coordinates in xyxy format, already aligned with H,W of video_frames
    target_size: Tuple[int, int],    # (H_out, W_out)
) -> torch.Tensor:
    """
    Crop each frame by its pixel-coordinate bbox and resize it to target_size.
    Assumes bboxes_xyxy_px is already aligned with the resolution of video_frames.
    If not, use scale_bboxes_xyxy first.
    """
    assert video_frames.ndim == 4, f"Expect [F,C,H,W], got {video_frames.shape}"
    F, C, H, W = video_frames.shape
    b = bboxes_xyxy_px.to(video_frames.device).float()

    # Clamp and ensure x2 > x1, y2 > y1
    b[:, 0] = b[:, 0].clamp(0, W - 1)  # x1
    b[:, 2] = b[:, 2].clamp(1, W)      # x2
    b[:, 1] = b[:, 1].clamp(0, H - 1)  # y1
    b[:, 3] = b[:, 3].clamp(1, H)      # y2
    b[:, 2] = torch.maximum(b[:, 2], b[:, 0] + 1)
    b[:, 3] = torch.maximum(b[:, 3], b[:, 1] + 1)

    outs = []
    for f in range(F):
        x1 = int(b[f, 0].item())
        y1 = int(b[f, 1].item())
        x2 = int(b[f, 2].item())
        y2 = int(b[f, 3].item())
        crop = video_frames[f:f+1, :, y1:y2, x1:x2]  # Note the order: [y, x]
        crop = torch.nn.functional.interpolate(
            crop, size=target_size, mode="bilinear", align_corners=True, antialias=True # 11.24: "bicubic"->"bilinear"
        )
        outs.append(crop)
    return torch.cat(outs, dim=0)  # [F,C,H_out,W_out]


def overlay_green_box(
    frames: torch.Tensor,
    boxes_xyxy: torch.Tensor,
    alpha: float = 0.5,
    thickness: int = 3,
):
    if frames is None:
        return

    F, C, H, W = frames.shape
    assert C >= 3

    # Use the original coordinates to check whether the box is out of bounds; do not clamp first
    boxes_int = boxes_xyxy.round().to(torch.int64)

    green = frames.new_tensor([0.0, 255.0, 0.0]).view(1, 3, 1, 1)
    t = int(thickness)
    if t <= 0:
        return frames

    for i in range(F):
        x1o, y1o, x2o, y2o = boxes_int[i].tolist()

        # Visible region used for slicing, equivalent to cropping the large canvas back to the original image window
        cx1 = max(0, min(x1o, W))
        cx2 = max(0, min(x2o, W))
        cy1 = max(0, min(y1o, H))
        cy2 = max(0, min(y2o, H))

        if cx2 <= cx1 or cy2 <= cy1:
            continue

        # -------- Top edge: draw only when the original y1 is inside the image --------
        if 0 <= y1o < H:
            y0 = y1o
            y1 = min(y1o + t, H)
            if y1 > y0:
                top = frames[i, :, y0:y1, cx1:cx2]
                frames[i, :, y0:y1, cx1:cx2] = top * (1.0 - alpha) + green * alpha

        # -------- Bottom edge: draw only when the original y2 is inside the image --------
        # Treat y2 as the boundary position here, allowing y2 == H
        if 0 < y2o <= H:
            y1 = y2o
            y0 = max(y2o - t, 0)
            if y1 > y0:
                bottom = frames[i, :, y0:y1, cx1:cx2]
                frames[i, :, y0:y1, cx1:cx2] = bottom * (1.0 - alpha) + green * alpha

        # -------- Left edge: draw only when the original x1 is inside the image --------
        if 0 <= x1o < W:
            x0 = x1o
            x1 = min(x1o + t, W)
            if x1 > x0:
                left = frames[i, :, cy1:cy2, x0:x1]
                frames[i, :, cy1:cy2, x0:x1] = left * (1.0 - alpha) + green * alpha

        # -------- Right edge: draw only when the original x2 is inside the image --------
        if 0 < x2o <= W:
            x1 = x2o
            x0 = max(x2o - t, 0)
            if x1 > x0:
                right = frames[i, :, cy1:cy2, x0:x1]
                frames[i, :, cy1:cy2, x0:x1] = right * (1.0 - alpha) + green * alpha

    return frames


# The following is code augmentation, temporarily not added
import torch.nn.functional as F
import torchvision.transforms.functional as tvF
from typing import Tuple, Optional


import numpy as np
from typing import Optional, Tuple, Union
import math


import torch
from typing import Optional, Sequence, Tuple


def generate_human_pos_maps(
    b_rp_orig_relative: torch.Tensor,  # (T,4) [x1,y1,x2,y2] in full-res coordinates
    new_H: int,
    new_W: int,
    downsample_factors: Optional[Sequence[int]] = None,  # e.g. [4,16,16]
):
    """
    Returns:
      if downsample_factors is None:
        human_pos_H, human_pos_W: (T, new_H, new_W)
      else:
        human_pos_H, human_pos_W: (T, new_H, new_W)
        human_pos_H_ds, human_pos_W_ds: (T_ds, new_H//ds_h, new_W//ds_w)
          where T_ds indices are [0, 1, 1+ds_t, 1+2*ds_t, ...] (< T)
    """
    assert b_rp_orig_relative.ndim == 2 and b_rp_orig_relative.shape[1] == 4
    T = b_rp_orig_relative.shape[0]

    # full-res
    human_pos_H, human_pos_W = _pos_maps_from_bboxes(b_rp_orig_relative, new_H, new_W)

    if downsample_factors is None:
        return human_pos_H, human_pos_W

    ds_t, ds_h, ds_w = map(int, downsample_factors)
    assert ds_t >= 1 and ds_h >= 1 and ds_w >= 1

    device = b_rp_orig_relative.device

    # 1) Temporal downsampling: keep frame 0, then take every ds_t frames from frame 1
    if ds_t == 1:
        t_idx = torch.arange(T, device=device, dtype=torch.long)
    else:
        t_idx = torch.cat([
            torch.zeros(1, device=device, dtype=torch.long),              # 0
            torch.arange(1, T, step=ds_t, device=device, dtype=torch.long) # 1, 1+ds_t, ...
        ], dim=0)  # (T_ds,)

    bboxes_t = b_rp_orig_relative.index_select(0, t_idx)  # (T_ds,4)

    # 2) Spatially downsampled grid size. Usually H/W can both be divided by 16;
    # here // is used to align with the actual downsampled feature-map size.
    H_ds = new_H // ds_h
    W_ds = new_W // ds_w
    if H_ds <= 0 or W_ds <= 0:
        raise ValueError(f"Downsample too large: got H_ds={H_ds}, W_ds={W_ds} from H={new_H},W={new_W}, ds={downsample_factors}")

    # 3) Scale bbox coordinates into the downsampled coordinate system by the same ratio
    bboxes_ds = bboxes_t.clone()
    bboxes_ds[:, 0] = bboxes_ds[:, 0] / float(ds_w)  # x1
    bboxes_ds[:, 2] = bboxes_ds[:, 2] / float(ds_w)  # x2
    bboxes_ds[:, 1] = bboxes_ds[:, 1] / float(ds_h)  # y1
    bboxes_ds[:, 3] = bboxes_ds[:, 3] / float(ds_h)  # y2

    human_pos_H_ds, human_pos_W_ds = _pos_maps_from_bboxes(bboxes_ds, H_ds, W_ds)

    return human_pos_H, human_pos_W, human_pos_H_ds, human_pos_W_ds

import torch
from typing import Optional, Sequence, Tuple

def _pos_maps_from_bboxes(bboxes_xyxy: torch.Tensor, H: int, W: int):
    """
    bboxes_xyxy: (T,4) float32/float16, [x1,y1,x2,y2] in the SAME coordinate system as the grid (0..W-1, 0..H-1)
    returns: human_pos_H, human_pos_W: (T,H,W), bbox-outside is -1
    """
    assert bboxes_xyxy.ndim == 2 and bboxes_xyxy.shape[1] == 4
    T = bboxes_xyxy.shape[0]
    device = bboxes_xyxy.device
    dtype = torch.float32 if bboxes_xyxy.dtype not in (torch.float16, torch.bfloat16, torch.float32) else bboxes_xyxy.dtype

    b = bboxes_xyxy.to(dtype)

    # (T,1,1)
    x_min = b[:, 0].view(T, 1, 1)
    y_min = b[:, 1].view(T, 1, 1)
    x_max = b[:, 2].view(T, 1, 1)
    y_max = b[:, 3].view(T, 1, 1)

    bw = (x_max - x_min).clamp(min=1e-6)
    bh = (y_max - y_min).clamp(min=1e-6)

    ys = torch.arange(H, device=device, dtype=dtype).view(1, H, 1)  # (1,H,1)
    xs = torch.arange(W, device=device, dtype=dtype).view(1, 1, W)  # (1,1,W)

    rel_W = (xs - x_min) / bw   # (T,1,W) -> broadcast to (T,H,W)
    rel_H = (ys - y_min) / bh   # (T,H,1) -> broadcast to (T,H,W)

    mask_x = (xs >= x_min) & (xs < x_max)  # (T,1,W)
    mask_y = (ys >= y_min) & (ys < y_max)  # (T,H,1)
    mask = mask_x & mask_y                 # (T,H,W)

    neg1 = torch.tensor(-1.0, device=device, dtype=dtype)
    human_pos_W = torch.where(mask, rel_W, neg1)
    human_pos_H = torch.where(mask, rel_H, neg1)
    return human_pos_H, human_pos_W


def axis_angle_to_matrix_cpu(axis_angle):
    """
    The most robust Rodrigues implementation: use stack to avoid slicing-assignment errors.
    """
    # Force conversion to CPU float32
    axis_angle = axis_angle.cpu().float()
    if axis_angle.ndim == 1:
        axis_angle = axis_angle.unsqueeze(0)
    
    N = axis_angle.shape[0]
    
    # 1. Angle and unit axis
    theta = torch.norm(axis_angle, dim=1, keepdim=True) # [N, 1]
    eps = 1e-8
    unit_axis = axis_angle / (theta + eps) # [N, 3]
    
    x = unit_axis[:, 0] # [N]
    y = unit_axis[:, 1] # [N]
    z = unit_axis[:, 2] # [N]
    zero = torch.zeros_like(x) # [N]
    
    # 2. Construct the skew-symmetric matrix K (N, 3, 3)
    # Construct each row via stack, completely avoiding the syntax K[:, i, j] = ...
    row0 = torch.stack([zero, -z, y], dim=1)  # [N, 3]
    row1 = torch.stack([z, zero, -x], dim=1)  # [N, 3]
    row2 = torch.stack([-y, x, zero], dim=1)  # [N, 3]
    K = torch.stack([row0, row1, row2], dim=1) # [N, 3, 3]
    
    # 3. Rodrigues formula computation
    I = torch.eye(3).unsqueeze(0).to(axis_angle.device) # [1, 3, 3]
    sin_t = torch.sin(theta).unsqueeze(2) # [N, 1, 1]
    cos_t = torch.cos(theta).unsqueeze(2) # [N, 1, 1]
    
    K2 = torch.bmm(K, K) # [N, 3, 3]
    
    # R = I + sin(theta)K + (1-cos(theta))K^2
    reg_rmats = I + sin_t * K + (1 - cos_t) * K2
    
    # 4. Special handling for zero rotation
    mask = (theta > 1e-6).float().view(N, 1, 1)
    return reg_rmats * mask + I * (1.0 - mask)


def select_ref_index_based_orientation(smplx_path, idx_list, k=3, select_fix_frame=None):
    data = np.load(smplx_path, allow_pickle=True)
    all_smpl_rotvec = data['smpl_rotvec']

    half_idx = len(all_smpl_rotvec) // 2
    assert any(x is not None for x in all_smpl_rotvec[half_idx:]), "Error: the second half of SMPL frames are all None"
    valid_ratio = np.mean([x is not None for x in all_smpl_rotvec])
    assert valid_ratio >= 0.4, f"Valid data ratio is too low: {valid_ratio}"

    valid_root_poses = []
    valid_global_idxs = []

    for i in idx_list:
        if i < len(all_smpl_rotvec):
            item = all_smpl_rotvec[i]
            if item is not None and (isinstance(item, np.ndarray) or isinstance(item, list)) and len(item) > 0:
                temp_arr = np.asarray(item).reshape(-1, 53, 3)
                if temp_arr.shape[0] > 0:
                    root_pose = temp_arr[0, 0]  # (3,)
                    valid_root_poses.append(root_pose)
                    valid_global_idxs.append(int(i))

    if len(valid_root_poses) <= k:
        return np.array(valid_global_idxs, dtype=np.int64)

    root_poses_np = np.stack(valid_root_poses).astype(np.float32)
    root_poses_tensor = torch.from_numpy(root_poses_np).cpu()

    rmats = axis_angle_to_matrix_cpu(root_poses_tensor).numpy()
    flat_rmats = rmats.reshape(len(valid_root_poses), 9)

    # Select the initial point; supports passing a global frame index
    # import ipdb; ipdb.set_trace()
    if select_fix_frame is not None:
        if select_fix_frame in valid_global_idxs:
            rand_ref_sub_idx = valid_global_idxs.index(select_fix_frame)
        else:
            # Provide a clearer error message
            raise ValueError(f"select_fix_frame={select_fix_frame} is not in valid_global_idxs; this frame may have smplx=None")
    else:
        rand_ref_sub_idx = np.random.randint(0, len(valid_root_poses))

    selected_sub_indices = [rand_ref_sub_idx]
    dist_to_set = np.sum((flat_rmats - flat_rmats[rand_ref_sub_idx])**2, axis=1)

    for _ in range(1, k):
        next_sub_idx = int(np.argmax(dist_to_set))
        selected_sub_indices.append(next_sub_idx)
        dist_to_set = np.minimum(dist_to_set, np.sum((flat_rmats - flat_rmats[next_sub_idx])**2, axis=1))

    # Key point: map subset indices back to global frame indices
    selected_global = np.array([valid_global_idxs[j] for j in selected_sub_indices], dtype=np.int64)
    return selected_global
    # return np.array([valid_global_idxs[i] for i in selected_sub_indices])

def _bbox_from_mask_tensor(mask_1hw: torch.Tensor, thr: float = 0.2):
    # mask_1hw: [1,H,W] or [H,W]
    if mask_1hw.dim() == 3:
        m = mask_1hw[0]
    else:
        m = mask_1hw
    ys, xs = torch.where(m > thr)
    if ys.numel() == 0:
        return None
    y1 = int(ys.min().item())
    y2 = int(ys.max().item()) + 1
    x1 = int(xs.min().item())
    x2 = int(xs.max().item()) + 1
    return torch.tensor([x1, y1, x2, y2], dtype=torch.float32)

def _match_bbox_aspect_xyxy(bbox_xyxy: torch.Tensor, img_hw, target_hw):
    # bbox_xyxy: [4] float32, img_hw=(H,W), target_hw=(Ht,Wt)
    H, W = img_hw
    Ht, Wt = target_hw
    x1, y1, x2, y2 = bbox_xyxy.tolist()
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    bw = max(2.0, (x2 - x1))
    bh = max(2.0, (y2 - y1))

    desired = Wt / float(Ht)
    cur = bw / float(bh)

    if cur > desired:
        # Too wide -> increase height
        bh = bw / desired
    else:
        # Too tall -> increase width
        bw = bh * desired

    nx1 = cx - 0.5 * bw
    nx2 = cx + 0.5 * bw
    ny1 = cy - 0.5 * bh
    ny2 = cy + 0.5 * bh

    # Clamp to image bounds
    nx1 = float(max(0.0, min(nx1, W - 1.0)))
    nx2 = float(max(1.0, min(nx2, W * 1.0)))
    ny1 = float(max(0.0, min(ny1, H - 1.0)))
    ny2 = float(max(1.0, min(ny2, H * 1.0)))

    return torch.tensor([nx1, ny1, nx2, ny2], dtype=torch.float32)


def get_three_ref_images_fullseq(
    video_reader,
    bboxes_cam_square_full,
    smplx_path,
    H_rp, W_rp,
    cur_resolution,
    k: int = 3,
    select_fix_frame: int | None = None,
    ref_crop_human_out: bool = False, # <--- Newly added parameter
):
    """
    Select k frames from the full sequence as references using FPS based on SMPLX root orientation,
    and return the cropped ref_image.

    Returns:
      - ref_image: torch.FloatTensor [k, 3, Ht, Wt], where Ht/Wt = cur_resolution
      - ref_global_idx: np.ndarray [k], global frame indices
      - ref_bbox_gt: torch.FloatTensor [k, 4], bbox at GT resolution, useful for debugging
    """
    # 1) Align the available full-sequence length.
    # First align video and bbox; smplx filtering with i < len is handled inside the selection function.
    T_video = len(video_reader)
    if bboxes_cam_square_full is None:
        raise ValueError("bboxes_cam_square_full is None, cannot crop ref images.")
    T_bbox = len(bboxes_cam_square_full)
    T_use = min(T_video, T_bbox)
    if T_use <= 0:
        raise ValueError(f"Invalid T_use={T_use} (T_video={T_video}, T_bbox={T_bbox})")

    idx_all = np.arange(T_use, dtype=np.int64)

    # 2) Select global frame indices. Requires select_ref_index_based_orientation to return global idx.
    ref_global_idx = select_ref_index_based_orientation(
        smplx_path, idx_all, k=k, select_fix_frame=select_fix_frame
    )
    ref_global_idx = np.array(ref_global_idx, dtype=np.int64)

    # 3) Pad to k frames if insufficient, preventing downstream asserts from crashing
    if len(ref_global_idx) == 0:
        raise ValueError("No valid SMPLX frames found for reference selection.")
    if len(ref_global_idx) < k:
        raise ValueError(f"Not enough valid SMPLX frames {len(ref_global_idx)} to select {k} references.")

    # 4) Directly read these k frames from the full video at GT resolution
    ref_frames = video_reader.get_batch(ref_global_idx.tolist())  # [k,H,W,3] uint8
    ref_frames = ref_frames.permute(0, 3, 1, 2).contiguous().float()  # [k,3,H_gt,W_gt]
    _, _, H_gt, W_gt = ref_frames.shape

    # 5) Take bbox at rp resolution H_rp/W_rp and scale it to GT resolution
    if isinstance(bboxes_cam_square_full, torch.Tensor):
        ref_bbox_rp = bboxes_cam_square_full[torch.from_numpy(ref_global_idx)].to(dtype=torch.float32)
    else:
        ref_bbox_rp = torch.tensor(bboxes_cam_square_full[ref_global_idx], dtype=torch.float32)

    ref_bbox_gt = scale_bboxes_xyxy(
        ref_bbox_rp,
        src_size=(H_rp, W_rp),
        dst_size=(H_gt, W_gt),
    )

    masks_tensor = None
    if ref_crop_human_out:
        masks_tensor = _load_and_process_masks(
            smplx_path,
            ref_global_idx,
            target_hw=(H_gt, W_gt),
            expand_kernel_size=3,
            dilate_iter=3,
            prefer_bbox_xyxy=ref_bbox_gt.cpu().numpy(),  # NEW: use bbox to select the main connected component
        )
        if masks_tensor is not None:
            masks_tensor = masks_tensor.to(ref_frames.device)
            ref_frames = ref_frames * masks_tensor

    ref_bbox_gt = scale_bboxes_xyxy(
        ref_bbox_rp,
        src_size=(H_rp, W_rp),
        dst_size=(H_gt, W_gt),
    )

    ref_bbox_for_crop = ref_bbox_gt

    if ref_crop_human_out and masks_tensor is not None:
        # Use the mask to find a tighter bbox, then match the aspect ratio of cur_resolution
        bxs = []
        for i in range(masks_tensor.shape[0]):
            bb = _bbox_from_mask_tensor(masks_tensor[i], thr=0.2)
            if bb is None:
                bb = ref_bbox_gt[i]  # fallback
            bb = _match_bbox_aspect_xyxy(bb, img_hw=(H_gt, W_gt), target_hw=cur_resolution)
            bxs.append(bb)
        ref_bbox_for_crop = torch.stack(bxs, dim=0).to(dtype=torch.float32, device=ref_bbox_gt.device)

    # 6) Crop and black-pad to cur_resolution
    ref_image = crop_blackpad_by_bboxes_xyxy(
        ref_frames,
        ref_bbox_for_crop,
        target_size=cur_resolution,
    )

    return ref_image, ref_global_idx, ref_bbox_gt


def crop_blackpad_by_bboxes_xyxy(
    video_frames: torch.Tensor,      # [F,C,H,W]
    bboxes_xyxy_px: torch.Tensor,    # [F,4] pixel coordinates in xyxy format
    target_size: Tuple[int, int],    # (H_out, W_out)
) -> torch.Tensor:
    """
    Crop each frame by pixel coordinates, preserve the aspect ratio during resizing,
    and pad missing areas on the right/bottom with black.
    """
    assert video_frames.ndim == 4, f"Expect [F,C,H,W], got {video_frames.shape}"
    F, C, H, W = video_frames.shape
    H_out, W_out = target_size
    device = video_frames.device
    dtype = video_frames.dtype
    
    b = bboxes_xyxy_px.to(device).float()

    # 1. Clamp boundaries to prevent out-of-bounds access
    b[:, 0] = b[:, 0].clamp(0, W - 1)  # x1
    b[:, 1] = b[:, 1].clamp(0, H - 1)  # y1
    b[:, 2] = b[:, 2].clamp(1, W)      # x2
    b[:, 3] = b[:, 3].clamp(1, H)      # y2
    
    # Ensure width and height are at least 1
    b[:, 2] = torch.maximum(b[:, 2], b[:, 0] + 1)
    b[:, 3] = torch.maximum(b[:, 3], b[:, 1] + 1)

    outs = []
    for f in range(F):
        x1, y1, x2, y2 = b[f].int().tolist()
        
        # 2. Crop the original region
        crop = video_frames[f:f+1, :, y1:y2, x1:x2]  # [1, C, h_crop, w_crop]
        h_crop = y2 - y1
        w_crop = x2 - x1

        # 3. Compute the aspect-ratio-preserving scale factor using the smaller width/height scale ratio
        scale = min(H_out / h_crop, W_out / w_crop)
        new_h = min(int(h_crop * scale), H_out)
        new_w = min(int(w_crop * scale), W_out)

        # 4. Resize the cropped image
        # Use bilinear for smoothness, and antialias=True to prevent aliasing
        resized_crop = torch.nn.functional.interpolate(
            crop, size=(new_h, new_w), mode="bilinear", align_corners=True, antialias=True
        )

        # 5. Create a black background canvas and place the image.
        # It is aligned to the top-left by default, so the right/bottom areas naturally remain black.
        canvas = torch.zeros((1, C, H_out, W_out), device=device, dtype=dtype)
        canvas[:, :, :new_h, :new_w] = resized_crop
        
        outs.append(canvas)

    return torch.cat(outs, dim=0)  # [F,C,H_out,W_out]


def _invalid_mask_all_minus1(bboxes: np.ndarray) -> np.ndarray:
    """Missing values are uniformly represented as [-1,-1,-1,-1]; geometrically invalid boxes are also treated as invalid."""
    b = bboxes
    miss = np.all(b == -1, axis=1)
    # For valid frames: max <= min is also a bad box
    bad_geom = (b[:, 2] <= b[:, 0]) | (b[:, 3] <= b[:, 1])
    return miss | bad_geom

def _moving_average_1d(x: np.ndarray, win: int) -> np.ndarray:
    """O(T) sliding-window average with edge padding at boundaries."""
    if win <= 1:
        return x.astype(np.float32)
    win = int(win)
    if win % 2 == 0:
        win += 1
    pad = win // 2
    xp = np.pad(x.astype(np.float32), (pad, pad), mode="edge")
    kernel = np.ones(win, dtype=np.float32) / win
    return np.convolve(xp, kernel, mode="valid").astype(np.float32)

def _ema_1d(x: np.ndarray, alpha: float) -> np.ndarray:
    if len(x) == 0:
        return x.astype(np.float32)
    x = x.astype(np.float32)
    y = np.empty_like(x, dtype=np.float32)
    y[0] = x[0]
    for i in range(1, len(x)):
        y[i] = alpha * x[i] + (1 - alpha) * y[i - 1]
    return y

def _interp_with_gap_limit(arr: np.ndarray, invalid: np.ndarray, max_gap: int | None):
    """
    Interpolate arr at invalid positions, but if the missing segment length is > max_gap,
    do not fill that segment and keep it invalid in the output as -1.

    Returns:
      arr_filled: short missing segments are filled; long missing segments are NaN
      long_gap_mask: frames belonging to long missing segments
    """
    T = len(arr)
    a = arr.astype(np.float32).copy()

    # Compute the long-missing-segment mask. Only mark long missing segments;
    # do not initialize it as invalid directly.
    long_gap = np.zeros(T, dtype=bool)
    if max_gap is not None:
        max_gap = int(max_gap)
        start = None
        for i in range(T + 1):
            cur = invalid[i] if i < T else False
            if cur and start is None:
                start = i
            if (not cur) and start is not None:
                end = i
                if (end - start) > max_gap:
                    long_gap[start:end] = True
                start = None

    # First set all invalid values to NaN for interpolation
    a[invalid] = np.nan
    ok = ~np.isnan(a)
    if ok.sum() == 0:
        # All values are missing
        return a, invalid  # Treat all frames as unavailable
    
    # Endpoint filling to ensure interp does not fail
    idx = np.where(ok)[0]
    first, last = idx[0], idx[-1]
    a[:first] = a[first]
    a[last+1:] = a[last]

    # Interpolation: only fill short missing segments, i.e., invalid but not in long_gap
    to_fill = invalid & (~long_gap)
    if to_fill.any():
        xs = np.arange(T, dtype=np.float32)
        ok2 = ~np.isnan(a)
        a[to_fill] = np.interp(xs[to_fill], xs[ok2], a[ok2]).astype(np.float32)

    # Keep long missing segments as NaN; they will be converted back to -1 later
    a[long_gap] = np.nan
    return a, long_gap

def smooth_bboxes_breathing(
    bboxes: np.ndarray,
    *,
    min_valid_ratio: float = 0.5,
    max_gap_interp: int = 30,
    smooth_win: int = 7,
    ema_alpha: float = 0.7,
    min_wh: float = 2.0,
) -> np.ndarray:
    """
    Goal: remove small bbox jitters / breathing artifacts, while allowing out-of-frame boxes.
    Missing values are represented as [-1,-1,-1,-1].

    - valid frame ratio < min_valid_ratio -> raise
    - consecutive missing frames > max_gap_interp -> do not interpolate; still output -1
    """
    bboxes = np.asarray(bboxes).astype(np.float32)
    if bboxes.ndim != 2 or bboxes.shape[1] != 4:
        raise ValueError(f"bboxes must be (T,4), got {bboxes.shape}")

    T = bboxes.shape[0]
    invalid = _invalid_mask_all_minus1(bboxes)
    valid_ratio = float((~invalid).mean()) if T > 0 else 0.0
    if valid_ratio < float(min_valid_ratio):
        raise ValueError(f"Too many invalid bboxes: valid_ratio={valid_ratio:.3f} < {min_valid_ratio}")

    x1, y1, x2, y2 = [bboxes[:, i].copy() for i in range(4)]
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    w  = (x2 - x1)
    h  = (y2 - y1)

    # Interpolate missing values, but do not fill long missing segments
    cx, long_gap = _interp_with_gap_limit(cx, invalid, max_gap_interp)
    cy, _            = _interp_with_gap_limit(cy, invalid, max_gap_interp)
    w,  _            = _interp_with_gap_limit(w,  invalid, max_gap_interp)
    h,  _            = _interp_with_gap_limit(h,  invalid, max_gap_interp)

    # Minimum size, applied only to non-NaN values
    w = np.where(np.isnan(w), w, np.maximum(w, min_wh))
    h = np.where(np.isnan(h), h, np.maximum(h, min_wh))

    # Smoothing: sliding-window mean + EMA for gentle jitter removal
    # Note: do not smooth NaN segments directly.
    # We temporarily fill NaN with 0 and then restore NaN to avoid convolution spreading NaN.
    def smooth_arr(a: np.ndarray) -> np.ndarray:
        nan_mask = np.isnan(a)
        if nan_mask.all():
            return a
        a2 = a.copy()
        # Fill NaN with the nearest non-NaN values, only for smoothing computation;
        # this does not change the final output for invalid_final segments.
        idx = np.where(~nan_mask)[0]
        first, last = idx[0], idx[-1]
        a2[:first] = a2[first]
        a2[last+1:] = a2[last]
        # Linearly fill intermediate NaNs
        if nan_mask.any():
            xs = np.arange(T, dtype=np.float32)
            a2[nan_mask] = np.interp(xs[nan_mask], xs[~nan_mask], a2[~nan_mask]).astype(np.float32)

        a2 = _moving_average_1d(a2, smooth_win)
        a2 = _ema_1d(a2, ema_alpha)
        a2[nan_mask] = np.nan
        return a2

    cx = smooth_arr(cx)
    cy = smooth_arr(cy)
    w  = smooth_arr(w)
    h  = smooth_arr(h)

    # Convert back to bbox
    out = np.stack([cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h], axis=1).astype(np.float32)

    # Long missing segments / originally invalid segments: output -1
    out[np.isnan(out).any(axis=1) | long_gap] = np.array([-1, -1, -1, -1], dtype=np.float32)
    return out

import torch
from typing import Tuple, Optional


def get_static_center_crop_bboxes(human_bboxes, H, W, wh_ratio=1.0):
    """
    For videos with dynamic cameras: do not perform camera panning;
    only apply the largest center crop to match the target aspect ratio.
    """
    device = human_bboxes.device
    T = human_bboxes.shape[0]
    
    # 1. Compute current video aspect ratio vs. target aspect ratio
    video_ratio = W / H
    if video_ratio > wh_ratio:
        # The video is wider -> use full height and crop width
        crop_h = float(H)
        crop_w = crop_h * wh_ratio
    else:
        # The video is taller -> use full width and crop height
        crop_w = float(W)
        crop_h = crop_w / wh_ratio

    # 2. Compute the centered crop box, fixed for all frames
    center_x, center_y = W / 2.0, H / 2.0
    x1 = center_x - crop_w / 2.0
    y1 = center_y - crop_h / 2.0
    x2 = center_x + crop_w / 2.0
    y2 = center_y + crop_h / 2.0
    
    # [x1, y1, x2, y2]
    box_single = torch.tensor([x1, y1, x2, y2], device=device, dtype=torch.float32)
    b_rp = box_single.unsqueeze(0).repeat(T, 1) # [T, 4]

    # 3. Compute relative coordinates: human absolute coordinates - crop box top-left corner
    # In dynamic videos, the person may go out of frame, so coordinates may be negative or exceed crop_w/h; this is normal.
    crop_top_left = b_rp[:, :2].repeat(1, 2)
    b_rp_orig_relative = human_bboxes - crop_top_left

    return b_rp, b_rp_orig_relative


def _load_and_process_masks(smplx_path, selected_indices, target_hw, expand_kernel_size=3, dilate_iter=2, prefer_bbox_xyxy=None):
    """
    Helper function: read, decompress, dilate, and resize masks.
    """
    npz_path = os.path.join(os.path.dirname(smplx_path), "camera_and_humanmask.npz")
    if not os.path.exists(npz_path):
        print(f"[Warning] Mask file not found at {npz_path}, skipping masking.")
        return None

    try:
        data = np.load(npz_path, allow_pickle=True)
        packed_mask = data['mask_packbits']
        mask_shape = data['mask_shape']     # [N, H, W]
        # bitorder = str(data['bitorder']) # Some older numpy versions may store this as an array
        bitorder = "little" # Usually little by default, or fixed according to your saving code

        N_total, H_mask, W_mask = mask_shape
        
        # 1. Only decompress the required frames for speed.
        # It is also possible to decompress all frames and then index them, depending on GPU/CPU memory.
        # For robustness, we decompress all frames first here, since masks are usually not large.
        # Note: unpackbits only handles uint8, and the output dimension is flattened, so reshape is required.
        unpacked = np.unpackbits(packed_mask, axis=-1, bitorder=bitorder)
        # Crop out padding bits
        all_masks = unpacked[:, :, :W_mask].reshape(N_total, H_mask, W_mask)
        
        # Extract the selected frames
        selected_masks = all_masks[selected_indices] # [k, H_m, W_m]
        
        processed_masks = []
        target_h, target_w = target_hw
        
        # Define dilation kernel
        kernel = np.ones((expand_kernel_size, expand_kernel_size), np.uint8)

        for idx_in_k, m in enumerate(selected_masks):
            # === NEW: Select the main connected component, preferably the one with the largest overlap with bbox ===
            m_bin = (m > 0).astype(np.uint8)  # [H_mask,W_mask] 0/1
            num, labels, stats, _ = cv2.connectedComponentsWithStats(m_bin, connectivity=8)
            if num > 1:
                chosen = None
                if prefer_bbox_xyxy is not None:
                    # Map bbox from target_hw to mask resolution (H_mask, W_mask).
                    # prefer_bbox_xyxy is in the coordinate system of target_hw=(target_h,target_w).
                    x1, y1, x2, y2 = prefer_bbox_xyxy[idx_in_k]  # Need to enumerate in the outer loop; see below
                    target_h, target_w = target_hw
                    sx = W_mask / float(target_w)
                    sy = H_mask / float(target_h)

                    bx1 = int(np.clip(x1 * sx, 0, W_mask - 1))
                    bx2 = int(np.clip(x2 * sx, 0, W_mask))
                    by1 = int(np.clip(y1 * sy, 0, H_mask - 1))
                    by2 = int(np.clip(y2 * sy, 0, H_mask))

                    if bx2 > bx1 and by2 > by1:
                        bbox_region = np.zeros((H_mask, W_mask), dtype=bool)
                        bbox_region[by1:by2, bx1:bx2] = True

                        best_overlap = 0
                        best_lab = None
                        for lab in range(1, num):
                            overlap = np.count_nonzero((labels == lab) & bbox_region)
                            if overlap > best_overlap:
                                best_overlap = overlap
                                best_lab = lab
                        if best_lab is not None and best_overlap > 0:
                            chosen = best_lab
                if chosen is None:
                    # Fallback: select the largest area, excluding background label=0
                    areas = stats[1:, cv2.CC_STAT_AREA]
                    chosen = 1 + int(np.argmax(areas))
                m = (labels == chosen).astype(np.uint8)  # Write back as 0/1

            # A. Convert to 0-255 for processing
            m_img = (m * 255).astype(np.uint8)
            
            # B. Dilate to address the issue of "covering human boundaries"
            # iterations=1 means dilating once; set it to 2 if more dilation is needed.
            m_dilated = cv2.dilate(m_img, kernel, iterations=dilate_iter)
            
            # C. Resize to the original image size to reduce jagged edges
            # INTER_LINEAR produces transitional gray values at boundaries, forming a soft mask.
            m_resized = cv2.resize(m_dilated, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            
            # D. Normalize back to 0.0-1.0
            m_float = m_resized.astype(np.float32) / 255.0
            processed_masks.append(torch.from_numpy(m_float))
            
        # Stack and add the channel dimension: [k, 1, H, W]
        return torch.stack(processed_masks).unsqueeze(1)

    except Exception as e:
        print(f"[Error] Failed to load/process masks: {e}")
        return None

import random
import torch
import torch.nn.functional as F
import cv2


# oneshot_data_utils.py


from pathlib import Path

def _xyxy_from_points(px, py):
    x1 = float(np.min(px)); x2 = float(np.max(px))
    y1 = float(np.min(py)); y2 = float(np.max(py))
    return x1, y1, x2, y2

def _expand_to_aspect_xyxy(x1, y1, x2, y2, target_ratio, scale=1.25):
    """
    With the center kept unchanged:
      1) First enlarge the box by the overall scale
      2) Then adjust the bbox to target_ratio = W/H to fill as much as possible and reduce padding
    """
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    w = (x2 - x1) * scale
    h = (y2 - y1) * scale
    w = max(w, 1.0); h = max(h, 1.0)

    # Adjust to the target aspect ratio
    cur_ratio = w / h
    if cur_ratio < target_ratio:
        # Too narrow: expand w
        w = h * target_ratio
    else:
        # Too wide: expand h
        h = w / target_ratio

    x1n = cx - 0.5 * w
    x2n = cx + 0.5 * w
    y1n = cy - 0.5 * h
    y2n = cy + 0.5 * h
    return x1n, y1n, x2n, y2n

def _clamp_xyxy(x1, y1, x2, y2, W, H):
    x1 = max(0.0, min(x1, W - 1.0))
    x2 = max(0.0, min(x2, W - 1.0))
    y1 = max(0.0, min(y1, H - 1.0))
    y2 = max(0.0, min(y2, H - 1.0))
    # Fallback to ensure x2 > x1 and y2 > y1
    if x2 <= x1: x2 = min(W - 1.0, x1 + 1.0)
    if y2 <= y1: y2 = min(H - 1.0, y1 + 1.0)
    return x1, y1, x2, y2

from pathlib import Path
import random
import numpy as np
import torch

def get_face_ref_image_fullseq(
    video_reader,
    video_path: str | Path,
    cur_resolution,                 # [H,W]
    drop_prob: float = 0.2,
    score_thr: float = 0.3,
    min_face_pts: int = 20,         # "most of the face": visible face-point count threshold, tunable from 15 to 30
    topk: int = 10,                 # randomly select from topk to ensure randomness
    enlarge_scale: float = 1.25,    # bbox enlargement scale, currently intended to use 1.25
    # --- Newly added: tighter rules for a more "frontal face" preference, without too much complexity ---
    require_frontal: bool = True,   # If there is no frontal face -> directly drop
    min_eye_pts: int = 3,           # At least this many visible keypoints for each eye (36-41/42-47), >=3
    max_roll_deg: float = 12.0,     # Upper bound for eye-line tilt angle, i.e., roll
    max_nose_offset_ratio: float = 0.16,  # Upper bound for nose-tip deviation from the midpoint between two eyes, relative to eye distance
    max_lr_vis_ratio: float = 1.8,  # The visible point counts on the left/right face should not be extremely imbalanced; profile faces often are
    min_bbox_size_px: int = 28,     # Ignore faces whose bbox is too small
):
    """
    Returns:
      face_ref: torch.Tensor [1,3,H,W]  (0~255 float32)
      face_idx: int or None

    Rules:
      - Directly drop with probability drop_prob -> all zeros
      - If no qualified "frontal face" frame is found -> all zeros when require_frontal=True
      - Randomly select one frame from the quality topk list, only within frontal-face candidates
    """
    Ht, Wt = int(cur_resolution[0]), int(cur_resolution[1])
    device = torch.device("cpu")

    # 1) drop
    if random.random() < drop_prob:
        return torch.zeros((1, 3, Ht, Wt), dtype=torch.float32, device=device), None

    video_path = Path(video_path)
    dwpose_path = video_path.parent / "dwpose.npz"
    if not dwpose_path.exists():
        return torch.zeros((1, 3, Ht, Wt), dtype=torch.float32, device=device), None

    z = np.load(dwpose_path, allow_pickle=False)
    cand = z["candidate"]          # [T,K,2] normalized, invisible=-1
    sub  = z["subset"]             # [T,K] scores
    valid = z.get("valid", None)   # [T]
    img_wh = z.get("img_wh", None) # [2] (W,H)

    T = cand.shape[0]
    if img_wh is not None:
        W0, H0 = int(img_wh[0]), int(img_wh[1])
    else:
        fr0 = video_reader[0]
        if isinstance(fr0, torch.Tensor):
            fr0 = fr0.cpu().numpy()
        H0, W0 = fr0.shape[0], fr0.shape[1]

    # Face index range: according to faces = candidate[:,24:92] in your DWposeDetector
    f0, f1 = 24, 92  # 68 pts

    # Internal indices of the 68 face points, typically following the OpenPose face convention
    NOSE_TIP = 30
    L_EYE_IDXS = list(range(36, 42))  # 36-41
    R_EYE_IDXS = list(range(42, 48))  # 42-47

    def _pt_visible(xy, sc):
        return (xy[0] >= 0) and (xy[1] >= 0) and (sc > score_thr)

    def _mean_xy(face_xy, face_sc, idxs):
        pts = []
        for k in idxs:
            if _pt_visible(face_xy[k], face_sc[k]):
                pts.append(face_xy[k])
        if len(pts) == 0:
            return None, 0
        pts = np.stack(pts, axis=0)
        return pts.mean(axis=0), len(pts)

    target_ratio = float(Wt) / float(Ht)

    # 2) Scan once to find "frontal-face candidates" and score them
    scored = []
    for t in range(T):
        if valid is not None and int(valid[t]) == 0:
            continue

        face_xy = cand[t, f0:f1]        # [68,2] normalized
        face_sc = sub[t, f0:f1]         # [68]

        vis = (face_xy[:, 0] >= 0) & (face_xy[:, 1] >= 0) & (face_sc > score_thr)
        nvis = int(vis.sum())
        if nvis < min_face_pts:
            continue

        px = face_xy[vis, 0] * W0
        py = face_xy[vis, 1] * H0
        x1, y1, x2, y2 = _xyxy_from_points(px, py)

        bw = x2 - x1
        bh = y2 - y1
        if bw < min_bbox_size_px or bh < min_bbox_size_px:
            continue

        # ---- Newly added rule 1: both eyes and nose tip should be reliably visible, strongly filtering profile/occluded faces ----
        nose_ok = _pt_visible(face_xy[NOSE_TIP], face_sc[NOSE_TIP])
        le_xy, le_cnt = _mean_xy(face_xy, face_sc, L_EYE_IDXS)
        re_xy, re_cnt = _mean_xy(face_xy, face_sc, R_EYE_IDXS)

        if require_frontal:
            if (not nose_ok) or (le_cnt < min_eye_pts) or (re_cnt < min_eye_pts):
                continue
        else:
            # If frontal face is not strictly required, missing points are allowed but will receive a lower score
            pass

        # If geometry cannot be computed, e.g., not requiring frontal face but eyes are missing, use a degraded score
        frontal_geom_ok = nose_ok and (le_xy is not None) and (re_xy is not None)

        # ---- Newly added rule 2: the eye-line roll should not be too large ----
        # ---- Newly added rule 3: the nose tip should be close to the midpoint between two eyes, relative to eye distance ----
        # ---- Newly added rule 4: visible point counts on the left/right face should not be extremely imbalanced; profile faces often are ----
        roll_ok = True
        nose_center_ok = True
        lr_balance_ok = True

        nose_offset_ratio = 1.0
        roll_deg = 90.0

        if frontal_geom_ok:
            le = le_xy * np.array([W0, H0], dtype=np.float32)
            re = re_xy * np.array([W0, H0], dtype=np.float32)
            nose = face_xy[NOSE_TIP] * np.array([W0, H0], dtype=np.float32)

            eye_dx = float(re[0] - le[0])
            eye_dy = float(re[1] - le[1])
            eye_dist = float(np.sqrt(eye_dx * eye_dx + eye_dy * eye_dy) + 1e-6)

            roll_deg = abs(np.degrees(np.arctan2(eye_dy, eye_dx)))
            roll_ok = (roll_deg <= max_roll_deg)

            eye_cx = 0.5 * (float(le[0]) + float(re[0]))
            nose_offset_ratio = abs(float(nose[0]) - eye_cx) / eye_dist
            nose_center_ok = (nose_offset_ratio <= max_nose_offset_ratio)

            # Balance left/right face point counts using the bbox center as the boundary
            cx = 0.5 * (x1 + x2)
            vx = face_xy[vis, 0] * W0
            left_cnt = int((vx < cx).sum())
            right_cnt = int((vx >= cx).sum())
            mn = max(1, min(left_cnt, right_cnt))
            mx = max(left_cnt, right_cnt)
            lr_ratio = mx / float(mn)
            lr_balance_ok = (lr_ratio <= max_lr_vis_ratio)

            if require_frontal and (not roll_ok or not nose_center_ok or not lr_balance_ok):
                continue
        else:
            if require_frontal:
                continue

        # Score is only used for ranking inside the frontal-face candidate set
        face_ratio = nvis / float(f1 - f0)  # 0~1
        size_term = np.sqrt((bw * bh) / (W0 * H0) + 1e-6)  # around 0~1

        # Higher score for more frontal faces
        frontal_term = 0.0
        if frontal_geom_ok:
            frontal_term += (1.0 - min(1.0, nose_offset_ratio / max_nose_offset_ratio)) * 2.0
            frontal_term += (1.0 - min(1.0, roll_deg / max_roll_deg)) * 1.0
            frontal_term += (1.0 if lr_balance_ok else 0.0) * 0.5

        score = 0.0
        score += 2.5 * face_ratio
        score += 2.0 * size_term
        score += frontal_term
        # Add a small hard bonus for visible nose tip / eyes for better stability
        score += 0.5 if nose_ok else 0.0
        score += 0.5 if (le_cnt >= min_eye_pts and re_cnt >= min_eye_pts) else 0.0

        scored.append((score, t, (x1, y1, x2, y2)))

    if len(scored) == 0:
        return torch.zeros((1, 3, Ht, Wt), dtype=torch.float32, device=device), None

    scored.sort(key=lambda x: x[0], reverse=True)
    pick_pool = scored[: min(topk, len(scored))]

    # 3) Randomly select from topk, only within frontal-face candidates, to preserve randomness
    score, t_pick, (x1, y1, x2, y2) = random.choice(pick_pool)

    # 4) Enlarge bbox + adjust to target aspect ratio + clamp
    x1, y1, x2, y2 = _expand_to_aspect_xyxy(x1, y1, x2, y2, target_ratio, scale=enlarge_scale)
    x1, y1, x2, y2 = _clamp_xyxy(x1, y1, x2, y2, W0, H0)

    # 5) Map dwpose frame index to video_reader frame index
    V = len(video_reader)
    if V == T:
        vid_idx = int(t_pick)
    else:
        vid_idx = int(round(t_pick * (V - 1) / max(1, (T - 1))))
        vid_idx = max(0, min(vid_idx, V - 1))

    fr = video_reader[vid_idx]
    if isinstance(fr, torch.Tensor):
        fr = fr.cpu().numpy()  # [H,W,3] uint8 RGB; usually RGB on your side
    fr_t = torch.from_numpy(fr).permute(2, 0, 1).float().unsqueeze(0)  # [1,3,H,W]

    # 6) Crop + keep ratio + pad to cur_resolution by reusing the existing function
    # If this function is already in oneshot_data_utils.py, directly calling crop_blackpad_by_bboxes_xyxy is enough
    try:
        from .oneshot_data_utils import crop_blackpad_by_bboxes_xyxy
    except Exception:
        crop_blackpad_by_bboxes_xyxy = globals().get("crop_blackpad_by_bboxes_xyxy", None)
        if crop_blackpad_by_bboxes_xyxy is None:
            raise RuntimeError("crop_blackpad_by_bboxes_xyxy not found. Please import/define it in this module.")

    bbox = torch.tensor([[x1, y1, x2, y2]], dtype=torch.float32)
    face_ref = crop_blackpad_by_bboxes_xyxy(fr_t, bbox, target_size=cur_resolution)  # [1,3,Ht,Wt]

    return face_ref, vid_idx


from PIL import Image, ImageOps
def _find_img(profile_dir: str, stem: str):
    # Allow ref1.png / ref1.jpg / ref1.jpeg / ref1.webp
    exts = [".png", ".jpg", ".jpeg", ".webp"]
    for ext in exts:
        p = os.path.join(profile_dir, stem + ext)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"[specify_ID_profile_path] cannot find {stem} with extensions {exts} under: {profile_dir}")

import torch
import torch.nn.functional as F

def resize_keep_ratio_pad_4d(x: torch.Tensor, target_hw, pad_value=0.0, mode="bilinear"):
    """
    x: [N,C,H,W], float/uint8 are both acceptable; internally converted to float for resizing.
    return: [N,C,target_H,target_W], preserving the full image by scaling and padding extra regions with black.
    """
    assert x.ndim == 4, x.shape
    N, C, H, W = x.shape
    th, tw = int(target_hw[0]), int(target_hw[1])

    xf = x.float()

    # Use the minimum scale that allows the image to fit completely inside the target canvas
    s = min(th / H, tw / W)
    nh = max(1, int(round(H * s)))
    nw = max(1, int(round(W * s)))

    # resize
    xr = F.interpolate(xf, size=(nh, nw), mode=mode, align_corners=False if mode in ("bilinear","bicubic") else None, antialias=True)

    # Pad to target size, centered
    out = xf.new_full((N, C, th, tw), float(pad_value))
    y0 = (th - nh) // 2
    x0 = (tw - nw) // 2
    out[:, :, y0:y0+nh, x0:x0+nw] = xr
    return out.to(dtype=x.dtype if x.dtype in (torch.uint8,) else xf.dtype)

from PIL import Image
import os
import torchvision.transforms.functional as tvF

def _load_profile_as_tensor_letterbox(path, target_size, device="cpu", dtype=torch.float32):
    th, tw = int(target_size[0]), int(target_size[1])
    # If missing, return a black image with the correct size: [1,3,H,W]
    if (path is None) or (not isinstance(path, (str, os.PathLike))) or (not os.path.exists(str(path))):
        return torch.zeros((1, 3, th, tw), device=device, dtype=dtype)

    img = Image.open(str(path)).convert("RGB")
    t = tvF.pil_to_tensor(img).float().unsqueeze(0)  # [1,3,H,W] 0~255
    t = resize_keep_ratio_pad_4d(t, target_size, pad_value=0.0, mode="bilinear")
    return t.to(device=device, dtype=dtype)