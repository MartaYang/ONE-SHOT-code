# myutil_savergbd_da3.py
import os
import math
import inspect
import numpy as np
import cv2
import torch

import numpy as np
import os

def _pack_human_mask_packbits(msks_to_vis, thr=0.5, bitorder="little"):
    """
    msks_to_vis: list of np arrays, each [1,H,W] float32 (human prob) or uint8/float
    return:
      mask_packbits: uint8 [N,H,ceil(W/8)]
      mask_shape: int32 [3] = (N,H,W)
    """
    m = np.stack([(m[0] > thr) for m in msks_to_vis], axis=0).astype(np.uint8)  # [N,H,W] in {0,1}
    packed = np.packbits(m, axis=-1, bitorder=bitorder)  # [N,H,ceil(W/8)]
    shape = np.array(m.shape, dtype=np.int32)
    return packed, shape

def save_camera_and_humanmask_npz(
    save_path: str,
    intrinsics: np.ndarray,          # [N,3,3] float32
    extrinsics_w2c: np.ndarray,      # [N,3,4] float32
    msks_to_vis,                     # list of [1,H,W]
    human_msk_thr: float = 0.5,
    bitorder: str = "little",
):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    mask_packbits, mask_shape = _pack_human_mask_packbits(msks_to_vis, thr=human_msk_thr, bitorder=bitorder)

    np.savez_compressed(
        save_path,
        intrinsics=intrinsics.astype(np.float32),
        extrinsics_w2c=extrinsics_w2c.astype(np.float32),
        mask_packbits=mask_packbits.astype(np.uint8),
        mask_shape=mask_shape,
        human_msk_thr=np.float32(human_msk_thr),
        bitorder=np.array(bitorder),
    )

def _w2c_to_c2w(w2c_3x4: np.ndarray) -> np.ndarray:
    """w2c: [3,4]  ->  c2w: [4,4]"""
    R = w2c_3x4[:, :3]
    t = w2c_3x4[:, 3:4]
    Rcw = R.T
    tcw = -Rcw @ t
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :3] = Rcw.astype(np.float32)
    c2w[:3, 3] = tcw[:, 0].astype(np.float32)
    return c2w

def _depth_to_cam_points(depth: np.ndarray, K: np.ndarray) -> np.ndarray:
    """
    depth: [H,W] (meters)
    K: [3,3]
    return: cam_xyz [H,W,3]
    """
    H, W = depth.shape
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])

    u = np.arange(W, dtype=np.float32)
    v = np.arange(H, dtype=np.float32)
    uu, vv = np.meshgrid(u, v)

    z = depth.astype(np.float32)
    x = (uu - cx) / fx * z
    y = (vv - cy) / fy * z
    return np.stack([x, y, z], axis=-1)  # [H,W,3]

def _apply_c2w(cam_xyz: np.ndarray, c2w: np.ndarray) -> np.ndarray:
    """cam_xyz [H,W,3] -> world_xyz [H,W,3]"""
    H, W, _ = cam_xyz.shape
    R = c2w[:3, :3]
    t = c2w[:3, 3]
    p = cam_xyz.reshape(-1, 3) @ R.T + t[None, :]
    return p.reshape(H, W, 3).astype(np.float32)

def _resize_mask_nearest(mask: np.ndarray, H: int, W: int) -> np.ndarray:
    """mask: [H0,W0] or [1,H0,W0] -> [1,H,W] float32"""
    if mask.ndim == 3:
        mask2 = mask[0]
    else:
        mask2 = mask
    mask_r = cv2.resize(mask2.astype(np.float32), (W, H), interpolation=cv2.INTER_NEAREST)
    return mask_r[None, ...].astype(np.float32)

def _choose_bg_conf_thr_quantile(conf_list, mask_list=None, msk_thr=0.5, q=0.10):
    """
    conf_list: list of [1,H,W] float
    mask_list: list of [1,H,W] float (human prob). If None -> use all pixels.
    """
    vals = []
    for i, cf in enumerate(conf_list):
        c = cf[0]
        if mask_list is not None:
            bg = (mask_list[i][0] <= msk_thr)
            if bg.any():
                vals.append(c[bg].reshape(-1))
        else:
            vals.append(c.reshape(-1))
    if not vals:
        return None
    v = np.concatenate(vals, axis=0)
    return float(np.quantile(v, q))

@torch.no_grad()
def render_bg_rgbd_with_da3(
    img_paths,
    out_dir,
    fps_in,
    # ---- masking：建议直接传 human3r 的 msks_to_vis（同一批帧）----
    human_msks_to_vis=None,  # list of np [1,Hh,Ww] 或 None
    human_msk_thr=0.5,

    # ---- DA3 ----
    da3_model_id=None,  # 由调用方传入（preprocess_video.py 走 constants.DA3_CKPT_DIR）
    device="cuda",
    size=512,
    export_format="npz",   # 关键：确保能拿到 extrinsics/intrinsics/conf（见 issue 讨论）
    # ---- quality knobs ----
    bg_conf_q=0.10,        # 取背景conf的分位数做阈值（你说“10%”就是0.10）
    bg_frame_stride=1,     # 额外再稀疏取帧做“合成全局云”（不是推理 stride）
    voxel_size=0.01,
    near_far=(0.3, 12.0),
    mask_morph=4,
    quiet=True,
    # ---- 原视频相关 ----
    gt_input_seq=None,
    video_fps=None,
    subsample=1,
    tmp_dir="/tmp/da3_tmp",
):
    """
    输出到 out_dir：
      - bg_world_cloud.ply（由你现有 render_rgbd_ply 写）
      - rgb_fill.mp4 / depth.mp4（由你现有 render_rgbd_ply 写）
    """
    os.makedirs(out_dir, exist_ok=True)

    # 1) DA3 推理
    assert da3_model_id is not None, "da3_model_id must be provided (see constants.DA3_CKPT_DIR)"
    from depth_anything_3.api import DepthAnything3
    model = DepthAnything3.from_pretrained(da3_model_id).to(device)
    model.eval()
    # model = model.half()

    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
        pred = model.inference(
            img_paths,
            export_dir=tmp_dir,          # 让它也能落盘（可选）
            export_format=export_format  # glb/npz/ply/...；建议 npz/ply
        )

    depth = pred.depth  # [N,H,W] float32
    conf  = pred.conf   # [N,H,W] float32
    ext   = pred.extrinsics  # [N,3,4] w2c
    intr  = pred.intrinsics  # [N,3,3]

    assert depth is not None and conf is not None and ext is not None and intr is not None, \
        "DA3 输出缺失（depth/conf/extrinsics/intrinsics）。请检查 export_format / 模型版本。"

    N, H, W = depth.shape

    # 2) 读取颜色并对齐到 [H,W]
    colors_to_vis = []
    for p in img_paths:
        im = cv2.imread(p, cv2.IMREAD_COLOR)
        assert im is not None, f"cv2.imread failed: {p}"
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im = cv2.resize(im, (W, H), interpolation=cv2.INTER_LINEAR)

        # 关键：转 float32, 0~1（对齐 human3r 的 render_rgbd_ply 预期）
        im = (im.astype(np.float32) / 255.0)
        im = np.ascontiguousarray(im)   # 顺便减少 swscaler "not aligned" 警告
        colors_to_vis.append(im[None, ...])  # [1,H,W,3] float32

    # 3) 生成 pts3ds_to_vis（world pointmaps），以及 cam_dict（c2w）
    pts3ds_to_vis = []
    conf_to_vis = []
    R_list, t_list, f_list, pp_list = [], [], [], []

    for i in range(N):
        K = intr[i].astype(np.float32)
        w2c = ext[i].astype(np.float32)

        c2w = _w2c_to_c2w(w2c)
        cam_xyz = _depth_to_cam_points(depth[i], K)
        world_xyz = _apply_c2w(cam_xyz, c2w)

        pts3ds_to_vis.append(world_xyz[None, ...])      # [1,H,W,3]
        conf_to_vis.append(conf[i][None, ...].astype(np.float32))  # [1,H,W]

        # cam_dict 用你 human3r 的格式（R_c2w, t_c2w）
        R_list.append(c2w[:3, :3])
        t_list.append(c2w[:3, 3])

        # render_rgbd_ply 里通常用 focal/pp（按人3r逻辑：单标量 focal）
        f = 0.5 * (float(K[0, 0]) + float(K[1, 1]))
        f_list.append(f)
        pp_list.append([float(K[0, 2]), float(K[1, 2])])

    cam_dict = {
        "focal": np.array(f_list, dtype=np.float32),
        "pp":    np.array(pp_list, dtype=np.float32),
        "R":     np.stack(R_list, axis=0).astype(np.float32),  # [N,3,3]
        "t":     np.stack(t_list, axis=0).astype(np.float32),  # [N,3]
        "K":     intr.astype(np.float32)   # [N,3,3],
    }

    # 4) mask：把 human3r mask resize 到 [H,W]，用于“无人的背景”
    if human_msks_to_vis is None:
        msks_to_vis = [np.zeros((1, H, W), np.float32) for _ in range(N)]
    else:
        assert len(human_msks_to_vis) == N, "human_msks_to_vis 帧数要和 img_paths 对齐"
        msks_to_vis = [_resize_mask_nearest(m, H, W) for m in human_msks_to_vis]

    # 5) 自动选 bg_conf_thr（对背景像素做分位数）
    bg_conf_thr = _choose_bg_conf_thr_quantile(
        conf_to_vis, mask_list=msks_to_vis, msk_thr=human_msk_thr, q=bg_conf_q
    )

    # 6) 复用你现有的 render_rgbd_ply 输出（保持格式一致）
    from myutil_savergbd import render_rgbd_ply
    kwargs = dict(
        pts3ds_to_vis=pts3ds_to_vis,
        colors_to_vis=colors_to_vis,
        msks_to_vis=msks_to_vis,
        cam_dict=cam_dict,
        output_dir=out_dir,
        image_size=(H, W),
        fps_out=max(1, int(round(fps_in))),
        msk_threshold=float(human_msk_thr),
        mask_morph=int(mask_morph),
        voxel_size=float(voxel_size),
        near_far=near_far,
        device=device,
        quiet=quiet,

        # 下面三个是你“改造版 render_rgbd_ply”才有的参数；老版本会自动忽略
        conf_to_vis=conf_to_vis,
        bg_conf_thr=bg_conf_thr,
        bg_frame_stride=int(bg_frame_stride),

        # 原视频相关
        gt_input_seq=gt_input_seq,
        video_fps=video_fps,
        subsample=subsample
    )

    sig = inspect.signature(render_rgbd_ply)
    kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
    render_rgbd_ply(**kwargs)

    save_camera_and_humanmask_npz(
        save_path=os.path.join(out_dir, "camera_and_humanmask.npz"),
        intrinsics=intr,                 # [N,3,3]
        extrinsics_w2c=ext,              # [N,3,4]
        msks_to_vis=msks_to_vis,         # list of [1,H,W] aligned to DA3
        human_msk_thr=float(human_msk_thr),
    )

    return {
        "H": H, "W": W, "N": N,
        "bg_conf_thr": bg_conf_thr,
        "out_dir": out_dir
    }