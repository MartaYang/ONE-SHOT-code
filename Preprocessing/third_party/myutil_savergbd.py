import numpy as np
import cv2
import imageio.v2 as iio
import subprocess, shlex, os, shutil
import torch

def write_ply_xyzrgb(path, xyz, rgb):
    """
    Saves a point cloud (xyz, rgb) to a binary PLY file.

    Args:
        path (str): The path to save the PLY file to.
        xyz (np.ndarray): The 3D coordinates of the points (N, 3).
        rgb (np.ndarray): The RGB colors of the points (N, 3).
    """
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Ensure data is in the correct format
    xyz = np.asarray(xyz, dtype=np.float32)
    rgb = np.asarray(rgb, dtype=np.uint8)

    # # 过滤 NaN/Inf、颜色越界
    # valid = np.isfinite(xyz).all(axis=1)
    # xyz = xyz[valid]
    # if rgb.dtype != np.uint8:
    #     rgb = np.clip(rgb[valid] * 255.0, 0, 255).astype(np.uint8)
    # else:
    #     rgb = rgb[valid]

    # Write the PLY header
    header = (
        f"ply\n"
        f"format binary_little_endian 1.0\n"
        f"element vertex {len(xyz)}\n"
        f"property float x\n"
        f"property float y\n"
        f"property float z\n"
        f"property uchar red\n"
        f"property uchar green\n"
        f"property uchar blue\n"
        f"end_header\n"
    )

    # Create a structured array with fields for x, y, z, red, green, blue
    # The 'f4' corresponds to float32, and 'u1' to uint8
    vertex_data = np.empty(len(xyz), dtype=[
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')
    ])

    vertex_data['x'] = xyz[:, 0]
    vertex_data['y'] = xyz[:, 1]
    vertex_data['z'] = xyz[:, 2]
    vertex_data['red'] = rgb[:, 0]
    vertex_data['green'] = rgb[:, 1]
    vertex_data['blue'] = rgb[:, 2]

    # Write the header and the data to the file
    with open(path, 'wb') as f:
        f.write(header.encode('utf-8'))
        f.write(vertex_data.tobytes())


def _voxel_keep_best(P, C, W, voxel_size: float):
    if voxel_size is None or voxel_size <= 0 or len(P) == 0:
        return P, C, W
    q = np.floor(P / voxel_size).astype(np.int64)  # [N,3]
    order = np.lexsort((q[:, 2], q[:, 1], q[:, 0]))
    q, P, C, W = q[order], P[order], C[order], W[order]

    same = np.all(q[1:] == q[:-1], axis=1)
    starts = np.r_[0, np.where(~same)[0] + 1]
    ends = np.r_[starts[1:], len(q)]

    sel = []
    for s, e in zip(starts, ends):
        j = s + int(np.argmax(W[s:e]))
        sel.append(j)
    sel = np.asarray(sel, dtype=np.int64)
    return P[sel], C[sel], W[sel]

def build_global_bg_cloud(
    pts_world_list, colors_list, masks_list,
    confs_list=None,
    conf_thr: float = None,
    cam_centers_list=None,
    near_far=None,
    voxel_size=0.02,
    frame_stride: int = 1,
    mask_morph: int = 0,      # ✅ 新增：人像mask膨胀像素半径（建议 3~6）
    mask_thr: float = 0.5,    # ✅ 新增：mask阈值
    verbose=True
):
    import numpy as np
    P_all, C_all, W_all = [], [], []

    # 仅在需要时import cv2，避免没装opencv时直接崩（你项目里应该已经有了）
    cv2 = None
    if mask_morph and mask_morph > 0:
        import cv2 as _cv2
        cv2 = _cv2
        k = 2 * int(mask_morph) + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

    for i, (P, C, M) in enumerate(zip(pts_world_list, colors_list, masks_list)):
        if frame_stride > 1 and (i % frame_stride != 0):
            continue

        P = P[0]  # [H,W,3]
        C = C[0]  # [H,W,3]
        M = M[0]  # [H,W]

        valid = np.isfinite(P).all(axis=-1)

        # --- ✅ mask morph（dilate）恢复 ---
        human = (M > mask_thr).astype(np.uint8)   # 1=human
        if cv2 is not None:
            human = cv2.dilate(human, kernel, iterations=1)
        bg = (human == 0)

        keep = valid & bg

        if confs_list is not None and conf_thr is not None:
            conf = confs_list[i][0]  # [H,W]
            keep = keep & (conf >= conf_thr)

        if cam_centers_list is not None and near_far is not None:
            c = cam_centers_list[i].reshape(1, 1, 3)
            dist = np.linalg.norm(P - c, axis=-1)
            keep = keep & (dist >= float(near_far[0])) & (dist <= float(near_far[1]))

        if keep.sum() == 0:
            continue

        Pk = P[keep].reshape(-1, 3)
        Ck = C[keep].reshape(-1, 3)

        if confs_list is not None:
            conf = confs_list[i][0]
            Wk = conf[keep].reshape(-1).astype(np.float32)
        else:
            Wk = np.ones((Pk.shape[0],), np.float32)

        P_all.append(Pk); C_all.append(Ck); W_all.append(Wk)

    if len(P_all) == 0:
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.float32)

    P_all = np.concatenate(P_all, axis=0)
    C_all = np.concatenate(C_all, axis=0)
    W_all = np.concatenate(W_all, axis=0)

    P_ds, C_ds, _ = _voxel_keep_best(P_all, C_all, W_all, voxel_size=float(voxel_size))

    if verbose:
        print(f"[build_global_bg_cloud] raw={len(P_all)} -> voxel={len(P_ds)} "
              f"(voxel={voxel_size}, conf_thr={conf_thr}, mask_morph={mask_morph}, frame_stride={frame_stride})")

    return P_ds, C_ds


def build_global_bg_cloud_old_noconf(pts_world_list, colors_list, masks_list,
                          msk_threshold=0.5, morph=0, voxel_size=None):
    """
    构建“全局背景点云”（世界坐标）。
    - pts_world_list: list[ (H,W,3) world xyz ]，来自每帧的 pts3ds_other
    - colors_list:    list[ (H,W,3) uint8 或 0..1 float ]
    - masks_list:     list[ (1,H,W) ]，人体前景概率
    - msk_threshold:  mask >= 阈值 判为“人像”
    - morph:          对人像mask做膨胀的核半径（像素），去掉边缘漏标（建议 3~7）
    - voxel_size:     体素下采样（米），如 0.02；None 表示不下采样
    返回: (P_all, C_all) -> (N,3) world xyz, (N,3) uint8 rgb
    """
    pts_all, col_all = [], []
    for P, C, M in zip(pts_world_list, colors_list, masks_list):
        P = np.asarray(P)                             # (H,W,3)
        C = np.asarray(C)
        if C.dtype != np.uint8:
            C = np.clip(C * 255.0, 0, 255).astype(np.uint8)
        M = np.asarray(M)[0]                          # (H,W) 人体概率

        human = (M >= msk_threshold).astype(np.uint8)
        if morph > 0:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph*2+1, morph*2+1))
            human = cv2.dilate(human, k)

        bg = (human == 0)
        valid = np.isfinite(P).all(axis=-1)
        keep = bg & valid

        if keep.any():
            pts_all.append(P[keep])
            col_all.append(C[keep])

    if len(pts_all) == 0:
        return np.empty((0,3), np.float32), np.empty((0,3), np.uint8)

    P_all = np.concatenate(pts_all, axis=0).astype(np.float32)
    C_all = np.concatenate(col_all, axis=0).astype(np.uint8)

    if voxel_size is not None and voxel_size > 0:
        q = np.floor(P_all / voxel_size).astype(np.int64)  # 体素量化
        q_view = q.view([('x', q.dtype), ('y', q.dtype), ('z', q.dtype)])
        _, idx = np.unique(q_view, return_index=True)
        P_all = P_all[idx]
        C_all = C_all[idx]

    return P_all, C_all

def fill_frame_from_global(P_world, C_world, K, cam_c2w, H, W,
                           depth_orig=None, rgb_orig=None, human_mask=None, fill_only_human=False):
    """
    直接返回“整图由全局背景点云重投影”的 RGB 与深度（相机系 z）。
    忽略原始帧内容与 mask，完全由全局背景决定。
    """
    import numpy as np
    w2c = np.linalg.inv(cam_c2w).astype(np.float32)
    R = w2c[:3,:3]; t = w2c[:3,3]

    Pc = P_world @ R.T + t[None,:]         # (N,3)
    z  = Pc[:,2]
    valid = z > 1e-6
    Pc = Pc[valid]; z = z[valid]
    uv = (Pc[:,:2] / z[:,None]) @ K[:2,:2].T + K[:2,2]
    u = np.round(uv[:,0]).astype(np.int32)
    v = np.round(uv[:,1]).astype(np.int32)
    inside = (u>=0)&(u<W)&(v>=0)&(v<H)
    u = u[inside]; v = v[inside]; z = z[inside]
    cols = C_world[valid][inside]

    # z-buffer
    zbuf = np.full((H,W), np.inf, dtype=np.float32)
    np.minimum.at(zbuf, (v,u), z)
    sel = (z == zbuf[v,u])
    u2, v2 = u[sel], v[sel]
    z2, cols2 = z[sel], cols[sel]

    depth_full = np.zeros((H,W), dtype=np.float32)
    rgb_full   = np.zeros((H,W,3), dtype=np.uint8)

    # 仅将最前景的全局背景点写回
    depth_full[v2, u2] = z2
    rgb_full[v2, u2]   = cols2

    return rgb_full, depth_full


def fill_frame_from_global_torch(P_world, C_world, K, cam_c2w, H, W, device='cuda'):
    """
    用 PyTorch 在 GPU 上做整图重投影 + z-buffer，返回 (rgb_full uint8, depth_full float32[meters])
    输入:
      P_world: (N,3) float32 numpy 或 torch（米，世界坐标）
      C_world: (N,3) uint8 颜色（或 0..1 float）
      K:       (3,3) float32 numpy（内参）
      cam_c2w: (4,4) float32 numpy（相机->世界），会在内部取逆
      H, W:    输出分辨率
      device:  'cuda' 或 'cpu'（推荐 'cuda'）
    """
    import numpy as np, torch

    # to torch
    if not torch.is_tensor(P_world):
        P = torch.as_tensor(P_world, dtype=torch.float32, device=device)
    else:
        P = P_world.to(device, dtype=torch.float32)
    if C_world.dtype != np.uint8:
        C = torch.clamp(torch.as_tensor(C_world, dtype=torch.float32, device=device)*255.0, 0, 255).to(torch.uint8)
    else:
        C = torch.as_tensor(C_world, dtype=torch.uint8, device=device)
    Kt = torch.as_tensor(K, dtype=torch.float32, device=device)
    c2w = torch.as_tensor(cam_c2w, dtype=torch.float32, device=device)
    w2c = torch.linalg.inv(c2w)

    # world -> cam
    R = w2c[:3,:3]            # (3,3)
    t = w2c[:3, 3]            # (3,)
    Pc = (P @ R.T) + t        # (N,3)
    z  = Pc[:,2]
    valid = z > 1e-6
    if valid.sum() == 0:
        rgb_full = torch.zeros((H,W,3), dtype=torch.uint8, device=device)
        depth_full = torch.zeros((H,W), dtype=torch.float32, device=device)
        return rgb_full.cpu().numpy(), depth_full.cpu().numpy()

    Pc = Pc[valid]; z = z[valid]
    u = (Kt[0,0]*Pc[:,0]/z + Kt[0,2]).round().to(torch.int64)
    v = (Kt[1,1]*Pc[:,1]/z + Kt[1,2]).round().to(torch.int64)
    inside = (u>=0)&(u<W)&(v>=0)&(v<H)
    if inside.sum() == 0:
        rgb_full = torch.zeros((H,W,3), dtype=torch.uint8, device=device)
        depth_full = torch.zeros((H,W), dtype=torch.float32, device=device)
        return rgb_full.cpu().numpy(), depth_full.cpu().numpy()

    u = u[inside]; v = v[inside]
    z = z[inside]
    cols = C[valid][inside]

    # flatten pixel index
    idx = v*W + u                           # (M,)
    numpix = H*W

    # # z-buffer (min z per pixel)
    # zbuf = torch.full((numpix,), float('inf'), dtype=torch.float32, device=device)
    # zbuf.scatter_reduce_(0, idx, z, reduce='amin', include_self=True)

    # # 选择 z 等于最小值的那一批（容忍微小数值误差）
    # z_sel = torch.isfinite(zbuf)[idx] & (z <= (zbuf[idx] + 1e-6))

    # # 输出
    # depth = torch.zeros((numpix,), dtype=torch.float32, device=device)
    # rgb   = torch.zeros((numpix,3), dtype=torch.uint8,   device=device)

    # write_idx = idx[z_sel]
    # depth[write_idx] = z[z_sel]
    # rgb[write_idx]   = cols[z_sel]

    # depth = depth.view(H,W)
    # rgb   = rgb.view(H,W,3)

    # 为了解决黑点闪烁修改
    # --- deterministic per-pixel nearest selection ---
    depth = torch.zeros((numpix,), dtype=torch.float32, device=device)
    rgb   = torch.zeros((numpix, 3), dtype=torch.uint8, device=device)

    # 量化 z：0.1mm（你 near_far 到 12m，完全够用；也能减少浮点抖动）
    zq = torch.clamp((z * 10000.0).to(torch.int64), 0, (1 << 20) - 1)  # 0.1mm, fits 20 bits for z<~104m
    key = (idx.to(torch.int64) << 20) + zq

    order = torch.argsort(key)   # key: idx primary, zq secondary
    idx_s = idx[order]
    z_s   = z[order]
    c_s   = cols[order]

    # 每个像素只保留第一个（最近的）
    is_first = torch.ones_like(idx_s, dtype=torch.bool)
    is_first[1:] = idx_s[1:] != idx_s[:-1]

    idx_k = idx_s[is_first]
    z_k   = z_s[is_first]
    c_k   = c_s[is_first]

    depth[idx_k] = z_k
    rgb[idx_k]   = c_k

    depth = depth.view(H, W)
    rgb   = rgb.view(H, W, 3)

    # # ================== 新增：消除空洞闪烁 ==================
    # # 将 (H, W, C) -> (1, C, H, W) 以便使用 torch.nn.functional
    # # 3x3 的 max_pool 相当于把每个点“膨胀”一圈，填补单像素空洞
    # import torch.nn.functional as F
    
    # # 1. 处理 Depth (取局部最小值，注意深度是越小越近，但 max_pool 是取最大)
    # # 这里的处理稍微tricky：因为背景是0，前景是值。
    # # 如果你的深度背景是0，我们想保留前景（非0值）。
    # # 如果深度背景是inf，我们想取min。
    
    # # 更简单通用的方法：对 RGB 和 Depth 结果图进行形态学膨胀 (Dilation)
    # # 我们用 MaxPool2d 模拟 Dilation (针对 RGB)
    # # 必须先把 RGB 里的 0 (黑色背景) 视为无效值，否则黑点会吃掉颜色
    # # 但如果仅仅是填补细微空隙，直接对结果图做 filter 即可。
    
    # # ----------------------------------------------------
    # # 极简方案：直接用 CPU 的 OpenCV 做个后处理（最稳健）
    # # ----------------------------------------------------
    # rgb_np = rgb.cpu().numpy()
    # depth_np = depth.cpu().numpy()
    
    # import cv2
    # # kernel size = 3 (填补 1 像素缝隙), size = 5 (填补更大缝隙)
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    
    # # 对 RGB 进行形态学闭运算 (Closing) 或者 膨胀 (Dilate)
    # # 闭运算 = 先膨胀后腐蚀，能填补黑点但保持轮廓
    # # 膨胀 = 会让物体稍微变胖一点点
    
    # # 建议尝试一下 Dilate，这是最立竿见影的“把点变大”的方法
    # rgb_filled = cv2.dilate(rgb_np, kernel)
    # depth_filled = cv2.dilate(depth_np, kernel) # 注意：简单的dilate对深度图可能不严谨（边缘处背景会吃掉前景），但在可视化层面足够了
    
    # return rgb_filled, depth_filled

    return rgb.cpu().numpy(), depth.cpu().numpy()

    
def export_gt_resized_video_with_ffmpeg(input_mp4, out_mp4, W, H, fps_in, subsample=1, keep_aspect=False, quiet=True):
    """
    抽帧 + resize 输出 GT 视频（与推理帧严格对齐）。
    - fps_out = round(fps_in / subsample)
    - subsample=1 时不抽帧，只做resize
    """
    os.makedirs(os.path.dirname(out_mp4), exist_ok=True)
    S = max(1, int(subsample))
    fps_out = max(1, int(round(float(fps_in) / S)))

    if keep_aspect:
        scale_filter = (
            f"scale={W}:{H}:flags=lanczos:force_original_aspect_ratio=decrease,"
            f"pad={W}:{H}:(ow-iw)/2:(oh-ih)/2"
        )
    else:
        scale_filter = f"scale={W}:{H}:flags=lanczos"

    if S > 1:
        # 精确抽每 S 帧 + resize；时长由 -r fps_out 保持不变
        vf = f"select='not(mod(n\\,{S}))',{scale_filter}"
    else:
        vf = scale_filter

    # 更安静的日志
    logflags = "-hide_banner -nostats -loglevel warning" if quiet else ""

    # 优先用 imageio_ffmpeg 自带的 ffmpeg（与 oneshot env 一起打包，免外部 PATH 依赖）。
    try:
        import imageio_ffmpeg
        FFMPEG_BIN = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        FFMPEG_BIN = "ffmpeg"

    cmd = (
        f'{shlex.quote(FFMPEG_BIN)} {logflags} -y -i {shlex.quote(input_mp4)} '
        f'-vf "{vf}" -r {fps_out} '
        f'-c:v libx264 -pix_fmt yuv420p -movflags +faststart -an '
        f'{shlex.quote(out_mp4)}'
    )
    subprocess.run(cmd, shell=True, check=True)

def _open3d_outlier_filter(P_bg, C_bg, nb_neighbors=20, std_ratio=2.0, nb_points=16, radius=0.05):
    try:
        import open3d as o3d
    except Exception:
        return P_bg, C_bg

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(P_bg.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(np.clip(C_bg, 0, 1).astype(np.float64))

    pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    pcd, ind2 = pcd.remove_radius_outlier(nb_points=nb_points, radius=radius)

    P = np.asarray(pcd.points).astype(np.float32)
    C = np.asarray(pcd.colors).astype(np.float32)
    return P, C

def render_rgbd_ply(
    pts3ds_to_vis, colors_to_vis, msks_to_vis, cam_dict,
    output_dir,                # 输出目录
    image_size,             # (H, W) —— 跟其它视频保持一致
    fps_out,                # 输出视频 FPS（比如 fps_out = round(video_fps/subsample)）
    msk_threshold=0.5,      # 人像阈值（仅在需要人像洞区填充的路径会用到）
    mask_morph=10,         # 人像掩码形态学膨胀像素（你已有的参数）
    voxel_size=0.01,        # 构建全局背景点云的体素下采样大小（米） # 0.02                 # 体素下采样 (m)，室内 1-2cm，室外可 3-5cm
    # near_far=(0.3, 12.0),   # 深度转 8-bit 反深度的近远裁剪
    near_far=(2.0, 100.0),   # 深度转 8-bit 反深度的近远裁剪
    device="cuda",
    also_export_resized_gt=False,  # 是否顺便导出 resize_gt_video（需要传入下面三个参数）
    gt_input_seq=None,      # 原始输入视频路径
    video_fps=None,            # 原视频 FPS
    subsample=1,            # 你推理解码的 subsample
    keep_aspect=False,      # resize_gt_video 是否保持长宽比
    quiet=True,
    conf_to_vis=None,
    bg_conf_thr: float = 1.25,
    bg_frame_stride: int = 1,
):
    os.makedirs(output_dir, exist_ok=True)
    H, W = int(image_size[0]), int(image_size[1])
    #============ 一次性构建“全局背景点云”（世界坐标） ============
    cam_centers = np.asarray(cam_dict["t"]).reshape(-1, 3)  # t_c2w 就是 camera center in world
    P_bg, C_bg = build_global_bg_cloud(
        pts_world_list=pts3ds_to_vis,
        colors_list=colors_to_vis,
        masks_list=msks_to_vis,
        confs_list=conf_to_vis,
        conf_thr=bg_conf_thr,          # 你选的阈值
        voxel_size=voxel_size,
        frame_stride=bg_frame_stride,
        mask_morph=4,           # ✅ 建议先 4；你之前 10 对 512x240 偏大
        verbose=(not quiet),
    )

    # P_bg, C_bg = _open3d_outlier_filter(P_bg, C_bg)

    # （可选）保存全局背景点云为 ply
    C_bg_ply = C_bg
    if C_bg_ply.dtype != np.uint8:
        # C_bg 通常是 0..1 float
        if C_bg_ply.max() <= 1.0 + 1e-6:
            C_bg_ply = (np.clip(C_bg_ply, 0, 1) * 255.0 + 0.5).astype(np.uint8)
        else:
            C_bg_ply = (np.clip(C_bg_ply, 0, 255) + 0.5).astype(np.uint8)

    write_ply_xyzrgb(os.path.join(output_dir, "bg_world_cloud_da3.ply"), P_bg, C_bg_ply)

    # # （可选）保存全局背景点云为 ply
    # write_ply_xyzrgb(os.path.join(output_dir, "bg_world_cloud_da3.ply"), P_bg, C_bg)

    # 从 cam_dict 还原每帧 K 和 cam2world
    f_list = np.array(cam_dict["focal"])    # (B,)
    pp     = np.array(cam_dict["pp"])       # (B,2)
    R_list = np.array(cam_dict["R"])        # (B,3,3)
    t_list = np.array(cam_dict["t"])        # (B,3)
    K_list = np.array(cam_dict["K"])

    rgb_writer   = iio.get_writer(os.path.join(output_dir, 'rgb_fill_da3.mp4'), fps=fps_out, ffmpeg_params=['-threads','4','-preset','veryfast'])
    depth_writer = iio.get_writer(os.path.join(output_dir, 'depth_da3.mp4'),     fps=fps_out, ffmpeg_params=['-threads','4','-preset','veryfast'])
    near, far = float(near_far[0]), float(near_far[1])

    P_bg_t = torch.from_numpy(P_bg).to(device=device, dtype=torch.float32)
    # C_bg 你这里通常是 0..1 float（open3d filter 返回的）
    C_bg_u8 = (np.clip(C_bg * 255.0, 0, 255) + 0.5).astype(np.uint8)
    C_bg_t = torch.from_numpy(C_bg_u8).to(device=device, dtype=torch.uint8)
    @torch.inference_mode()
    def render_one_frame(P_t, C_t, K_np, c2w_np, H, W, device):
        K = torch.as_tensor(K_np, dtype=torch.float32, device=device)
        c2w = torch.as_tensor(c2w_np, dtype=torch.float32, device=device)
        w2c = torch.linalg.inv(c2w)

        R = w2c[:3, :3]
        t = w2c[:3, 3]

        Pc = (P_t @ R.T) + t
        z = Pc[:, 2]
        valid = z > 1e-6
        Pc = Pc[valid]; z = z[valid]
        cols = C_t[valid]

        u = (K[0,0] * (Pc[:,0] / z) + K[0,2]).round().to(torch.int64)
        v = (K[1,1] * (Pc[:,1] / z) + K[1,2]).round().to(torch.int64)
        inside = (u>=0)&(u<W)&(v>=0)&(v<H)
        u=u[inside]; v=v[inside]; z=z[inside]; cols=cols[inside]

        idx = v*W + u
        numpix = H*W
        zbuf = torch.full((numpix,), float('inf'), dtype=torch.float32, device=device)
        zbuf.scatter_reduce_(0, idx, z, reduce='amin', include_self=True)

        keep = torch.isfinite(zbuf[idx]) & (z <= (zbuf[idx] + 1e-6))
        idxk = idx[keep]
        zk = z[keep]
        ck = cols[keep]

        depth = torch.zeros((numpix,), dtype=torch.float32, device=device)
        rgb   = torch.zeros((numpix,3), dtype=torch.uint8, device=device)
        depth[idxk] = zk
        rgb[idxk]   = ck
        return rgb.view(H,W,3).cpu().numpy(), depth.view(H,W).cpu().numpy()

    for i in range(len(pts3ds_to_vis)):
        # K
        K = K_list[i].astype(np.float32)
        # f = float(f_list[i])
        # K = np.array([[f,0,pp[i,0]],
        #             [0,f,pp[i,1]],
        #             [0,0,1]], dtype=np.float32)
        # cam2world 4x4
        c2w = np.eye(4, dtype=np.float32)
        c2w[:3,:3] = R_list[i]
        c2w[:3, 3] = t_list[i]

        # import ipdb; ipdb.set_trace()
        rgb_full, depth_full = render_one_frame(P_bg_t, C_bg_t, K, c2w, H, W, device)
        # # 回填
        # rgb_full, depth_full = fill_frame_from_global_torch(
        #     P_world=P_bg, C_world=C_bg, K=K, cam_c2w=c2w, H=H, W=W, device='cuda'
        # )

        # 1) RGB：直接 8-bit
        rgb_writer.append_data(rgb_full)
        # iio.imwrite(os.path.join(output_dir, "bg_filled_rgb_png", f"{i:06d}.png"), rgb_full) # DEBUG: 
        # 2) Depth：先转成 8-bit 反深度，再写
        d = depth_full.copy()
        mask = d > 0
        inv = np.zeros_like(d, dtype=np.float32)
        inv[mask] = (1.0/d[mask] - 1.0/far) / (1.0/near - 1.0/far)
        inv = np.clip(inv, 0.0, 1.0)
        depth8 = (inv * 255.0 + 0.5).astype(np.uint8)
        depth_writer.append_data(depth8)
        # iio.imwrite(os.path.join(output_dir, "bg_filled_depth_png", f"{i:06d}.png"), depth8) # DEBUG: 
        # import ipdb; ipdb.set_trace()

    rgb_writer.close()
    depth_writer.close()

    def _encode_from_images(img_list, out_mp4, fps, mbs=8):
        writer = iio.get_writer(
            out_mp4,
            fps=fps,
            macro_block_size=mbs,          # ★ 关键：改成 8 或 1
            ffmpeg_params=['-threads','4','-preset','veryfast']
        )
        for p in img_list:
            im = iio.imread(p)
            writer.append_data(im)
        writer.close()

    def _encode_from_images_resized(img_list, out_mp4, fps, H, W):
        writer = iio.get_writer(out_mp4, fps=fps, ffmpeg_params=['-threads','4','-preset','veryfast'])
        for p in img_list:
            im = iio.imread(p)
            if (im.shape[0] != H) or (im.shape[1] != W):
                import cv2
                im = cv2.resize(im, (W, H), interpolation=cv2.INTER_LINEAR)
            writer.append_data(im)
        writer.close()

    dst_resized  = os.path.join(output_dir, 'resize_gt_video.mp4')
    dst_original = os.path.join(output_dir, 'original_video.mp4')
    # 要求：gt_input_seq 必须是“当前 clip 的帧列表（已 subsample）”
    if isinstance(gt_input_seq, list) and len(gt_input_seq) > 0:
        if also_export_resized_gt:
            _encode_from_images_resized(gt_input_seq, dst_resized, fps_out, H, W)
        else:
            _encode_from_images(gt_input_seq, dst_original, fps_out)
    else:
        # 如果没传或传了不对，就跳过（不再拷整段视频，避免把整个 sequence 搞进来）
        print("[render_rgbd_ply] 跳过参考视频导出：请传入当前 clip 的帧列表（已 subsample）。")