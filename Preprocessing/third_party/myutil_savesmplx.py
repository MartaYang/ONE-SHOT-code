# ===================== PyTorch3D 离屏渲染（稳定尺度版）=====================
import os, numpy as np, torch, imageio.v2 as iio
from typing import Dict, List, Tuple
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    MeshRenderer, MeshRasterizer, RasterizationSettings,
    SoftPhongShader, HardPhongShader, PerspectiveCameras,
    PointLights, TexturesVertex
)
from pytorch3d.renderer.blending import BlendParams


# Inlined from viser_utils.get_color so we don't have to import viser_utils
# at module load time — viser_utils.py top-level pulls in matplotlib / viser /
# dust3r.viz, none of which is otherwise needed in our preprocessing flow.
def get_color(idx):
    root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    colors = np.loadtxt(os.path.join(root_dir, "smpl_colors.txt")).astype(int)
    return colors[idx % len(colors)]


# ----------- 小工具 -----------
_tex_cache: Dict[Tuple[int,int,str], TexturesVertex] = {}  # (pid, V, device) -> TexturesVertex

def _as_torch(x, device, dtype=None):
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    if dtype is not None:
        x = x.to(dtype)
    return x.to(device)

def get_tex_for_pid(pid: int, V: int, device: torch.device):
    """
    为单个 mesh 构造/复用顶点颜色纹理 [1,V,3]，颜色按人 ID 固定。
    用 (pid, V, device) 作为缓存键，避免重复构建。
    """
    key = (int(pid), int(V), str(device))
    if key in _tex_cache:
        return _tex_cache[key]
    c = np.array(get_color(pid), dtype=np.float32) / 255.0  # [3]
    c = torch.tensor(c, device=device).view(1, 1, 3).expand(1, V, 3)  # [1,V,3]
    tex = TexturesVertex(verts_features=c)
    _tex_cache[key] = tex
    return tex

def build_pytorch3d_renderer(img_size, device, use_soft=True, mode="naive"):
    """
    构建 PyTorch3D 渲染器（白底、离屏）。
    """
    from pytorch3d.renderer import (
        MeshRenderer, MeshRasterizer, RasterizationSettings,
        SoftPhongShader, HardPhongShader, PointLights
    )
    from pytorch3d.renderer.blending import BlendParams

    H, W = img_size  # 注意这里用 (H, W)

    if mode == "naive":
        raster_settings = RasterizationSettings(
            image_size=(H, W),   # (H, W)!
            blur_radius=1e-5,
            faces_per_pixel=8,
            cull_backfaces=False,
            bin_size=0,          # 关闭 coarse binning，杜绝 overflow
        )
    else:
        raster_settings = RasterizationSettings(
            image_size=(H, W),
            blur_radius=0.0,
            faces_per_pixel=1,
            cull_backfaces=False,
            bin_size=32,
            max_faces_per_bin=200000
        )

    lights = PointLights(device=device, location=[[0.0, 0.0, 0.0]])
    # blend = BlendParams(background_color=(1.0, 1.0, 1.0))  # 纯白
    blend = BlendParams(background_color=(0.0, 0.0, 0.0))  # 纯黑

    def make(cameras):
        shader = (SoftPhongShader if use_soft else HardPhongShader)(
            device=device, lights=lights, cameras=cameras, blend_params=blend
        )
        return MeshRenderer(
            rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
            shader=shader
        )
    return make


def load_smplx_vertex_colors_txt(txt_path: str, verts_num: int, device: torch.device):
    """
    加载一份社区常用的 smplx_verts_colors.txt（10475 行），每行 3 个数（0..1 或 0..255）。
    返回 TexturesVertex（[1,V,3]）。
    """
    arr = np.loadtxt(txt_path, dtype=np.float32)  # [V,3] or [>V,3]
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"Bad color file format: {txt_path}")
    if arr.shape[0] != verts_num:
        raise ValueError(f"Color file V={arr.shape[0]} mismatch SMPL-X V={verts_num}")
    if arr.max() > 1.0:
        arr = arr / 255.0
    feat = torch.from_numpy(arr).to(device=device).view(1, verts_num, 3)
    return TexturesVertex(verts_features=feat)


# ----------- 稳定尺度：常值 z_ref（每人固定） + 去平移 -----------
def center_and_place_constant_depth(verts_b1vx3: torch.Tensor, z_ref: float) -> torch.Tensor:
    """
    把相机系下的 mesh 平移到画面中心（x=y=0），并把 Z 平移到 z_ref（常值）。
    verts_b1vx3: [1,V,3]（torch, on device）
    """
    v = verts_b1vx3  # [1,V,3]
    center = v.mean(dim=1, keepdim=True)      # [1,1,3] 近似根部
    v = v.clone()
    v[..., 0] -= center[..., 0]               # x -> 0
    v[..., 1] -= center[..., 1]               # y -> 0
    v[..., 2] += (float(z_ref) - center[..., 2])  # z -> z_ref
    return v

# ----------- 一次/人：用 betas 计算“标准 3D 身高”（T-pose）-----------
def build_canonical_height_map(smpl_shape: List[torch.Tensor], smpl_id: List, device="cuda") -> Dict[int, float]:
    """
    返回 {pid: H_canon}，每人一个“标准身高”（T-pose，单位：3D 长度）。
    仅依赖 betas（形状），不受举手/蹲下等姿态影响。
    smpl_shape: 长度 B 的列表；第 f 个元素形状为 [n_humans_f, num_betas]
    smpl_id   : 长度 B 的列表；第 f 个元素是长度 n_humans_f 的 id 列表/张量
    """
    from dust3r.utils import SMPL_Layer  # 本项目已有
    dev = torch.device(device)
    # 找一个样本拿 betas 维度
    num_betas = 10 if len(smpl_shape) == 0 or smpl_shape[0].numel() == 0 else int(smpl_shape[0].shape[-1])
    smpl_layer = SMPL_Layer(type='smplx', gender='neutral', num_betas=num_betas,
                            kid=False, person_center='head').to(dev).eval()
    K_dummy = torch.tensor(
        [[[1000.0,   0.0, 0.0],
          [  0.0, 1000.0, 0.0],
          [  0.0,   0.0, 1.0]]],
        device=dev, dtype=torch.float32
    )  # [1,3,3]

    pid2betas = {}
    for f, ids in enumerate(smpl_id):
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        if ids is None: 
            continue
        for k, pid in enumerate(ids):
            pid = int(pid)
            if pid not in pid2betas:
                b = smpl_shape[f][k]  # [num_betas]
                pid2betas[pid] = _as_torch(b, device=dev, dtype=torch.float32).unsqueeze(0)  # [1,num_betas]

    pid2H = {}
    with torch.no_grad():
        for pid, betas in pid2betas.items():
            # T-pose：全零姿态 & 平移
            zeros_rot = torch.zeros((1, 53, 3), device=dev)
            zeros_trl = torch.zeros((1, 3), device=dev)
            out = smpl_layer(zeros_rot, betas, zeros_trl, None, None, K=K_dummy, expression=None)
            verts = out['smpl_v3d']  # [1,V,3]
            y = verts[0, :, 1]
            H_canon = float((y.max() - y.min()).item())
            pid2H[pid] = max(H_canon, 1e-6)
    return pid2H

def build_zref_map(pid2H: Dict[int,float], focal_px: float, H_img: int, target_ratio: float) -> Dict[int,float]:
    """
    由“标准身高 H_canon”推导每人的 z_ref 常值： z_ref = f * H_canon / (target_ratio * H_img)
    focal_px: 统一使用一个像素焦距（建议全序列中位数，以免抖动）
    H_img   : 图像高度（像素）
    """
    zref = {}
    h_tgt = max(target_ratio * float(H_img), 1e-6)
    for pid, Hcanon in pid2H.items():
        zref[pid] = float(focal_px * Hcanon / h_tgt)
    return zref

# ----------- 合并多人的 mesh（同帧同屏）-----------
def merge_person_meshes(verts_list: List[torch.Tensor], face_idx: torch.Tensor,
                        tex_list: List[TexturesVertex]) -> Meshes:
    """
    把同一帧的多个人合成一个 Mesh（batch=1），方便“多人同屏”渲染。
    verts_list: [ [1,V_i,3], ... ] on device
    face_idx:   [F,3] LongTensor on device（基础面片）
    tex_list:   每个人一个 TexturesVertex（[1,V_i,3]）
    """
    dev = verts_list[0].device
    # 顶点拼接
    verts_cat = torch.cat(verts_list, dim=1)  # [1, sumV, 3]
    # faces 偏移
    faces = []
    ofs = 0
    for v in verts_list:
        Vi = v.shape[1]
        faces.append(face_idx + ofs)
        ofs += Vi
    faces_cat = torch.cat(faces, dim=0).unsqueeze(0)  # [1, sumF, 3]
    # 纹理拼接
    tex_feat = torch.cat([t.verts_features_padded() for t in tex_list], dim=1)  # [1, sumV, 3]
    tex = TexturesVertex(verts_features=tex_feat)
    return Meshes(verts=verts_cat, faces=faces_cat, textures=tex)

# ----------- 单个 ID：逐帧对齐写满（没出现就白帧）-----------
def render_id_video_aligned(
    pid: int, B: int, all_smpl_verts, smpl_id, smpl_transl, face_idx: torch.Tensor,
    out_path: str, image_size: Tuple[int,int], fps_out: int, device: torch.device,
    zref_map: Dict[int,float], focal_px: float, cam_dict=None, all_delta_hp=None,
    tex_palette=None
):
    """
    - 输入 all_smpl_verts[f]: 这一帧所有人的 SMPL 顶点（世界坐标）
    - smpl_transl[f][k]: 这一帧第 k 个人的 root 平移（相机坐标系）
    - 用 cam_dict[R,t] 把世界→相机 (与 transl 对齐)，再做 root 居中 + z_ref 固定
    - 渲染时使用像素内参 (f, cx, cy)，相机置 Identity（因为顶点已经在相机系）
    """
    H, W = image_size
    dev = device
    writer = iio.get_writer(out_path, fps=fps_out, macro_block_size=1, ffmpeg_params=['-threads','4','-preset','veryfast'])
    # white = np.full((H, W, 3), 255, np.uint8)
    black = np.full((H, W, 3), 0, np.uint8)

    fx = fy = float(focal_px)
    px = W * 0.5
    py = H * 0.5

    # Identity 相机（因为我们把顶点放到相机系）
    R = torch.eye(3, device=dev)[None]
    T = torch.zeros((1,3), device=dev)
    cameras = PerspectiveCameras(
        R=R, T=T, device=dev,
        focal_length=torch.tensor([[fx, fy]], device=dev),
        principal_point=torch.tensor([[px, py]], device=dev),
        in_ndc=False, image_size=((H, W),)
    )
    make_renderer = build_pytorch3d_renderer(img_size=(H, W), device=dev, use_soft=False, mode="naive")
    renderer = make_renderer(cameras)

    for f in range(B):
        ids = smpl_id[f]
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()

        if not ids or all_smpl_verts[f].numel() == 0 or pid not in ids:
            writer.append_data(black); continue

        k = ids.index(pid)

        # 1) 取该人的 world 顶点
        v_w = _as_torch(all_smpl_verts[f][k], device=dev, dtype=torch.float32)  # [V,3]
        v_w = v_w.unsqueeze(0)  # [1,V,3]

        # 2) 世界 -> 相机（与 smpl_transl 的坐标系对齐）
        R_c2w = _as_torch(cam_dict["R"][f], device=dev, dtype=torch.float32)  # [3,3]
        t_c2w = _as_torch(cam_dict["t"][f], device=dev, dtype=torch.float32)  # [3]
        # p_cam = R^T (p_world - t)
        # v_c = (v_w - t_c2w.view(1,1,3)).matmul(R_c2w)   # [1,V,3]
        v_c = torch.einsum('ij,bkj->bki', R_c2w.T, (v_w - t_c2w.view(1,1,3)))

        # 3) root 居中 + 固定 z_ref（相机系）
        transl_fk = _as_torch(smpl_transl[f][k] + all_delta_hp[f][k], device=dev, dtype=torch.float32)  # [3]
        root_xy = transl_fk[:2]
        root_z  = float(transl_fk[2])
        z_ref   = float(zref_map[pid])

        v_c = v_c.clone()
        v_c[..., 0] -= root_xy[0]
        v_c[..., 1] -= root_xy[1]
        v_c[..., 2] =  v_c[..., 2] - root_z + z_ref

        # # 4) 坐标系对齐：OpenCV(y↓) → PyTorch3D(y↑)，只翻 y
        v_c[..., 1] *= -1.0
        v_c[..., 0] *= -1.0

        # 5) 上色 & 渲染
        # import ipdb; ipdb.set_trace()
        V = v_c.shape[1]
        if (tex_palette is not None) and (tex_palette.verts_features_packed().shape[0] == V):
            tex = tex_palette
        else:
            tex = get_tex_for_pid(pid, V, dev)   # fallback：纯色（按 id）
        mesh = Meshes(verts=v_c, faces=face_idx[None], textures=tex)

        img = renderer(mesh, cameras=cameras)         # [1,H,W,4]；RGB 已含白底
        frame = (img[0, :, :, :3].clamp(0,1).detach().cpu().numpy() * 255.0 + 0.5).astype(np.uint8)
        writer.append_data(frame)
        # import ipdb; ipdb.set_trace()
        # iio.imwrite(os.path.join('out/clip_900_1100/smplx_mesh', f"mplx_mesh_{f:06d}.png"), frame) # DEBUG: 

    writer.close()
    # import ipdb; ipdb.set_trace()


# ----------- 多人同屏：按真实相机参数渲染（每帧都写）----------
def render_camera_video_aligned(
    B: int,
    all_smpl_verts,        # [B]，每帧一个 list 或 tensor；元素是“世界坐标”的 SMPL 顶点
    smpl_id,               # [B]，每帧的人 ID 列表
    face_idx: torch.Tensor,
    cam_dict,              # 包含 R(=R_c2w), t(=t_c2w), focal, pp
    out_path: str,
    image_size: Tuple[int,int],  # (H, W) —— 和你其它视频一致
    fps_out: int,
    device: torch.device,
):
    H, W = image_size
    dev = device
    writer = iio.get_writer(out_path, fps=fps_out, ffmpeg_params=['-threads','4','-preset','veryfast'])
    # white = np.full((H, W, 3), 255, np.uint8)
    black = np.full((H, W, 3), 255, np.uint8)

    # import ipdb; ipdb.set_trace()
    R_c2w_all = _as_torch(cam_dict["R"], device=dev, dtype=torch.float32) # [b, 3,3]
    t_c2w_all = _as_torch(cam_dict["t"], device=dev, dtype=torch.float32)  # [b, 3]
    R_w2c_all = R_c2w_all.transpose(1,2)
    t_w2c_all = - torch.einsum('bij, bj->bi', R_c2w_all.transpose(1,2), t_c2w_all)

    # fx = fy = float(cam_dict["focal"][0])
    # px, py = cam_dict["pp"][0]
    focal_all = np.asarray(cam_dict["focal"]).reshape(-1)   # [B]，你管线里 fx=fy
    pp_all    = np.asarray(cam_dict["pp"]).reshape(-1, 2)   # [B,2]

    # cameras = PerspectiveCameras(
    #     R=torch.eye(3, device=dev)[None],
    #     T=torch.zeros((1,3), device=dev),
    #     focal_length=torch.tensor([[fx, fy]], device=dev),
    #     principal_point=torch.tensor([[px, py]], device=dev),
    #     in_ndc=False, image_size=((H, W),), device=dev
    # )

    for f in range(B):
        # 空帧处理
        if all_smpl_verts[f].numel() == 0:
            writer.append_data(black); continue
        ids = smpl_id[f]
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        if not ids:
            writer.append_data(black); continue

        # 1) 当帧相机参数（OpenCV：R_c2w, t_c2w）
        R_w2c = R_w2c_all[f]
        t_w2c = t_w2c_all[f]

        # 2) 把“世界坐标”的每个人顶点变到“相机坐标”
        verts_list_cam, tex_list = [], []
        for k, pid in enumerate(ids):
            v_w = _as_torch(all_smpl_verts[f][k], device=dev, dtype=torch.float32)  # [V,3] 世界
            v_c = (R_w2c @ v_w.T + t_w2c[:, None]).T.unsqueeze(0)                   # [1,V,3] 相机
            v_c[..., 0:2] *= -1.0
            V = v_c.shape[1]
            verts_list_cam.append(v_c)
            tex_list.append(get_tex_for_pid(int(pid), V, dev))

        if len(verts_list_cam) == 0:
            writer.append_data(black); continue

        # 3) 合并多人 mesh
        verts_cat = torch.cat(verts_list_cam, dim=1)  # [1, sumV, 3]
        faces = []
        ofs = 0
        for v in verts_list_cam:
            Vi = v.shape[1]
            faces.append(face_idx + ofs)
            ofs += Vi
        faces_cat = torch.cat(faces, dim=0).unsqueeze(0)  # [1, sumF, 3]
        tex_feat = torch.cat([t.verts_features_padded() for t in tex_list], dim=1)
        mesh = Meshes(verts=verts_cat, faces=faces_cat, textures=TexturesVertex(tex_feat))

        fx = fy = float(focal_all[f])
        cx, cy = map(float, pp_all[f])
        cameras = PerspectiveCameras(
            R=torch.eye(3, device=dev)[None],
            T=torch.zeros((1,3), device=dev),
            focal_length=torch.tensor([[fx, fy]], device=dev),
            principal_point=torch.tensor([[cx, cy]], device=dev),
            in_ndc=False, image_size=((H, W),), device=dev
        )

        # 5) 渲染（白底 + 方向光 + Phong），和单人那条线一致
        make_renderer = build_pytorch3d_renderer(img_size=(H, W), device=dev, use_soft=False, mode="naive")
        renderer = make_renderer(cameras)
        img = renderer(mesh, cameras=cameras)  # [1,H,W,3]
        frame = (img[0, :, :, :3].clamp(0,1).detach().cpu().numpy() * 255.0 + 0.5).astype(np.uint8)
        writer.append_data(frame)

    writer.close()


def save_square_bboxes_npy(
    out_path: str,
    smpl_transl,            # list 长度 B；第 f 个元素形状 [n_f, 3]（相机坐标系、以 head 为 center 的 transl）
    delta_hp,               # list 长度 B；第 f 个元素形状 [n_f, 3]（pelvis - head 的相机系偏移）
    smpl_id,                # list 长度 B；每帧的 id 列表/张量（长度 n_f）
    pid2H: dict,            # {pid: H_canon}，T-pose 身高（3D 长度，单位与 SMPL 一致）
    cam_dict: dict,         # 含 "focal" [B]，"pp" [B,2]；单位像素；OpenCV 投影约定
    image_size: tuple,      # (H, W) —— 仅用于 clamp
    clip_to_image: bool = True
):
    """
    生成一个长度为 B 的 list，list[f] 是 {pid: [minx,miny,maxx,maxy]}（像素坐标，float），
    然后 np.save(..., allow_pickle=True) 存为单个 .npy 文件。
    规则：
      - bbox 是正方形
      - bbox 中心 = 根 (pelvis) 的 2D 投影
      - 边长 = T-pose 身高 H_canon 在“当前帧根深度 z”下的投影长度（像素）
        L = fy * H_canon / z
    """
    H_img, W_img = int(image_size[0]), int(image_size[1])
    B = len(smpl_id)

    # 统一把 focal / pp 变成 numpy，便于索引
    focal = np.asarray(cam_dict["focal"]).reshape(-1)          # [B]
    pp    = np.asarray(cam_dict["pp"]).reshape(B, 2)           # [B,2]

    results = []
    for f in range(B):
        ids_f = smpl_id[f]
        if isinstance(ids_f, torch.Tensor):
            ids_f = ids_f.tolist()

        # 若本帧没人：给空 dict
        if not ids_f:
            results.append({})
            continue

        transl_f = smpl_transl[f]
        delta_f  = delta_hp[f]
        if isinstance(transl_f, np.ndarray):
            transl_f = torch.from_numpy(transl_f)
        if isinstance(delta_f, np.ndarray):
            delta_f = torch.from_numpy(delta_f)

        transl_f = transl_f.float().cpu()   # [n,3]
        delta_f  = delta_f.float().cpu()    # [n,3]

        fx = float(focal[f])
        fy = float(focal[f])                # 你管线里 fx==fy；若将来不等，也建议用 fy 做“身高投影”
        cx, cy = float(pp[f,0]), float(pp[f,1])

        dict_f = {}
        for k, pid in enumerate(ids_f):
            pid = int(pid)
            if k >= transl_f.shape[0]:
                continue

            # 根 (pelvis) 的 3D 相机坐标（OpenCV）
            root_cam = transl_f[k] + delta_f[k]     # [3] = head_root + (pelvis - head)
            x, y, z = float(root_cam[0]), float(root_cam[1]), float(root_cam[2])

            # 深度不合理，跳过
            if z <= 1e-6:
                continue

            # T-pose 身高（3D）
            H_canon = float(pid2H.get(pid, 1.7))    # 若没算到，兜底 1.7m（你也可以改成全局中位数）

            # 中心像素（OpenCV 像素投影）
            u = fx * (x / z) + cx
            v = fy * (y / z) + cy

            # 边长（像素）：正方形，以“竖直方向的投影长度”为准
            L = fy * (H_canon / z)

            # 正方形 bbox
            half = 0.5 * L
            xmin = u - half
            xmax = u + half
            ymin = v - half
            ymax = v + half

            # clamp 到图像范围
            if clip_to_image:
                xmin = float(np.clip(xmin, 0.0, W_img - 1.0))
                xmax = float(np.clip(xmax, 0.0, W_img - 1.0))
                ymin = float(np.clip(ymin, 0.0, H_img - 1.0))
                ymax = float(np.clip(ymax, 0.0, H_img - 1.0))

            dict_f[pid] = [xmin, ymin, xmax, ymax]

        results.append(dict_f)

    # 存一个 .npy（list[dict]），简单直观；需要矩阵化时你再读出来加工即可
    np.save(out_path, np.array(results, dtype=object), allow_pickle=True)



# 放在 myutil_savesmplx.py 顶部的 import 里补充：
import cv2

def _to_bgr(color_rgb_float):
    """get_color(pid) 返回 0..255 的 RGB，这里转 BGR 给 cv2 用。"""
    r, g, b = color_rgb_float
    return (int(b), int(g), int(r))

def visualize_bboxes_npy(
    bboxes_npy_path: str,
    out_video_path: str,
    image_size: tuple,          # (H, W) —— 跟你其它视频一致
    fps_out: int,
    bg_mode: str = "black",     # "black" | "video"
    bg_video_path: str = None,  # 当 bg_mode="video" 时提供：例如 smplx_mesh_camera.mp4 / resize_gt_video.mp4
    draw_ids: bool = False,
    draw_centers: bool = False, # 若想画出根结点中心点，需要再传 smpl_transl/delta_hp/cam_dict
    smpl_transl=None,
    delta_hp=None,
    cam_dict=None,
    thickness: int = 2,
    font_scale: float = 0.6,
    only_draw_longest: bool = False, 
):
    """
    读取 save_square_bboxes_npy 存下的 list[dict]，把每帧每个 pid 的正方形 bbox 叠加到背景上，
    生成一个校验视频。背景可选纯白或某个已渲染/已对齐分辨率的视频。
    """
    H, W = int(image_size[0]), int(image_size[1])
    arr = np.load(bboxes_npy_path, allow_pickle=True)
    B = len(arr)

    # 背景帧读取器
    reader = None
    if bg_mode == "video":
        assert bg_video_path is not None and os.path.isfile(bg_video_path), \
            f"bg_video_path invalid: {bg_video_path}"
        reader = iio.get_reader(bg_video_path)
        meta = reader.get_meta_data()
        # 如果帧数不完全相等也没关系：越界时回落到白底
        # 若分辨率不一致，下面会 resize 到 (W,H)

    writer = iio.get_writer(out_video_path, fps=fps_out, ffmpeg_params=['-threads','4','-preset','veryfast'])

    for f in range(B):
        # 背景（保持 RGB）
        if reader is not None:
            try:
                bg = reader.get_data(f)  # RGB
                if bg.shape[0] != H or bg.shape[1] != W:
                    bg = cv2.resize(bg, (W, H), interpolation=cv2.INTER_LINEAR)
                if bg.ndim == 2:
                    bg = cv2.cvtColor(bg, cv2.COLOR_GRAY2RGB)   # ← 转成 RGB（不是 BGR）
                elif bg.shape[2] == 4:
                    bg = cv2.cvtColor(bg, cv2.COLOR_RGBA2RGB)   # ← 转成 RGB
            except Exception:
                bg = np.full((H, W, 3), 255, np.uint8)
        else:
            bg = np.full((H, W, 3), 255, np.uint8)  # RGB

        boxes_f = arr[f]
        if isinstance(boxes_f, np.ndarray):
            boxes_f = boxes_f.item()
        if boxes_f is None:
            writer.append_data(bg); continue

        # 画框前：RGB -> BGR
        bg_bgr = cv2.cvtColor(bg, cv2.COLOR_RGB2BGR)

        for pid_str, box in boxes_f.items():
            pid = int(pid_str) if not isinstance(pid_str, int) else pid_str
            if only_draw_longest and pid!=0:
                continue
            xmin, ymin, xmax, ymax = map(float, box)
            p1 = (int(round(xmin)), int(round(ymin)))
            p2 = (int(round(xmax)), int(round(ymax)))

            if only_draw_longest:
                color_bgr = (0, 255, 0)
            else:
                color_bgr = _to_bgr(get_color(pid))  # get_color 是 RGB，这里转 BGR 给 cv2 用
            cv2.rectangle(bg_bgr, p1, p2, color_bgr, thickness)
            if draw_ids:
                label = f"id={pid}"
                tx = max(0, p1[0] + 2)
                ty = max(0, p1[1] + 18)
                cv2.putText(bg_bgr, label, (tx, ty),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, color_bgr, 1, cv2.LINE_AA)

        # 画完：BGR -> RGB，再写视频
        bg_rgb = cv2.cvtColor(bg_bgr, cv2.COLOR_BGR2RGB)
        writer.append_data(bg_rgb)

    writer.close()
    if reader is not None:
        reader.close()
    print(f"[BBox-Vis] saved to: {out_video_path}")

def visualize_bboxes_npy_fast(
    bboxes_npy_path: str,
    out_video_path: str,
    image_size: tuple,      # (H, W)
    fps_out: int,
    bg_mode: str = "black", # "black" | "video"
    bg_video_path: str = None,
    draw_ids: bool = False,
    draw_centers: bool = False,  # 这里先不改这块逻辑
    smpl_transl=None,
    delta_hp=None,
    cam_dict=None,
    thickness: int = 2,
    font_scale: float = 0.6,
    only_draw_longest: bool = False,
):
    import cv2, numpy as np, os

    H, W = map(int, image_size)
    arr = np.load(bboxes_npy_path, allow_pickle=True)
    B = len(arr)

    # --- 背景读取器（OpenCV 顺序解码） ---
    cap = None
    if bg_mode == "video":
        assert bg_video_path and os.path.isfile(bg_video_path), f"bg_video_path invalid: {bg_video_path}"
        cap = cv2.VideoCapture(bg_video_path)
        if not cap.isOpened():
            raise RuntimeError(f"cannot open video: {bg_video_path}")

    # --- 视频写出（BGR） ---
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")   # 兼容性最好；要 H264 就 "avc1"/"H264"
    writer = cv2.VideoWriter(out_video_path, fourcc, fps_out, (W, H))
    if not writer.isOpened():
        raise RuntimeError(f"cannot open writer: {out_video_path}")

    # 预分配白底（BGR）
    # white_bgr = np.full((H, W, 3), 255, np.uint8)
    black_bgr = np.full((H, W, 3), 0, np.uint8)

    # 预计算最长 id 颜色 & 各 pid 颜色（避免每帧 get_color）
    def _to_bgr(c_rgb):
        r, g, b = c_rgb
        return int(b), int(g), int(r)
    color_cache = {}
    green = (0, 255, 0)

    # 顺序遍历帧
    for f in range(B):
        # 背景（BGR）
        if cap is not None:
            ok, frame_bgr = cap.read()  # 顺序读，最快
            if not ok:
                frame_bgr = black_bgr
            else:
                if frame_bgr.shape[1] != W or frame_bgr.shape[0] != H:
                    frame_bgr = cv2.resize(frame_bgr, (W, H), interpolation=cv2.INTER_LINEAR)
        else:
            frame_bgr = black_bgr.copy()  # 用 copy 避免写到共享白底上

        # 取该帧 boxes
        boxes_f = arr[f]
        if isinstance(boxes_f, np.ndarray):
            boxes_f = boxes_f.item()
        if boxes_f is None:
            writer.write(frame_bgr); continue

        # 画框（BGR）
        for pid_str, box in boxes_f.items():
            pid = int(pid_str) if not isinstance(pid_str, int) else pid_str
            if only_draw_longest and pid != 0:
                continue

            xmin, ymin, xmax, ymax = box  # 已是 float
            p1 = (int(xmin + 0.5), int(ymin + 0.5))
            p2 = (int(xmax + 0.5), int(ymax + 0.5))

            if only_draw_longest:
                color_bgr = green
            else:
                if pid not in color_cache:
                    color_cache[pid] = _to_bgr(get_color(pid))
                color_bgr = color_cache[pid]

            cv2.rectangle(frame_bgr, p1, p2, color_bgr, thickness)

            if draw_ids:
                # putText 也挺慢，尽量少用；确需的话使用 8 位图像+LINE_AA 已经是最快组合之一
                label = f"id={pid}"
                tx = p1[0] + 2
                ty = max(0, p1[1] + int(18 * font_scale))
                cv2.putText(frame_bgr, label, (tx, ty),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, color_bgr, 1, cv2.LINE_AA)

        writer.write(frame_bgr)

    writer.release()
    if cap is not None:
        cap.release()
    print(f"[BBox-Vis] saved to: {out_video_path}")



def scale_cam_dict_for_size(cam_dict, H_model, W_model, H_out, W_out):
    """
    把 cam_dict 的内参从 (H_model, W_model) 缩放到 (H_out, W_out)。
    - cam_dict['focal'] 是标量列表 -> 这里按各向独立生成 fx, fy（保持等比：fx用宽向缩放，fy用高向缩放）
    - cam_dict['pp']    是 [cx, cy] 列表
    返回新字典，不改原始。
    """
    import copy, numpy as np
    out = copy.deepcopy(cam_dict)
    sx = float(W_out) / float(W_model)
    sy = float(H_out) / float(H_model)

    focal = np.asarray(cam_dict["focal"]).reshape(-1)
    # 生成 fx, fy 两列（原本 focal 是一个列，这里展开成两列，后续在用的地方按 fx=fy 或分别取都行）
    fx = focal * sx
    fy = focal * sy
    out["focal_fx_fy"] = np.stack([fx, fy], axis=1).tolist()  # 新增字段：[B,2]
    out["focal"] = ((fx + fy) * 0.5).tolist()

    pp = np.asarray(cam_dict["pp"])
    pp_scaled = pp.copy()
    pp_scaled[:, 0] = pp[:, 0] * sx  # cx
    pp_scaled[:, 1] = pp[:, 1] * sy  # cy
    out["pp"] = pp_scaled.tolist()
    return out




# ----------- 总入口：生成每个 ID 的视频（居中+固定尺度）& 可选相机版 -----------
def render_smplx_videos_pytorch3d(
    all_smpl_verts, smpl_faces, smpl_id, smpl_shape, smpl_transl, cam_dict, cam_dict_bbox,
    out_dir, image_size: Tuple[int,int], image_size_bbox: Tuple[int,int], fps_out: int,
    device="cuda", target_pixel_ratio=0.6, keep_camera_video=True, all_delta_hp=None,
    tex_palette: torch.Tensor = None,   # ← 新增
):
    """
    - 对每个 pid 生成：smplx_mesh_{pid}.mp4（帧级严格对齐、白底、root 居中 + 常值 z_ref）
    - 可选再生成：smplx_mesh_camera.mp4（真实相机、多人与一帧）
    """
    os.makedirs(out_dir, exist_ok=True)
    dev = torch.device(device)
    H, W = image_size
    B = len(all_smpl_verts)
    face_idx = _as_torch(smpl_faces, device=dev, dtype=torch.int64)

    # 全部 pid
    all_ids = set()
    for f in range(B):
        ids = smpl_id[f]
        if isinstance(ids, torch.Tensor): ids = ids.tolist()
        if ids:
            for pid in ids: all_ids.add(int(pid))

    # 1) 每人标准身高（T-pose）
    pid2H = build_canonical_height_map(smpl_shape, smpl_id, device=dev)

    # 2) 统一像素焦距（中位数避免抖动）
    focal_px = float(np.median(np.asarray(cam_dict["focal"]).reshape(-1)))

    # 3) 固定 z_ref
    zref_map = build_zref_map(pid2H, focal_px=focal_px, H_img=H, target_ratio=target_pixel_ratio)

    # 4) 逐 ID 渲染
    for pid in sorted(all_ids):
        out_path = os.path.join(out_dir, f"smplx_mesh_{pid}_black.mp4")
        render_id_video_aligned(
            pid, B, all_smpl_verts, smpl_id, smpl_transl, face_idx,
            out_path, (H, W), fps_out, dev, zref_map, focal_px=focal_px, cam_dict=cam_dict, all_delta_hp=all_delta_hp, tex_palette=tex_palette
        )

    # 5) 可选多人同屏
    if keep_camera_video:
        out_cam = os.path.join(out_dir, "smplx_mesh_camera.mp4")
        render_camera_video_aligned(
            B, all_smpl_verts, smpl_id, face_idx, cam_dict,
            out_cam, image_size_bbox, fps_out, dev
        )

    bbox_path = os.path.join(out_dir, "bboxes_cam_square.npy")
    save_square_bboxes_npy(
        out_path=bbox_path,
        smpl_transl=smpl_transl,
        delta_hp=all_delta_hp,
        smpl_id=smpl_id,
        pid2H=pid2H,
        cam_dict=cam_dict_bbox,
        image_size=image_size_bbox,
        clip_to_image=False,    
    )
    print(f"[BBox] saved to: {bbox_path}")
# ===================== 以上为可直接使用的渲染段落 =====================



