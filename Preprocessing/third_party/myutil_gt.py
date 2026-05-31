# myutil_gt.py
import os, numpy as np, torch, pickle

# 关键映射：SMPL body23 -> SMPL-X body21（不含 root；丢掉 left_hand/right_hand）
SMPL_BODY23_TO_SMPLX_BODY21 = list(range(21))  # 即 [0,1,2,...,20]

def _as_like(x, like_tensor):
    if torch.is_tensor(x): 
        return x.to(device=like_tensor.device, dtype=like_tensor.dtype)
    return torch.as_tensor(x, device=like_tensor.device, dtype=like_tensor.dtype)

def remap_smpl_id_list(smpl_id_list, id_remap):
    """对整段视频的 smpl_id 做重映射（保持每帧形状不变）"""
    out = []
    for ids in smpl_id_list:
        if torch.is_tensor(ids):
            arr = ids.detach().cpu().numpy().tolist()
            arr2 = [id_remap.get(int(p), int(p)) for p in arr]
            out.append(torch.as_tensor(arr2, device=ids.device, dtype=ids.dtype))
        else:
            arr2 = [id_remap.get(int(p), int(p)) for p in ids]
            out.append(arr2)
    return out

def build_pid_remap_by_frequency(smpl_id_list):
    """把“出场帧数最多”的人映射为 0，次多为 1 ..."""
    from collections import Counter
    c = Counter()
    for ids in smpl_id_list:
        if torch.is_tensor(ids): ids = ids.tolist()
        if ids: c.update([int(x) for x in ids])
    order = [pid for pid,_ in c.most_common()]
    return {old:new for new,old in enumerate(order)}

import os
def save_smplx_pred_params(outdir, smpl_rotvec, smpl_shape, smpl_transl, smpl_expression, smpl_id):
    """
    把变长 list 序列以 dtype=object 的形式保存为 .npz。
    读取时：np.load(path, allow_pickle=True)
    """
    os.makedirs(outdir, exist_ok=True)
    to_obj = lambda lst: np.array(
        [ (x.detach().cpu().numpy() if torch.is_tensor(x) else x) for x in lst ],
        dtype=object
    )
    np.savez_compressed(
        os.path.join(outdir, "smplx_pred_params_all.npz"),
        smpl_rotvec=to_obj(smpl_rotvec),
        smpl_shape=to_obj(smpl_shape),
        smpl_transl=to_obj(smpl_transl),
        smpl_expression=np.array([
            (None if (x is None) else (x.detach().cpu().numpy() if torch.is_tensor(x) else x))
            for x in smpl_expression
        ], dtype=object),
        smpl_id=np.array([
            (ids.detach().cpu().numpy() if torch.is_tensor(ids) else np.array(ids))
            for ids in smpl_id
        ], dtype=object)
    )




# ============== 各数据集的“就地覆盖”实现 ==============


def override_emdb_into_lists(
    smpl_rotvec,      # list len=B, [n_i, 53, 3]
    smpl_shape,       # list len=B, [n_i, beta]
    smpl_transl,      # list len=B, [n_i, 3]
    smpl_id,          # list len=B, list/tensor of ids
    emdb_pkl_path: str,
    emdb_frame_ids=None,     # list len=B, 每个输出帧对应的“EMDB原始帧号”，若 None 则用 range(B)
    pid_target: int = 0,     # 只覆盖这个 id（我们已 remap 让“最长出场者==0”）
    override_transl: bool = False,  # 仅当 person_center='pelvis' 再开
):
    assert os.path.isfile(emdb_pkl_path) and emdb_pkl_path.endswith(".pkl"), \
        f"EMDB GT 路径无效: {emdb_pkl_path}"
    annot = pickle.load(open(emdb_pkl_path, "rb"))

    poses_body = annot["smpl"]["poses_body"]     # [F, 23*3]
    betas0     = annot["smpl"]["betas"]          # [10]
    trans_w    = annot["smpl"]["trans"]          # [F, 3] world
    extr_w2c   = annot["camera"]["extrinsics"]   # [F, 4, 4] world->camera
    masks      = annot.get("good_frames_mask", None)  # [F] or None
    smpl_gender = annot["gender"]

    F = poses_body.shape[0]
    B = len(smpl_id)
    if emdb_frame_ids is None:
        emdb_frame_ids = list(range(B))
    assert len(emdb_frame_ids) == B, f"emdb_frame_ids 长度应为 B={B}"

    body23 = poses_body.reshape(F, 23, 3)
    body21_all = body23[:, SMPL_BODY23_TO_SMPLX_BODY21, :]  # [F,21,3]
    betas_rep = np.repeat(betas0.reshape(1, -1), repeats=F, axis=0)

    # 预备：如果要覆盖 transl（pelvis），先算 cam 下的 transl
    if override_transl:
        R_wc = extr_w2c[:, :3, :3]   # [F,3,3]
        t_wc = extr_w2c[:, :3, 3]    # [F,3]
        transl_cam_all = (R_wc @ trans_w[..., None]).squeeze(-1) + t_wc  # [F,3]

    for f in range(B):
        ids_f = smpl_id[f]
        if ids_f is None or (torch.is_tensor(ids_f) and ids_f.numel()==0) or (not torch.is_tensor(ids_f) and len(ids_f)==0):
            continue
        if torch.is_tensor(ids_f): 
            ids_f = ids_f.tolist()

        # 找该帧里 pid_target 的位置
        if pid_target not in ids_f:
            continue
        k = ids_f.index(pid_target)

        f_emdb = int(emdb_frame_ids[f])
        if not (0 <= f_emdb < F):
            continue
        if masks is not None and (not masks[f_emdb]):
            # 该 GT 帧被标注为 bad，可选择跳过
            continue
        # import ipdb; ipdb.set_trace()
        # 覆盖 betas（帧无关，但写入一致即可）
        # nb = min(smpl_shape[f].shape[1], betas_rep.shape[1])
        # smpl_shape[f][k, :nb] = _as_like(betas_rep[f_emdb, :nb], smpl_shape[f])

        # 覆盖 body 21（root 保留预测；手/下颌保持预测）
        smpl_rotvec[f][k, 1:22, :] = _as_like(body21_all[f_emdb], smpl_rotvec[f])

        # 可选：覆盖 transl（仅 pelvis 语义一致时才开）
        if override_transl:
            smpl_transl[f][k] = _as_like(transl_cam_all[f_emdb], smpl_transl[f])

    return smpl_rotvec, smpl_shape, smpl_transl, smpl_gender


def override_motionx(
    smpl_rotvec, smpl_shape, smpl_transl, smpl_expression, smpl_id,
    gt_path: str, apply_to_main_only: bool = False
):
    """
    期望 MotionX 的 GT npz 包含 SMPL-X 完整参数（可能的键）：
      - rotvec53_list: 长度 B 的 list；每个是 dict{pid: (53,3)}
      - betas_dict   : dict{pid: (10,)}
      - transl_list  : 长度 B 的 list；每个是 dict{pid: (3,)}
      - expr_list    : (可选) 长度 B 的 list；每个是 dict{pid: (10,)}
    任意缺失就跳过。
    """
    raise NotImplemented

# 统一入口
def override_with_gt(
    dataset: str,
    smpl_rotvec, smpl_shape, smpl_transl, smpl_expression, smpl_id,
    gt_path: str,
    apply_to_main_only: bool = False,
):
    if not dataset or not gt_path: 
        return smpl_rotvec, smpl_shape, smpl_transl, smpl_expression
    dataset = dataset.lower()
    if dataset == "emdb":
        return override_emdb(smpl_rotvec, smpl_shape, smpl_transl, smpl_expression, smpl_id,
                             gt_path=gt_path, apply_to_main_only=apply_to_main_only)
    elif dataset == "motionx":
        return override_motionx(smpl_rotvec, smpl_shape, smpl_transl, smpl_expression, smpl_id,
                                gt_npz=gt_path, apply_to_main_only=apply_to_main_only)
    else:
        print(f"[GT] Unknown dataset '{dataset}', skip overriding.")
        return smpl_rotvec, smpl_shape, smpl_transl, smpl_expression