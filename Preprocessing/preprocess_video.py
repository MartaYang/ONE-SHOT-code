#!/usr/bin/env python3
"""Single-video preprocessing for ONE-SHOT.

Adapted from Human3R_DA3/01_demo_v2_Demo.py with three changes:
  1. third_party/ is vendored locally (no Human3R_DA3 path dependency).
  2. clip_len_s / stride_s / min_total_s exposed as CLI args (defaults 6/5/6).
  3. Human3R + DA3 ckpt paths default to constants.HUMAN3R_CKPT / DA3_CKPT_DIR.

Outputs (per clip): same layout as Human3R_DA3 demo
  <out_dir>/<seq_name>/clip_<tag>/
    original_video.mp4
    rgb_fill_da3.mp4
    depth_da3.mp4
    smplx_pred_params_all.npz
    camera_and_humanmask.npz
    bg_world_cloud_da3.ply
    control_signals/
      smplx_mesh_camera.mp4
      smplx_mesh_0_black.mp4
      bboxes_cam_square.npy
"""
from __future__ import annotations

import argparse
import os
import sys

# Make conda env's bin (ffmpeg/ffprobe) discoverable by myutil_parse_clips subprocess.
_PYBIN_DIR = os.path.dirname(os.path.abspath(sys.executable))
if _PYBIN_DIR not in os.environ.get("PATH", "").split(os.pathsep):
    os.environ["PATH"] = _PYBIN_DIR + os.pathsep + os.environ.get("PATH", "")

# --- Path injection: vendored code lives under ./third_party/ ---
_PREPROC_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _PREPROC_DIR)  # so `import constants` works
from constants import (  # noqa: E402
    DA3_CKPT_DIR,
    DEFAULT_CLIP_LEN_S,
    DEFAULT_MIN_TOTAL_S,
    DEFAULT_STRIDE_S,
    HUMAN3R_CKPT,
    THIRD_PARTY_DIR,
    ensure_smpl_symlinks,
    setup_torch_hub,
)

sys.path.insert(0, THIRD_PARTY_DIR)
# Wire up SMPL/SMPLX body-model symlinks against PREPROCESS_ROOT.
ensure_smpl_symlinks()
# Point torch.hub at the pre-extracted facebookresearch_dinov2_main/ so that
# Dinov2Backbone(torch.hub.load(...)) runs fully offline.
setup_torch_hub()

# --- Begin: copy of 01_demo_v2_Demo.py logic, lightly patched ---
import glob
import math
import random
import tempfile
import time
from copy import deepcopy

import cv2
import imageio.v2 as iio  # noqa: F401  (kept for parity)
import numpy as np
import roma  # noqa: F401  (kept for parity)
import torch

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

random.seed(42)

from add_ckpt_path import add_path_to_dust3r  # noqa: E402
from myutil_savergbd_da3 import render_bg_rgbd_with_da3  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="ONE-SHOT preprocessing (Human3R + DA3).")
    parser.add_argument("--seq_path", type=str, required=True, help="Input video file path.")
    parser.add_argument("--output_dir", type=str, required=True, help="Output dir (a clip_* subdir is created under it).")
    parser.add_argument("--model_path", type=str, default=HUMAN3R_CKPT, help="Human3R ckpt path.")
    parser.add_argument("--da3_ckpt_dir", type=str, default=DA3_CKPT_DIR, help="DA3 ckpt dir.")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--vis_threshold", type=float, default=1.5)
    parser.add_argument("--msk_threshold", type=float, default=0.1)
    parser.add_argument("--clip_len_s", type=float, default=DEFAULT_CLIP_LEN_S)
    parser.add_argument("--stride_s", type=float, default=DEFAULT_STRIDE_S)
    parser.add_argument("--min_total_s", type=float, default=DEFAULT_MIN_TOTAL_S)
    parser.add_argument("--whole_video", action="store_true", help="Force single-clip = whole video (overrides clip_len_s).")
    parser.add_argument("--max_clips", type=int, default=None,
                        help="Stop after this many clips (default: process all). Use 1 for fast preview.")
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--subsample", type=int, default=1)
    parser.add_argument("--reset_interval", type=int, default=10000000)
    parser.add_argument("--use_ttt3r", action="store_true", default=False)
    parser.add_argument("--downsample_factor", type=int, default=10)
    parser.add_argument("--smpl_downsample", type=int, default=1)
    parser.add_argument("--camera_downsample", type=int, default=1)
    parser.add_argument("--mask_morph", type=int, default=5)
    parser.add_argument("--save_smpl", action="store_true")
    parser.add_argument("--save_video", action="store_true")
    parser.add_argument("--save_smplx_camera", action="store_true")
    parser.add_argument("--save_render_rgb", action="store_true")
    parser.add_argument("--fix_main_id0", action="store_false")
    parser.add_argument("--render_bbox", action="store_false")
    parser.add_argument("--max_fps", type=float, default=30.0)
    parser.add_argument("--skip_multi_person_filter", action="store_true", default=True,
                        help="(default true) keep behaviour of Demo.py — no multi-person/height reject.")
    return parser.parse_args()


def prepare_input(img_paths, img_mask, size, raymaps=None, raymap_mask=None,
                  revisit=1, update=True, img_res=None, reset_interval=100):
    from dust3r.utils.image import load_images, pad_image
    from dust3r.utils.geometry import get_camera_parameters

    images = load_images(img_paths, size=size)
    if img_res is not None:
        K_mhmr = get_camera_parameters(img_res, device="cpu")

    views = []
    if raymaps is None and raymap_mask is None:
        for i in range(len(images)):
            view = {
                "img": images[i]["img"],
                "ray_map": torch.full(
                    (images[i]["img"].shape[0], 6, images[i]["img"].shape[-2], images[i]["img"].shape[-1]),
                    torch.nan,
                ),
                "true_shape": torch.from_numpy(images[i]["true_shape"]),
                "idx": i, "instance": str(i),
                "camera_pose": torch.from_numpy(np.eye(4, dtype=np.float32)).unsqueeze(0),
                "img_mask": torch.tensor(True).unsqueeze(0),
                "ray_mask": torch.tensor(False).unsqueeze(0),
                "update": torch.tensor(True).unsqueeze(0),
                "reset": torch.tensor((i + 1) % reset_interval == 0).unsqueeze(0),
            }
            if img_res is not None:
                view["img_mhmr"] = pad_image(view["img"], img_res)
                view["K_mhmr"] = K_mhmr
            views.append(view)
            if (i + 1) % reset_interval == 0:
                overlap_view = deepcopy(view)
                overlap_view["reset"] = torch.tensor(False).unsqueeze(0)
                views.append(overlap_view)
    else:
        # original branch (kept for parity; not used in single-video preprocessing)
        raise NotImplementedError
    if revisit > 1:
        new_views = []
        for r in range(revisit):
            for i, view in enumerate(views):
                new_view = deepcopy(view)
                new_view["idx"] = r * len(views) + i
                new_view["instance"] = str(r * len(views) + i)
                if r > 0 and not update:
                    new_view["update"] = torch.tensor(False).unsqueeze(0)
                new_views.append(new_view)
        return new_views
    return views


from myutil_gt import build_pid_remap_by_frequency, remap_smpl_id_list, save_smplx_pred_params  # noqa: E402


def prepare_output(outputs, outdir, revisit=1, use_pose=True,
                   save_smpl=False, save_video=False, img_res=None, subsample=1,
                   gt_dataset=None, gt_smpl_pkl=None, gt_override_transl=False, gt_frame_ids=None):
    from dust3r.utils.camera import pose_encoding_to_camera
    from dust3r.post_process import estimate_focal_knowing_depth
    from dust3r.utils.geometry import geotrf, matrix_cumprod
    from dust3r.utils import SMPL_Layer
    from dust3r.utils.image import unpad_image

    valid_length = len(outputs["pred"]) // revisit
    outputs["pred"] = outputs["pred"][-valid_length:]
    outputs["views"] = outputs["views"][-valid_length:]

    reset_mask = torch.cat([view["reset"] for view in outputs["views"]], 0)
    shifted_reset_mask = torch.cat([torch.tensor(False).unsqueeze(0), reset_mask[:-1]], dim=0)
    outputs["pred"] = [pred for pred, mask in zip(outputs["pred"], shifted_reset_mask) if not mask]
    outputs["views"] = [view for view, mask in zip(outputs["views"], shifted_reset_mask) if not mask]
    reset_mask = reset_mask[~shifted_reset_mask]

    pts3ds_self_ls = [output["pts3d_in_self_view"] for output in outputs["pred"]]
    pts3ds_other = [output["pts3d_in_other_view"] for output in outputs["pred"]]
    conf_self = [output["conf_self"] for output in outputs["pred"]]
    conf_other = [output["conf"] for output in outputs["pred"]]
    pts3ds_self = torch.cat(pts3ds_self_ls, 0)

    pr_poses = [pose_encoding_to_camera(pred["camera_pose"].clone()).cpu() for pred in outputs["pred"]]
    if reset_mask.any():
        pr_poses = torch.cat(pr_poses, 0)
        identity = torch.eye(4, device=pr_poses.device)
        reset_poses = torch.where(reset_mask.unsqueeze(-1).unsqueeze(-1), pr_poses, identity)
        cumulative_bases = matrix_cumprod(reset_poses)
        shifted_bases = torch.cat([identity.unsqueeze(0), cumulative_bases[:-1]], dim=0)
        pr_poses = torch.einsum("bij,bjk->bik", shifted_bases, pr_poses)
        pr_poses = list(pr_poses.unsqueeze(1).unbind(0))

    R_c2w = torch.cat([pr_pose[:, :3, :3] for pr_pose in pr_poses], 0)
    t_c2w = torch.cat([pr_pose[:, :3, 3] for pr_pose in pr_poses], 0)
    if use_pose:
        transformed_pts3ds_other = []
        for pose, pself in zip(pr_poses, pts3ds_self):
            transformed_pts3ds_other.append(geotrf(pose, pself.unsqueeze(0)))
        pts3ds_other = transformed_pts3ds_other
        conf_other = conf_self

    B, H, W, _ = pts3ds_self.shape
    pp = torch.tensor([W // 2, H // 2], device=pts3ds_self.device).float().repeat(B, 1)
    focal = estimate_focal_knowing_depth(pts3ds_self, pp, focal_mode="weiszfeld")

    colors = [0.5 * (output["img"].permute(0, 2, 3, 1) + 1.0) for output in outputs["views"]]
    cam_dict = {"focal": focal.numpy(), "pp": pp.numpy(), "R": R_c2w.numpy(), "t": t_c2w.numpy()}

    pts3ds_self_tosave = pts3ds_self
    depths_tosave = pts3ds_self_tosave[..., 2]  # noqa: F841
    pts3ds_other_tosave = torch.cat(pts3ds_other)  # noqa: F841
    conf_self_tosave = torch.cat(conf_self)  # noqa: F841
    conf_other_tosave = torch.cat(conf_other)  # noqa: F841
    cam2world_tosave = torch.cat(pr_poses)
    intrinsics_tosave = torch.eye(3).unsqueeze(0).repeat(cam2world_tosave.shape[0], 1, 1)
    intrinsics_tosave[:, 0, 0] = focal.detach()
    intrinsics_tosave[:, 1, 1] = focal.detach()
    intrinsics_tosave[:, 0, 2] = pp[:, 0]
    intrinsics_tosave[:, 1, 2] = pp[:, 1]

    smpl_shape = [output.get("smpl_shape", torch.empty(1, 0, 10))[0] for output in outputs["pred"]]
    smpl_rotvec = [roma.rotmat_to_rotvec(output.get("smpl_rotmat", torch.empty(1, 0, 53, 3, 3))[0]) for output in outputs["pred"]]
    smpl_transl = [output.get("smpl_transl", torch.empty(1, 0, 3))[0] for output in outputs["pred"]]
    smpl_expression = [output.get("smpl_expression", [None])[0] for output in outputs["pred"]]
    smpl_id = [output.get("smpl_id", torch.empty(1, 0))[0] for output in outputs["pred"]]

    has_mask = "msk" in outputs["pred"][0]
    if has_mask:
        msks = [output["msk"][..., 0] for output in outputs["pred"]]
        if img_res is not None:
            msks = [unpad_image(m, [H, W]) for m in msks]
    else:
        msks = [torch.zeros(1, H, W) for _ in range(B)]

    remap = build_pid_remap_by_frequency(smpl_id)
    smpl_id = remap_smpl_id_list(smpl_id, remap)

    smpl_layer = SMPL_Layer(type="smplx", gender="neutral",
                            num_betas=smpl_shape[0].shape[-1], kid=False, person_center="head")
    smpl_faces = smpl_layer.bm_x.faces

    all_verts = []
    all_delta_hp = []
    for f_id in range(B):
        n_humans_i = smpl_shape[f_id].shape[0]
        if n_humans_i > 0:
            with torch.no_grad():
                smpl_out = smpl_layer(
                    smpl_rotvec[f_id], smpl_shape[f_id], smpl_transl[f_id],
                    None, None,
                    K=intrinsics_tosave[f_id].expand(n_humans_i, -1, -1),
                    expression=smpl_expression[f_id],
                )
                j3d_cam = smpl_out.get("smpl_j3d", smpl_out.get("j3d_cam", None))
                pelvis_idx, head_idx = 0, 15
                pelvis_cam = j3d_cam[:, pelvis_idx, :]
                head_cam = j3d_cam[:, head_idx, :]
                delta_hp = pelvis_cam - head_cam
                all_delta_hp.append(delta_hp.cpu())
            all_verts.append(geotrf(pr_poses[f_id], smpl_out["smpl_v3d"].unsqueeze(0))[0])
        else:
            all_delta_hp.append(torch.empty(0))
            all_verts.append(torch.empty(0))

    return (pts3ds_other, colors, conf_other, cam_dict, all_verts, smpl_faces,
            smpl_id, msks, smpl_shape, smpl_transl, smpl_rotvec, smpl_expression, all_delta_hp)


def run_inference(args):
    # Resolve to absolute paths so callers can use relative paths regardless of cwd.
    args.seq_path = os.path.abspath(args.seq_path)
    args.output_dir = os.path.abspath(args.output_dir)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    add_path_to_dust3r(args.model_path)
    from dust3r.inference import inference_recurrent_lighter
    from dust3r.model import ARCroco3DStereo
    from myutil_savesmplx import render_smplx_videos_pytorch3d, load_smplx_vertex_colors_txt
    from myutil_parse_clips import parse_and_chunk_seq

    print(f"[Preprocessing] loading Human3R from {args.model_path}")
    _t_load = time.time()
    model = ARCroco3DStereo.from_pretrained(args.model_path).to(device)
    model.eval()
    print(f"[Preprocessing] Human3R loaded in {time.time()-_t_load:.1f}s")

    # ---- Sanity: warn (not fail) if shorter than inference minimum ----
    _cap = cv2.VideoCapture(args.seq_path)
    if not _cap.isOpened():
        raise RuntimeError(f"cannot open video: {args.seq_path}")
    _fps = _cap.get(cv2.CAP_PROP_FPS) or 0.0
    _nfr = int(_cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    _cap.release()
    _dur = _nfr / _fps if _fps > 0 else 0.0
    print(f"[Preprocessing] {args.seq_path}: {_nfr} frames @ {_fps:.2f} fps = {_dur:.2f}s")
    if _dur < 5.0:
        print(f"[Preprocessing] WARNING: duration {_dur:.2f}s < 5s; inference will pad → "
              f"results may be degraded.")

    if args.whole_video:
        clip_len_s, stride_s, min_total_s = 1e9, 1e9, 1e9
    else:
        clip_len_s, stride_s, min_total_s = args.clip_len_s, args.stride_s, args.min_total_s

    clips = parse_and_chunk_seq(
        args.seq_path,
        clip_len_s=clip_len_s, stride_s=stride_s, min_total_s=min_total_s,
        target_fps=16.0 * args.subsample,
    )

    smplx_color_tex = load_smplx_vertex_colors_txt(
        os.path.join(THIRD_PARTY_DIR, "smplx_verts_colors.txt"), 10475, torch.device(device)
    )

    seq_stem = os.path.splitext(os.path.basename(args.seq_path))[0]
    seq_out_root = os.path.join(args.output_dir, seq_stem)
    os.makedirs(seq_out_root, exist_ok=True)

    out_clip_dirs = []
    _t_clips_start = time.time()
    for ci, clip in enumerate(clips):
        if args.max_clips is not None and ci >= args.max_clips:
            print(f"[Preprocessing] max_clips={args.max_clips} reached, breaking.")
            break
        _t_clip = time.time()
        img_paths = clip["img_paths"]
        video_fps = clip["fps"]
        H0, W0 = clip["orig_size"]
        tmpdirname = clip["tmpdir"]
        tag = clip["clip_tag"]

        out_dir_clip = os.path.join(seq_out_root, f"clip_{tag}")
        os.makedirs(out_dir_clip, exist_ok=True)
        out_clip_dirs.append(out_dir_clip)

        if not img_paths:
            print(f"[Preprocessing] no frames in clip {tag}, skipping")
            continue

        auto_sub = 1
        if args.max_fps is not None and video_fps > args.max_fps + 1e-6:
            auto_sub = int(math.ceil(video_fps / float(args.max_fps)))
        subsample_eff = max(int(args.subsample), auto_sub)
        img_paths = img_paths[::subsample_eff]
        if args.max_frames is not None:
            img_paths = img_paths[: args.max_frames]
        video_fps_eff = video_fps / subsample_eff
        fps_out = max(1, int(round(video_fps_eff)))

        img_mask = [True] * len(img_paths)
        img_res = getattr(model, "mhmr_img_res", None)
        views = prepare_input(img_paths=img_paths, img_mask=img_mask, size=args.size,
                              revisit=1, update=True, img_res=img_res, reset_interval=args.reset_interval)
        print(f"[Preprocessing] clip {tag}: {len(views)} views @ fps={fps_out}, infer...")
        t0 = time.time()
        outputs, _ = inference_recurrent_lighter(views, model, device, use_ttt3r=args.use_ttt3r)
        print(f"[Preprocessing] clip {tag}: infer done in {time.time() - t0:.1f}s")

        (pts3ds_other, colors, conf, cam_dict, all_smpl_verts, smpl_faces,
         smpl_id, msks, smpl_shape, smpl_transl, smpl_rotvec, smpl_expression,
         all_delta_hp) = prepare_output(
            outputs, out_dir_clip, 1, True,
            args.save_smpl, args.save_video, img_res, subsample_eff,
        )

        # Optional multi-person reject (kept off by default — same as Demo.py)
        if not args.skip_multi_person_filter:
            pass  # placeholder; original code commented-out — preserve behaviour

        save_smplx_pred_params(
            outdir=out_dir_clip,
            smpl_rotvec=smpl_rotvec, smpl_shape=smpl_shape,
            smpl_transl=smpl_transl, smpl_expression=smpl_expression,
            smpl_id=smpl_id,
        )

        msks_to_vis = [m.cpu().numpy() for m in msks]
        H_model, W_model = pts3ds_other[0].cpu().numpy().shape[1:3]
        H_smpl, W_smpl = 512, 512

        info = render_bg_rgbd_with_da3(
            img_paths=img_paths, out_dir=out_dir_clip, fps_in=fps_out,
            human_msks_to_vis=msks_to_vis, human_msk_thr=args.msk_threshold,
            da3_model_id=args.da3_ckpt_dir,
            device=device, size=args.size, export_format="npz", bg_conf_q=0.05,
            bg_frame_stride=max(1, int(round(video_fps_eff / 3))),
            voxel_size=0.005, near_far=(2.0, 100.0), mask_morph=4, quiet=True,
            gt_input_seq=img_paths, video_fps=video_fps,
            subsample=subsample_eff, tmp_dir=tmpdirname,
        )
        print(f"[Preprocessing] clip {tag}: DA3 done -> {info}")

        render_out_dir = os.path.join(out_dir_clip, "control_signals")
        render_smplx_videos_pytorch3d(
            all_smpl_verts=all_smpl_verts, smpl_faces=smpl_faces, smpl_id=smpl_id,
            smpl_shape=smpl_shape, smpl_transl=smpl_transl,
            cam_dict=cam_dict, cam_dict_bbox=cam_dict, out_dir=render_out_dir,
            image_size=(H_smpl, W_smpl), image_size_bbox=(H_model, W_model),
            fps_out=fps_out, device=device, target_pixel_ratio=0.9,
            keep_camera_video=True, all_delta_hp=all_delta_hp,
            tex_palette=smplx_color_tex,
        )
        print(f"[Preprocessing] clip {tag}: total {time.time()-_t_clip:.1f}s")

    print(f"[Preprocessing] all clips done in {time.time()-_t_clips_start:.1f}s "
          f"({len(out_clip_dirs)} clips)")

    import shutil
    tmpdirs = set([c["tmpdir"] for c in clips if c.get("tmpdir") is not None])
    for td in tmpdirs:
        shutil.rmtree(td, ignore_errors=True)

    return out_clip_dirs


def main():
    args = parse_args()
    out_dirs = run_inference(args)
    # echo back the resulting clip dirs for orchestration scripts
    for d in out_dirs:
        print(f"CLIP_DIR={d}")


if __name__ == "__main__":
    main()
