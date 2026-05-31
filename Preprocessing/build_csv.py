#!/usr/bin/env python3
"""Build cross-modal CSVs for ONE-SHOT inference.

Three subcommands:
  - id_swap     : same video, just override specify_ID_profile_path
  - motion_swap : keep src video & scene, replace human_mesh / smplx_param with motion video's
  - scene_swap  : keep src video & motion, replace depth / geom with scene video's

CSV header (paths are absolute by default; inference: pass --train_data_root /):
  video_path, depth_rgb3_path, geom_rgb_path, human_bbox_path,
  human_mesh_path, smplx_param_path, specify_ID_profile_path, prompt
"""
from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

_PREPROC_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_PREPROC_DIR)
# So `from datasets.oneshot_data_utils import ...` works in _extract_id_profile_from_clip.
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Ensure conda env's bin (where ffmpeg lives) is on PATH for subprocesses.
_PYBIN_DIR = os.path.dirname(os.path.abspath(sys.executable))
if _PYBIN_DIR not in os.environ.get("PATH", "").split(os.pathsep):
    os.environ["PATH"] = _PYBIN_DIR + os.pathsep + os.environ.get("PATH", "")

CSV_HEADER = [
    "video_path", "depth_rgb3_path", "geom_rgb_path", "human_bbox_path",
    "human_mesh_path", "smplx_param_path", "specify_ID_profile_path", "prompt",
]

_MIN_DURATION_S = 5.0  # inference needs 81 frames @ 16fps ≈ 5.06s


def _check_duration(path: str) -> float:
    import cv2
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    nfr = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    dur = nfr / fps if fps > 0 else 0.0
    if dur < _MIN_DURATION_S:
        raise SystemExit(
            f"[build_csv] ERROR: {path} duration {dur:.2f}s < {_MIN_DURATION_S}s; "
            f"inference requires >=81 frames @16fps. Aborting."
        )
    return dur


# ----------------------------- preprocess wrapper ---------------------------

def _find_cached_clip(seq_path: str, output_dir: str) -> str | None:
    """Return an existing clip dir for this seq with all key artifacts present.

    Layout produced by preprocess_video.py: <output_dir>/<seq_name>/clip_<tag>/
    with smplx_pred_params_all.npz, original_video.mp4, and
    control_signals/bboxes_cam_square.npy. We pick the lexicographically first
    clip that has all three.
    """
    import glob
    seq_name = os.path.splitext(os.path.basename(seq_path))[0]
    seq_root = os.path.join(os.path.abspath(output_dir), seq_name)
    if not os.path.isdir(seq_root):
        return None
    for clip_dir in sorted(glob.glob(os.path.join(seq_root, "clip_*"))):
        smplx = os.path.join(clip_dir, "smplx_pred_params_all.npz")
        ovid = os.path.join(clip_dir, "original_video.mp4")
        bbox = os.path.join(clip_dir, "control_signals", "bboxes_cam_square.npy")
        if all(os.path.exists(p) for p in (smplx, ovid, bbox)):
            return clip_dir
    return None


def _preprocess_one(seq_path: str, output_dir: str, run_dilate: bool,
                    extra_args: list[str] | None = None,
                    gpu_id: int | None = None) -> str:
    """Run preprocess_video.py + (optionally) make_scene_masked_dilate.py for one video.

    Returns the resulting clip dir (the first clip if multiple).
    Cached: if an existing clip dir under <output_dir>/<seq_name>/ already has
    smplx + original_video + bboxes, reuse it (also re-validate dilate output).
    """
    seq_path = os.path.abspath(seq_path)
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    cached = _find_cached_clip(seq_path, output_dir)
    if cached is not None:
        print(f"[build_csv] cache hit: {os.path.basename(seq_path)} → {cached}")
        clip_dir = cached
    else:
        env = os.environ.copy()
        if gpu_id is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        cmd = [
            sys.executable, "-u", os.path.join(_PREPROC_DIR, "preprocess_video.py"),
            "--seq_path", seq_path,
            "--output_dir", output_dir,
        ]
        if extra_args:
            cmd += extra_args
        print(f"[build_csv] $ CUDA_VISIBLE_DEVICES={env.get('CUDA_VISIBLE_DEVICES','-')} {' '.join(cmd)}")
        _t = time.time()
        p = subprocess.run(cmd, capture_output=True, text=True, env=env)
        print(f"[build_csv] preprocess_video.py finished in {time.time()-_t:.1f}s "
              f"(rc={p.returncode}) for {os.path.basename(seq_path)}")
        if p.returncode != 0:
            sys.stderr.write(p.stdout)
            sys.stderr.write(p.stderr)
            raise RuntimeError(f"preprocess_video.py failed for {seq_path}")
        sys.stdout.write(p.stdout)

        clip_dirs = [ln.split("=", 1)[1].strip()
                     for ln in p.stdout.splitlines() if ln.startswith("CLIP_DIR=")]
        if not clip_dirs:
            raise RuntimeError(f"no CLIP_DIR= emitted for {seq_path}")
        clip_dir = clip_dirs[0]

    if run_dilate:
        dilate_env = os.environ.copy()
        if gpu_id is not None:
            dilate_env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        # Skip dilate if its outputs already exist (e.g. masked_video.mp4 etc.)
        dilate_marker = os.path.join(clip_dir, "scene_masked_resize_dilate.mp4")
        if os.path.exists(dilate_marker):
            print(f"[build_csv] dilate cache hit: {dilate_marker}")
            return clip_dir
        cmd2 = [
            sys.executable, "-u", os.path.join(_PREPROC_DIR, "make_scene_masked_dilate.py"),
            "--clip_dir", clip_dir,
        ]
        print(f"[build_csv] $ {' '.join(cmd2)}")
        _t2 = time.time()
        p2 = subprocess.run(cmd2, capture_output=True, text=True, env=dilate_env)
        print(f"[build_csv] make_scene_masked_dilate.py finished in {time.time()-_t2:.1f}s "
              f"(rc={p2.returncode})")
        sys.stdout.write(p2.stdout)
        if p2.returncode != 0:
            sys.stderr.write(p2.stderr)
            raise RuntimeError(f"make_scene_masked_dilate.py failed for {clip_dir}")
    return clip_dir


def _pick_free_gpus(n: int, min_free_mib: int = 20000) -> list[int]:
    """Return up to n GPU IDs ordered by free memory (most free first).

    Honors CUDA_VISIBLE_DEVICES (returned IDs are local indices into that set,
    matching what subprocesses will see when we re-set CUDA_VISIBLE_DEVICES).
    Filters out GPUs with < min_free_mib free memory (default ~20GB).
    Falls back to [0]*n if nvidia-smi unavailable.
    """
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True,
        )
        rows = [ln.strip().split(",") for ln in out.stdout.strip().splitlines()]
        gpus = [(int(r[0]), int(r[1])) for r in rows]  # (phys_idx, free_mib)
    except Exception:
        return [0] * n

    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd:
        allowed = [int(x) for x in cvd.split(",") if x.strip()]
        # local index into CUDA_VISIBLE_DEVICES list
        local_map = {phys: local for local, phys in enumerate(allowed)}
        gpus = [(local_map[p], f) for p, f in gpus if p in local_map]
    free = [g for g in gpus if g[1] >= min_free_mib]
    free.sort(key=lambda x: -x[1])
    if len(free) < n:
        print(f"[build_csv] WARN: only {len(free)} GPU(s) with >={min_free_mib}MiB free; "
              f"need {n}. Will recycle.")
    if not free:
        return [0] * n
    ids = [g[0] for g in free]
    return [ids[i % len(ids)] for i in range(n)]


def _preprocess_parallel(jobs: list[tuple], max_workers: int = 2) -> list[str]:
    """jobs = [(seq_path, output_dir, run_dilate, extra_args), ...]

    Always parallel (never serial). 自动挑空闲 GPU 分配；卡少时按 i%N 复用。
    """
    if len(jobs) == 1:
        gpu_id = _pick_free_gpus(1)[0]
        seq, out, dil, extra = jobs[0]
        return [_preprocess_one(seq, out, dil, extra, gpu_id=gpu_id)]
    gpu_ids = _pick_free_gpus(len(jobs))
    print(f"[build_csv] parallel: {len(jobs)} jobs on GPUs {gpu_ids}")
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_preprocess_one, *j, gpu_id=gpu_ids[i])
                   for i, j in enumerate(jobs)]
        return [f.result() for f in futures]


# ---------------------------- path helpers ----------------------------------

def _emit(path: str | None) -> str:
    """Return absolute path (or empty string if None/empty)."""
    if not path:
        return ""
    return os.path.abspath(path)


def _build_paths(clip_dir: str) -> dict:
    """Standard product layout produced by preprocess_video.py."""
    cd = clip_dir
    return {
        "video":     os.path.join(cd, "original_video.mp4"),
        "depth":     os.path.join(cd, "depth_da3.mp4"),
        "geom":      os.path.join(cd, "scene_masked_resize_dilate.mp4"),  # 源动作 mask；仅 id_swap/scene_swap 用
        "geom_fill": os.path.join(cd, "rgb_fill_da3.mp4"),                 # 无人 inpaint 背景；motion_swap 用
        "bbox":      os.path.join(cd, "control_signals", "bboxes_cam_square.npy"),
        "mesh":      os.path.join(cd, "control_signals", "smplx_mesh_0_black.mp4"),
        "smplx":     os.path.join(cd, "smplx_pred_params_all.npz"),
    }


# --------------------------- CSV writer -------------------------------------

def _write_row(out_csv: str, append: bool, row: dict):
    is_new = (not append) or (not os.path.exists(out_csv))
    mode = "w" if is_new else "a"
    os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
    with open(out_csv, mode, newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if is_new:
            w.writerow(CSV_HEADER)
        w.writerow([row[k] for k in CSV_HEADER])
    print(f"[build_csv] wrote 1 row to {out_csv} (mode={mode})")


# ----------------------- id_profile auto-extract ----------------------------

_ID_PROFILE_FILES = ("ref1.png", "ref2.png", "ref3.png", "face.png")


def _extract_id_profile_from_clip(clip_dir: str, out_dir: str) -> str:
    """Auto-derive the 4 id-profile images (ref1/2/3.png + face.png) from a
    preprocessed clip dir produced by preprocess_video.py.

    - ref1/2/3 : 3 multi-angle full-body refs via SMPLX-orientation FPS
                 (uses get_three_ref_images_fullseq from datasets.oneshot_data_utils).
    - face     : heuristic head-region crop on the most-frontal frame
                 (no face detector required; uses body bbox top portion).

    Cached: skips if all 4 files already exist under out_dir.
    Returns out_dir.
    """
    os.makedirs(out_dir, exist_ok=True)
    if all(os.path.exists(os.path.join(out_dir, n)) for n in _ID_PROFILE_FILES):
        print(f"[build_csv] id_profile cache hit: {out_dir}")
        return out_dir

    import numpy as np
    import torch
    import decord
    from decord import VideoReader
    from PIL import Image as _PILImage

    # get_three_ref_images_fullseq assumes video_reader.get_batch() returns a
    # torch.Tensor (uses .permute). decord defaults to NDArray; flip the bridge.
    decord.bridge.set_bridge("torch")

    from datasets.oneshot_data_utils import select_ref_index_based_orientation

    video_p = os.path.join(clip_dir, "original_video.mp4")
    smplx_p = os.path.join(clip_dir, "smplx_pred_params_all.npz")
    cam_npz = os.path.join(clip_dir, "camera_and_humanmask.npz")
    for p in (video_p, smplx_p, cam_npz):
        if not os.path.exists(p):
            raise FileNotFoundError(f"[id_profile] missing required artifact: {p}")

    vr = VideoReader(video_p)
    H_gt, W_gt = int(vr[0].shape[0]), int(vr[0].shape[1])

    # --- Load DA3 human masks (aspect-preserved downscale of GT) ----------
    d = np.load(cam_npz, allow_pickle=True)
    mp = d["mask_packbits"]
    T_da, H_da, W_da = [int(v) for v in d["mask_shape"]]
    bitorder = str(d["bitorder"])
    sx, sy = W_gt / float(W_da), H_gt / float(H_da)
    # SMPLX params + camera (used by face-frame frontness selection and as
    # an optional disambiguation hint inside Haar cascade head detection).
    cm = d
    sd = np.load(smplx_p, allow_pickle=True)
    _det_cache = {"_init": False}  # lazy-init YuNet detector + per-frame results

    def _mask(idx):
        i = min(int(idx), T_da - 1)
        return np.unpackbits(mp[i], axis=-1, bitorder=bitorder)[:, :W_da].astype(bool)

    def _body_bbox_gt(idx, margin_w=0.05, margin_h=0.05):
        m = _mask(idx)
        ys, xs = np.where(m)
        if ys.size == 0:
            return None
        gx1, gx2 = float(xs.min()) * sx, float(xs.max() + 1) * sx
        gy1, gy2 = float(ys.min()) * sy, float(ys.max() + 1) * sy
        bw, bh = gx2 - gx1, gy2 - gy1
        gx1 -= bw * margin_w; gx2 += bw * margin_w
        gy1 -= bh * margin_h; gy2 += bh * margin_h
        return (
            max(0, int(round(gx1))), max(0, int(round(gy1))),
            min(W_gt, int(round(gx2))), min(H_gt, int(round(gy2))),
        )

    def _yunet_detect_all():
        """Run YuNet face detector on every frame once; cache best
        detection per frame. Returns dict idx → tuple
        (x, y, w, h, eye_r_xy, eye_l_xy, nose_xy, mouth_r_xy, mouth_l_xy, conf).

        YuNet is a modern (2023) ONNX face detector shipped through
        OpenCV's FaceDetectorYN API. It does NOT fire on hair / torso /
        clothing the way Haar cascades do, so we don't need any
        multi-detection disambiguation, mask-region filters or SMPLX
        frontness gates — keep the highest-confidence box per frame
        and rank across frames using landmark-based geometric scores.
        """
        import cv2 as _cv2
        if _det_cache.get("_init"):
            return _det_cache["per_frame"]
        onnx = os.path.join(_PREPROC_DIR, "checkpoints",
                            "face_detection_yunet_2023mar.onnx")
        if not os.path.exists(onnx):
            print(f"[id_profile] WARN: YuNet model not found at {onnx}")
            _det_cache.update({"_init": True, "per_frame": {}})
            return {}
        det = _cv2.FaceDetectorYN_create(
            onnx, "", (W_gt, H_gt),
            score_threshold=0.6, nms_threshold=0.3, top_k=5000,
        )
        per_frame = {}
        for i in range(min(len(vr), T_da)):
            fr = vr.get_batch([int(i)])
            if not isinstance(fr, torch.Tensor):
                fr = torch.from_numpy(
                    fr.asnumpy() if hasattr(fr, "asnumpy") else np.asarray(fr))
            arr = fr[0].numpy().astype(np.uint8)
            bgr = _cv2.cvtColor(arr, _cv2.COLOR_RGB2BGR)
            _, faces = det.detect(bgr)
            if faces is None or len(faces) == 0:
                continue
            # YuNet returns N×15:
            #   [x, y, w, h,
            #    eye_r_x, eye_r_y, eye_l_x, eye_l_y,
            #    nose_x,  nose_y,
            #    mouth_r_x, mouth_r_y, mouth_l_x, mouth_l_y,
            #    confidence]
            best = max(faces, key=lambda f: float(f[14]))
            per_frame[int(i)] = (
                float(best[0]), float(best[1]),
                float(best[2]), float(best[3]),
                (float(best[4]),  float(best[5])),   # eye_r
                (float(best[6]),  float(best[7])),   # eye_l
                (float(best[8]),  float(best[9])),   # nose
                (float(best[10]), float(best[11])),  # mouth_r
                (float(best[12]), float(best[13])),  # mouth_l
                float(best[14]),                      # conf
            )
        _det_cache.update({"_init": True, "per_frame": per_frame})
        print(f"[id_profile] YuNet face detector: "
              f"{len(per_frame)}/{min(len(vr), T_da)} frames have a face")
        return per_frame

    def _head_bbox_gt(idx, margin=0.40):
        """Head bbox in GT coords from cached YuNet detections, padded
        by `margin` on all sides. Returns None if no face on this frame.

        YuNet bboxes are tight to facial features (brow→chin), so we
        pad generously — `margin=0.40` adds ~40% of the bbox size on
        every side so the crop includes forehead / hair / chin /
        ears. Using `margin=0.0` recovers the raw detection size for
        area measurements.
        """
        per_frame = _yunet_detect_all()
        det = per_frame.get(int(idx))
        if det is None:
            return None
        x, y, w, h = det[:4]
        gx1 = x - margin * w; gx2 = x + w + margin * w
        gy1 = y - margin * h; gy2 = y + h + margin * h
        gx1 = max(0.0, min(float(W_gt), gx1))
        gy1 = max(0.0, min(float(H_gt), gy1))
        gx2 = max(0.0, min(float(W_gt), gx2))
        gy2 = max(0.0, min(float(H_gt), gy2))
        if gx2 - gx1 < 2 or gy2 - gy1 < 2:
            return None
        return (
            int(round(gx1)), int(round(gy1)),
            int(round(gx2)), int(round(gy2)),
        )

    def _save_native_crop(idx, bbox, out_path):
        """Crop GT frame at bbox (no resize, no padding). Saves natural-aspect."""
        x1, y1, x2, y2 = bbox
        fr = vr.get_batch([int(idx)])  # [1,H,W,3] uint8
        if not isinstance(fr, torch.Tensor):
            fr = torch.from_numpy(fr.asnumpy() if hasattr(fr, "asnumpy") else np.asarray(fr))
        arr = fr[0, y1:y2, x1:x2, :].numpy().astype(np.uint8)
        _PILImage.fromarray(arr).save(out_path)

    T_use = min(len(vr), T_da)
    idx_all = np.arange(T_use, dtype=np.int64)

    # --- Pick face frame: among top-area frames, prefer the most frontal
    # face with clearest features.
    #
    # Why a composite score (size + frontal-ness + confidence) instead of
    # plain max-area:
    #   - On 360° pans (circle_test.mp4) the largest YuNet detection can
    #     still be a 3/4-profile shot. Landmark geometry tells us if the
    #     view is frontal.
    #   - On natural videos people blink. YuNet detects closed-eye
    #     frames just fine (lower conf), but the resulting crop is bad
    #     for ID. The conf score and the eye-separation ratio both drop
    #     for blink/squint frames, so the score nudges away from them.
    #   - Stitched videos have many near-identical large-face frames; the
    #     composite tiebreaks deterministically without affecting which
    #     segment is chosen.
    per_frame = _yunet_detect_all()

    def _frontal_score(rec):
        """Return ~1.0 for an ideal frontal face with crisp landmarks,
        and lower (down to ~0) for profile / tilted / occluded faces.

        Three signals combined:
          - eye_sep:  |eye_l.x - eye_r.x| / face_w. Frontal ≈ 0.40,
            profile compresses below 0.20.
          - tilt:     |eye_l.y - eye_r.y| / face_h. Frontal ≈ 0,
            head tilt grows it.
          - nose_off: |nose.x - eye_mid.x| / face_w. Frontal ≈ 0,
            profile pushes nose off the eye midline.
        """
        x, y, w, h, er, el, no, _mr, _ml, conf = rec
        if w <= 0 or h <= 0:
            return 0.0
        eye_sep  = abs(el[0] - er[0]) / w
        tilt     = abs(el[1] - er[1]) / h
        eye_mid  = ((el[0] + er[0]) * 0.5)
        nose_off = abs(no[0] - eye_mid) / w
        # Each component clamped to [0,1] where 1 == ideal frontal.
        s_sep    = max(0.0, min(1.0, eye_sep / 0.40))
        s_tilt   = max(0.0, 1.0 - tilt * 4.0)
        s_nose   = max(0.0, 1.0 - nose_off * 5.0)
        return s_sep * s_tilt * s_nose

    if per_frame:
        # First narrow to "near-largest" pool (top 25% by raw bbox area,
        # min 1 candidate). This guarantees we never trade a clearly-big
        # face for a tiny-but-frontal one in the distance.
        items = list(per_frame.items())
        items.sort(key=lambda kv: -(kv[1][2] * kv[1][3]))
        n_top = max(1, len(items) // 4)
        top = items[:n_top]
        # Within the pool, score = frontal_score * confidence. Conf
        # captures landmark crispness and downweights blurry/squint/blink
        # frames in practice.
        f_idx, rec = max(
            top, key=lambda kv: _frontal_score(kv[1]) * kv[1][9]
        )
        fx, fy, fw, fh = rec[:4]
        f_area = int(fw * fh)
        f_front = _frontal_score(rec)
        f_conf = rec[9]
        print(f"[id_profile] face frame={f_idx} face_area={f_area} "
              f"frontal={f_front:.3f} conf={f_conf:.3f} "
              f"(n_yunet={len(per_frame)}/{T_use}, top_pool={n_top})")
    else:
        # No face detected anywhere — degenerate fallback to first frame.
        f_idx = int(idx_all[0])
        print(f"[id_profile] face frame={f_idx} (degenerate: YuNet found "
              f"no face in any of {T_use} frames)")

    # --- Pick 3 multi-angle ref indices via SMPLX FPS over all frames -----
    # Do NOT exclude the face frame: a frontal-large-face frame is often the
    # canonical view we'd want as a ref too. ref crop (body) vs face crop
    # (head) are different visuals; redundancy is mild and historically
    # intentional (old code hard-set f_idx = ref_global_idx[0]).
    ref_global_idx = select_ref_index_based_orientation(
        smplx_p, idx_all, k=3, select_fix_frame=0,
    )
    ref_global_idx = np.array(ref_global_idx, dtype=np.int64)
    while len(ref_global_idx) < 3:
        ref_global_idx = np.concatenate([ref_global_idx, ref_global_idx[-1:]])

    # --- ref1/2/3: tight body crop preserving natural aspect --------------
    for i in range(3):
        ridx = int(ref_global_idx[i])
        bb = _body_bbox_gt(ridx, margin_w=0.05, margin_h=0.03)
        if bb is None:
            # fallback: full frame
            bb = (0, 0, W_gt, H_gt)
        _save_native_crop(ridx, bb, os.path.join(out_dir, f"ref{i+1}.png"))

    # --- face: tight head crop on the chosen face frame -------------------
    # margin=0.40: pad ~40% on every side. YuNet bbox is tight to brow→
    # chin so without generous padding the forehead/hair gets clipped.
    fb = _head_bbox_gt(f_idx, margin=0.40)
    if fb is None:
        # fallback: top portion of body bbox as a square
        bb = _body_bbox_gt(f_idx) or (0, 0, W_gt, H_gt)
        side = min(bb[2] - bb[0], (bb[3] - bb[1]) // 4 or 1)
        fb = (bb[0], bb[1], bb[0] + side, bb[1] + side)
    _save_native_crop(f_idx, fb, os.path.join(out_dir, "face.png"))

    print(f"[build_csv] wrote 4 id-profile images to {out_dir} "
          f"(face heuristic from frame {f_idx})")
    return out_dir


def _id_profile_preproc_job(args):
    """If --id_profile_video given, return its preprocess job tuple; else None.

    Validates mutual exclusion with --id_profile_dir up-front so the failure
    happens before kicking off any GPU work.
    """
    if getattr(args, "id_profile_video", None):
        if args.id_profile_dir:
            raise SystemExit("[build_csv] --id_profile_video and --id_profile_dir are mutually exclusive.")
        _check_duration(args.id_profile_video)
        return (args.id_profile_video, args.preproc_out, False, args.extra)
    return None


def _finalize_id_profile_dir(args, id_clip: str | None) -> str | None:
    """Convert a preprocessed id_profile clip dir into the 4-image profile dir.

    - id_clip is None  → fall back to args.id_profile_dir (legacy/None).
    - id_clip provided → run _extract_id_profile_from_clip and return its out dir.
    """
    if id_clip is None:
        return args.id_profile_dir
    out_dir = os.path.join(id_clip, "id_profile")
    return _extract_id_profile_from_clip(id_clip, out_dir)


# ----------------------------- subcommands ----------------------------------

def cmd_id_swap(args):
    _check_duration(args.video_path)
    jobs = [(args.video_path, args.preproc_out, True, args.extra)]
    id_job = _id_profile_preproc_job(args)
    if id_job is not None:
        jobs.append(id_job)
    results = _preprocess_parallel(jobs)
    src_clip = results[0]
    id_clip = results[1] if id_job is not None else None
    src = _build_paths(src_clip)
    id_profile_dir = _finalize_id_profile_dir(args, id_clip)
    row = {
        "video_path":              _emit(src["video"]),
        "depth_rgb3_path":         _emit(src["depth"]),
        "geom_rgb_path":           _emit(src["geom"]),
        "human_bbox_path":         _emit(src["bbox"]),
        "human_mesh_path":         _emit(src["mesh"]),
        "smplx_param_path":        _emit(src["smplx"]),
        "specify_ID_profile_path": _emit(id_profile_dir),
        "prompt":                  args.prompt,
    }
    _write_row(args.out_csv, args.append, row)


def cmd_motion_swap(args):
    _check_duration(args.video_path)
    _check_duration(args.motion_video_path)
    # src 需要 dilate mask（geom_rgb_path 用 scene_masked_resize_dilate.mp4）；motion 不需要
    jobs = [
        (args.video_path,        args.preproc_out, True,  args.extra),
        (args.motion_video_path, args.preproc_out, False, args.extra),
    ]
    id_job = _id_profile_preproc_job(args)
    if id_job is not None:
        jobs.append(id_job)
    results = _preprocess_parallel(jobs)
    src_clip, mo_clip = results[0], results[1]
    id_clip = results[2] if id_job is not None else None
    src = _build_paths(src_clip)
    mo  = _build_paths(mo_clip)
    id_profile_dir = _finalize_id_profile_dir(args, id_clip)
    row = {
        "video_path":              _emit(src["video"]),
        "depth_rgb3_path":         _emit(src["depth"]),
        "geom_rgb_path":           _emit(src["geom"]),  # 源动作 dilated mask（实测效果优于 rgb_fill_da3）
        "human_bbox_path":         _emit(src["bbox"]),
        "human_mesh_path":         _emit(mo["mesh"]),
        "smplx_param_path":        _emit(mo["smplx"]),
        "specify_ID_profile_path": _emit(id_profile_dir),
        "prompt":                  args.prompt,
    }
    _write_row(args.out_csv, args.append, row)


def cmd_scene_swap(args):
    _check_duration(args.video_path)
    _check_duration(args.scene_video_path)
    jobs = [
        (args.video_path,       args.preproc_out, False, args.extra),
        (args.scene_video_path, args.preproc_out, True,  args.extra),
    ]
    id_job = _id_profile_preproc_job(args)
    if id_job is not None:
        jobs.append(id_job)
    results = _preprocess_parallel(jobs)
    src_clip, sc_clip = results[0], results[1]
    id_clip = results[2] if id_job is not None else None
    src = _build_paths(src_clip)
    sc  = _build_paths(sc_clip)
    id_profile_dir = _finalize_id_profile_dir(args, id_clip)
    row = {
        "video_path":              _emit(src["video"]),
        "depth_rgb3_path":         _emit(sc["depth"]),
        "geom_rgb_path":           _emit(sc["geom"]),
        "human_bbox_path":         _emit(src["bbox"]),
        "human_mesh_path":         _emit(src["mesh"]),
        "smplx_param_path":        _emit(src["smplx"]),
        "specify_ID_profile_path": _emit(id_profile_dir),
        "prompt":                  args.prompt,
    }
    _write_row(args.out_csv, args.append, row)


# ------------------------------ argparse ------------------------------------

def _common(p: argparse.ArgumentParser):
    p.add_argument("--video_path", required=True, help="Source video (provides ID/GT/anchor).")
    p.add_argument("--prompt", required=True)
    p.add_argument("--id_profile_dir", default=None,
                   help="(Legacy) Directory containing user-prepared ref1/ref2/ref3/face images. "
                        "Empty → inference 抽 ID from video. Mutually exclusive with --id_profile_video.")
    p.add_argument("--id_profile_video", default=None,
                   help="ID profile video (multi-angle short clip). build_csv 会对其跑一次 preprocess_video.py，"
                        "再用 SMPLX-orientation FPS 自动抽 3 张多视角 ref + 1 张启发式 face crop，"
                        "落盘到 <preproc_out>/<vid>/clip_*/id_profile/ 并填入 specify_ID_profile_path。"
                        "Mutually exclusive with --id_profile_dir.")
    p.add_argument("--out_csv", required=True)
    p.add_argument("--append", action="store_true",
                   help="Append to out_csv instead of overwrite.")
    p.add_argument("--infer_long", action="store_true",
                   help="Process the WHOLE video (multiple clips). Default: only the first 6s clip "
                        "(≈ first inference window) for fast turnaround.")
    p.add_argument("--preproc_out", default=None,
                   help="Where preprocess_video.py writes its outputs. Default: <out_csv>_dir/preproc/")
    p.add_argument("--extra", nargs=argparse.REMAINDER, default=[],
                   help="Forward remaining args to preprocess_video.py (e.g. --whole_video).")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("id_swap")
    _common(p1)
    p1.set_defaults(func=cmd_id_swap)

    p2 = sub.add_parser("motion_swap")
    _common(p2)
    p2.add_argument("--motion_video_path", required=True)
    p2.set_defaults(func=cmd_motion_swap)

    p3 = sub.add_parser("scene_swap")
    _common(p3)
    p3.add_argument("--scene_video_path", required=True)
    p3.set_defaults(func=cmd_scene_swap)

    args = ap.parse_args()
    if args.preproc_out is None:
        args.preproc_out = os.path.join(os.path.dirname(os.path.abspath(args.out_csv)), "preproc")

    # Default: first-clip-only for fast turnaround. --infer_long disables this.
    if not args.infer_long:
        args.extra = list(args.extra) + ["--max_clips", "1"]

    _t_total = time.time()
    print(f"[build_csv] === {args.cmd} START ===")
    args.func(args)
    print(f"[build_csv] === {args.cmd} DONE in {time.time()-_t_total:.1f}s ===")


if __name__ == "__main__":
    main()
