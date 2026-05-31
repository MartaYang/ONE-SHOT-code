#!/usr/bin/env python3
"""Generate scene_masked_resize_dilate.mp4 for one or more clip dirs.

Adapted from Human3R_DA3/05_make_scene_masked_dilate.py with two changes:
  1. Accept either --clip_dir (single) or --root (recursive glob like the original).
  2. Standalone — no other deps from Human3R_DA3.
"""
import argparse
import glob
import os

import cv2
import imageio.v2 as iio
import numpy as np


def load_mask(npz_path):
    d = np.load(npz_path, allow_pickle=True)
    packed = d["mask_packbits"]
    Ts, Hm, Wm = (int(x) for x in d["mask_shape"])
    bitorder = (
        str(d["bitorder"].tolist()) if isinstance(d["bitorder"], np.ndarray)
        else str(d["bitorder"])
    )
    unpacked = np.unpackbits(packed, axis=-1, bitorder=bitorder)
    return unpacked[..., :Wm].astype(np.uint8)


def get_video_hw(path):
    cap = cv2.VideoCapture(path)
    h = int(cap.get(4)); w = int(cap.get(3))
    cap.release()
    return h, w


def process_clip(clip_dir, kernel_radius, ref_name, out_name, overwrite, video_name="original_video.mp4"):
    video = os.path.join(clip_dir, video_name)
    npz = os.path.join(clip_dir, "camera_and_humanmask.npz")
    ref = os.path.join(clip_dir, ref_name)
    out = os.path.join(clip_dir, out_name)
    if not (os.path.exists(video) and os.path.exists(npz) and os.path.exists(ref)):
        return f"SKIP_MISSING: {clip_dir}"
    if os.path.exists(out) and not overwrite:
        return f"SKIP_EXIST: {out}"

    mask_thw = load_mask(npz)
    T_mask = mask_thw.shape[0]
    H_out, W_out = get_video_hw(ref)

    k = 2 * int(kernel_radius) + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 16
    writer = iio.get_writer(
        out, fps=fps, codec="libx264",
        ffmpeg_params=["-pix_fmt", "yuv420p", "-crf", "18", "-preset", "veryfast"],
    )
    last_m = None
    idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame.shape[0] != H_out or frame.shape[1] != W_out:
                frame = cv2.resize(frame, (W_out, H_out), interpolation=cv2.INTER_LINEAR)
            if idx < T_mask:
                m = cv2.resize(mask_thw[idx], (W_out, H_out), interpolation=cv2.INTER_NEAREST)
                m = cv2.dilate(m, kernel, iterations=1)
                last_m = m
            else:
                m = last_m if last_m is not None else np.zeros((H_out, W_out), np.uint8)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgb[m.astype(bool)] = 0
            writer.append_data(frame_rgb)
            idx += 1
    finally:
        cap.release(); writer.close()
    return f"OK: {out} (T={idx}, dil_r={kernel_radius}, {H_out}x{W_out})"


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--clip_dir", help="Single clip dir to process.")
    g.add_argument("--root", help="Walk root/*/clip_*/ recursively.")
    ap.add_argument("--kernel_radius", type=int, default=9)
    ap.add_argument("--ref", default="rgb_fill_da3.mp4")
    ap.add_argument("--out_name", default="scene_masked_resize_dilate.mp4")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    if args.clip_dir:
        clip_dirs = [args.clip_dir]
    else:
        clip_dirs = sorted(glob.glob(os.path.join(args.root, "*", "clip_*")))
    print(f"[INFO] {len(clip_dirs)} clips, dilate radius={args.kernel_radius}px, out={args.out_name}")
    for i, cd in enumerate(clip_dirs):
        try:
            msg = process_clip(cd, args.kernel_radius, args.ref, args.out_name, args.overwrite)
        except Exception as e:
            msg = f"FAIL: {cd}: {e!r}"
        print(f"[{i+1}/{len(clip_dirs)}] {msg}")


if __name__ == "__main__":
    main()
