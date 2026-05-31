# ONE-SHOT / Preprocessing

Preprocesses source video(s) into a CSV consumed by `tools/inference_short.py` for three tasks:

| Task | What changes | What stays |
|---|---|---|
| `id_swap` | Identity reference | Motion, scene from source video |
| `motion_swap` | SMPL-X motion | Appearance, scene, identity from source video |
| `scene_swap` | Background / camera trajectory | Appearance, motion, identity from source video |

## Checkpoint Paths

All large assets live under `$ONESHOT_MODEL_DIR/preprocess/` (~14 GB total):

| Asset | Size | Path |
|---|---|---|
| Human3R | 4.4 GB | `preprocess/human3r.pth` |
| Depth-Anything-3 | 6.3 GB | `preprocess/DA3NESTED-GIANT-LARGE-1.1/` |
| SMPL/SMPLX body models | 3.2 GB | `preprocess/smpl_models/` |
| DINOv2 hub cache | ~5 MB | `preprocess/torch_hub/facebookresearch_dinov2_main/` |

These are included in the `ONESHOT-14B-diffusers` HuggingFace download — no separate download needed.

## Usage

All commands should be run from the `Preprocessing/` directory with the `oneshot` conda env active.

> **Naming constraint:** `--video_path`, `--motion_video_path`, and `--scene_video_path` must have distinct basenames (stems), since the preproc output directory is named after the video stem. Reusing the same video for multiple roles is fine.

### id\_swap

```bash
python build_csv.py id_swap \
    --video_path $ONESHOT_MODEL_DIR/demo/C01.mp4 \
    --id_profile_video $ONESHOT_MODEL_DIR/demo/WillSmith.mp4 \
    --prompt "A cozy indoor living space with clean white walls and warm round pendant lights. Will Smith, wearing a black suit, faces the camera and speaks naturally." \
    --out_csv ./tmpout/out.csv
```

`--id_profile_video` auto-extracts 4 reference images from the video. Alternatively, use `--id_profile_dir <dir>` with manually prepared images named `ref1.png` (front), `ref2.png` (back/3-4 view), `ref3.png` (side), `face.png` (face crop).

### motion\_swap

```bash
python build_csv.py motion_swap \
    --video_path $ONESHOT_MODEL_DIR/demo/museum4_human.mp4 \
    --motion_video_path $ONESHOT_MODEL_DIR/demo/taiji.mp4 \
    --prompt "An indoor space resembling a museum interior. A man in a suit is performing tai chi movements." \
    --out_csv ./tmpout/out_motion_swap.csv
    # To also swap identity: add --id_profile_video $ONESHOT_MODEL_DIR/demo/WillSmith.mp4
```

### scene\_swap

```bash
python build_csv.py scene_swap \
    --video_path $ONESHOT_MODEL_DIR/demo/palace_human.mp4 \
    --scene_video_path $ONESHOT_MODEL_DIR/demo/museum4_scene.mp4 \
    --prompt "An indoor space resembling a museum interior. A man in a suit is walking." \
    --out_csv ./tmpout/out_scene_swap.csv
    # To also swap identity: add --id_profile_video $ONESHOT_MODEL_DIR/demo/WillSmith.mp4
```

## Output Structure

### Preprocessing intermediates

Each input video produces a `tmpout/preproc/<stem>/clip_<start>-<end>/` directory:

| File | Description |
|---|---|
| `original_video.mp4` | Resized source video |
| `depth_da3.mp4` | Depth map (DA3) |
| `scene_masked_resize_dilate.mp4` | Scene with dilated human mask |
| `smplx_pred_params_all.npz` | Per-frame SMPL-X parameters |
| `camera_and_humanmask.npz` | Camera intrinsics + human masks |
| `control_signals/bboxes_cam_square.npy` | Per-frame bounding boxes |
| `control_signals/smplx_mesh_0_black.mp4` | SMPL-X mesh render |

### CSV

- Via `build_csv.py`: written to `--out_csv`
- Via `scripts/run_pipeline.sh`: auto-written to `Preprocessing/tmpout/<task>_<timestamp>.csv`

All paths in the CSV are absolute.

## Video Length

- **< 5s**: Supported but inference will pad to 81 frames (quality may degrade).
- **> 6s (default)**: Only the first 6s clip is processed (`--max_clips 1`).
- **`--infer_long`**: Process all clips for long-video inference.

## End-to-End Pipeline

The recommended way is to use `scripts/run_pipeline.sh` from the repo root, which runs `build_csv.py` followed by `inference_short.sh` automatically. See the top-level `README.en.md` for full commands.

To run inference manually on a generated CSV:

```bash
CSV_NAME=/path/to/your.csv \
NPROC_PER_NODE=2 \
CUDA_VISIBLE_DEVICES=0,1 \
bash scripts/inference_short.sh
```
