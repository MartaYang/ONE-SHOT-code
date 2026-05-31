<div align="center">

# ONE-SHOT
## Compositional Human-Environment Video Synthesis via<br>Spatial-Decoupled Motion Injection and Hybrid Context Integration

Fengyuan Yang<sup>1,2</sup>&ensp;
Luying Huang<sup>2,†</sup>&ensp;
Jiazhi Guan<sup>2,✉</sup>&ensp;
Quanwei Yang<sup>2</sup>&ensp;
Dongwei Pan<sup>2</sup>&ensp;
Jianglin Fu<sup>2</sup>&ensp;
Haocheng Feng<sup>2</sup>&ensp;
Wei He<sup>2</sup>&ensp;
Kaisiyuan Wang<sup>2</sup>&ensp;
Hang Zhou<sup>2,✉</sup>&ensp;
Angela Yao<sup>1</sup>

<sup>1</sup>National University of Singapore&emsp;<sup>2</sup>Baidu, AMU

<sup>†</sup>Project leader&emsp;<sup>✉</sup>Corresponding authors&emsp;Work done during Fengyuan's internship at Baidu

[![arXiv](https://img.shields.io/badge/arXiv-2604.01043-b31b1b.svg)](https://arxiv.org/abs/2604.01043)
&nbsp;
[![Project Page](https://img.shields.io/badge/Project-Page-blue.svg)](https://martayang.github.io/ONE-SHOT/)

</div>

---

Recent advances in Video Foundation Models (VFMs) have revolutionized human-centric video synthesis, yet fine-grained and independent editing of subjects and scenes remains a critical challenge. We introduce **ONE-SHOT**, a parameter-efficient framework built upon pre-trained VFMs that achieves high-fidelity synthesis of human-environment videos with independent control over subject appearance, human dynamics, spatial environments, and camera trajectories. By optimizing only a sparse set of parameters, it achieves precise control while preserving responsiveness to textual instructions. A canonical-space motion injection mechanism mitigates conditioning competition between rigid human priors and text prompts. By anchoring static and dynamic context, ONE-SHOT ensures persistent subject identity and stable human-environment interactions across minute-scale generations. Extensive experiments demonstrate that ONE-SHOT significantly outperforms existing methods in structural control and creative diversity.

<p align="center">
  <img src="assets/fig_teaser.webp" width="90%">
</p>

---

## 1. Installation

```bash
git clone https://github.com/MartaYang/ONE-SHOT-code.git
cd ONE-SHOT-code
```

```bash
# 1. Create conda environment
conda create -n oneshot python=3.12 -y
conda activate oneshot

# 2. ffmpeg with libx264 (GPL build — required for video encoding)
conda install 'ffmpeg=*=*gpl*' -y

# 3. PyTorch 2.5.1 + CUDA 12.1
pip install --index-url https://download.pytorch.org/whl/cu121 \
    torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121

# 4. PyTorch3D (prebuilt wheel — no CUDA compilation needed)
pip install Preprocessing/third_party/wheels/pytorch3d-0.7.9-cp312-cp312-manylinux_2_31_x86_64.whl

# 5. Remaining dependencies
pip install -r requirements.txt
```

> **Shortcut:** `bash install.sh` runs all five steps above and prints a smoke test on completion.

---

## 2. Checkpoints

Download the pretrained weights from HuggingFace:

```bash
huggingface-cli download MartaYang007/ONE-SHOT-14B \
    --local-dir pretrained_models/ONESHOT-14B-diffusers
export ONESHOT_MODEL_DIR=pretrained_models/ONESHOT-14B-diffusers
```

<details>
<summary>Checkpoint directory layout</summary>

```
ONESHOT-14B-diffusers/
├── transformer/          # 14B video diffusion transformer
├── vae/
├── text_encoder/
├── tokenizer/
├── scheduler/
├── model_index.json
├── preprocess/           # preprocessing models
│   ├── human3r.pth           # multi-person SMPL-X estimation (Human3R)
│   ├── DA3NESTED-GIANT-LARGE-1.1/  # monocular depth (Depth-Anything-3)
│   ├── smpl_models/          # SMPL / SMPL-X body models
│   └── torch_hub/            # DINOv2 (offline cache)
└── demo/                 # demo input videos 
```
</details>

---

## 3. Quick Inference

Demo videos are included in the checkpoint download above, users are recommended to use their own videos as input.

| Task | Command |
|------|---------|
| **ID Swap**<br><sub>Replace the actor's identity</sub> | `bash scripts/run_pipeline.sh id_swap \` <br> `    --video_path $ONESHOT_MODEL_DIR/demo/walkinforest.mp4 \` <br> `    --id_profile_video $ONESHOT_MODEL_DIR/demo/WillSmith.mp4 \` <br> `    --prompt "A sunlit forest trail with dense green trees and soft natural light filtering through the leaves. Will Smith, wearing a black suit, walks steadily along the forest path while holding a wooden walking stick, looking slightly upward as he move forward."` <br><br> `# --id_profile_video: recommended — video with multi-angle coverage of the target person` <br> `#   (front + side + other angles) for best identity fidelity.` <br> `# Alternative: --id_profile_dir <dir> with 4 images named exactly:` <br> `#   ref1.png (front view)  ref2.png (back / 3/4 view)  ref3.png (side view)  face.png (face crop)` |
| **Motion Swap**<br><sub>Apply a new motion to the original video</sub> | `bash scripts/run_pipeline.sh motion_swap \` <br> `    --video_path $ONESHOT_MODEL_DIR/demo/museum4_human.mp4 \` <br> `    --motion_video_path $ONESHOT_MODEL_DIR/demo/taiji.mp4 \` <br> `    --prompt "An indoor space resembling the interior of a museum. A man in a suit is performing tai chi movements."` <br><br> `# to also swap identity: add --id_profile_video $ONESHOT_MODEL_DIR/demo/WillSmith.mp4` <br> `# Note: update the prompt accordingly (e.g., gender, name) to match the new identity and avoid conflicts with the video content.` |
| **Scene Swap**<br><sub>Drop the person into a new background</sub> | `bash scripts/run_pipeline.sh scene_swap \` <br> `    --video_path $ONESHOT_MODEL_DIR/demo/palace_human.mp4 \` <br> `    --scene_video_path $ONESHOT_MODEL_DIR/demo/museum4_scene.mp4 \` <br> `    --id_profile_video $ONESHOT_MODEL_DIR/demo/WillSmith.mp4 \` <br> `    --prompt "An indoor space resembling the interior of a museum. Will Smith is walking, wearing a black suit."` <br><br> `# to also swap identity: add --id_profile_video $ONESHOT_MODEL_DIR/demo/WillSmith.mp4` <br> `# Note: update the prompt accordingly (e.g., gender, name) to match the new identity and avoid conflicts with the video content.` |

Outputs are saved to `exp/<scheduler>_<task>_<timestamp>/<save_name>.mp4`, e.g.:<br>
`exp/lcm_id_swap_20260527_221947/ID_WillSmith-SMPLX_clip_000-006-Scene_C01_gen_xxx_ourGen81.mp4`

---

## TODO

- [ ] **Multi-GPU inference** — support distributed single-sample inference with FSDP-based model sharding and sequence parallelism to reduce per-GPU memory usage for high-resolution generation
- [ ] **ComfyUI support** — node-based workflow for easier experimentation and demo
- [ ] **Fully compositional generation** — freely specify any combination of identity, motion, and scene sources, with explicit control over the person's position in the frame

---

## Repository Layout

```
ONE-SHOT/
├── install.sh                 # one-shot conda env installer
├── requirements.txt
├── tools/
│   ├── inference_short.py     # short-video inference entry point
│   └── inference_long.py      # long-video inference (scene memory)
├── scripts/
│   ├── run_pipeline.sh        # end-to-end: preprocessing + inference
│   ├── inference_short.sh
│   └── inference_long.sh
├── Preprocessing/
│   ├── preprocess_video.py    # per-video preprocessing
│   ├── build_csv.py           # build inference CSV from task arguments
│   ├── make_scene_masked_dilate.py
│   └── third_party/           # vendored dust3r / Human3R / CroCo + utilities
├── oneshot_diffusers/         # ONE-SHOT override package on top of diffusers
│   ├── transformer_wan_oneshot.py
│   ├── pipeline_wan_oneshot.py
│   └── oneshot_util.py
├── utils/                     # ODE solvers, video I/O helpers
└── datasets/                  # DWPose drawing, data utilities
```

---

## Acknowledgments

This project builds on:
[Wan2.1](https://github.com/Wan-Video/Wan2.1) · [Human3R](https://github.com/fanegg/Human3R) · [DUSt3R](https://github.com/naver/dust3r) · [Depth-Anything-3](https://github.com/DepthAnything/Depth-Anything-V2) · [SMPL-X](https://smpl-x.is.tue.mpg.de/) · [PyTorch3D](https://github.com/facebookresearch/pytorch3d) · [Diffusers](https://github.com/huggingface/diffusers)

---

## Citation

```bibtex
@misc{yang2026oneshot,
  title={ONE-SHOT: Compositional Human-Environment Video Synthesis via Spatial-Decoupled Motion Injection and Hybrid Context Integration},
  author={Fengyuan Yang and Luying Huang and Jiazhi Guan and Quanwei Yang and Dongwei Pan and Jianglin Fu and Haocheng Feng and Wei He and Kaisiyuan Wang and Hang Zhou and Angela Yao},
  year={2026},
  eprint={2604.01043},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2604.01043}
}
```
