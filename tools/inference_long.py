"""/usr/bin/env python3"""

import os
import logging
import random
import sys
import argparse
import torch
import json
import numpy as np
from diffusers import (
    AutoencoderKLWan,
    FlowMatchEulerDiscreteScheduler,
)
from oneshot_diffusers import (
    WanOneshotPipeline,
    WanOneshotTransformer3DModel,
)
from utils.fm_solvers_diffsynth import FlowMatchScheduler
from utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from utils.fm_solvers_dpmpp import FlowDPMSolverMultistepScheduler
from utils.fm_solvers_lcm import FlowMatchLCMScheduler
from diffusers.utils.torch_utils import randn_tensor

import decord
decord.bridge.set_bridge('torch')
from pathlib import Path
import torchvision.transforms.functional as tvF
from torchvision import transforms
import torch.distributed as dist


import cv2
import copy
import csv
import tempfile
import traceback
import time

def _init_logging(rank):
    # logging
    # set format
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        handlers=[logging.StreamHandler(stream=sys.stdout)])


def get_all_resolution(image_size):
    all_resolution = []
    divided_by = 16
    min_edge = int(image_size / 1.4)
    max_edge = int(image_size * 1.4)
    token_number = image_size * image_size / divided_by / divided_by
    for i in range(min_edge // divided_by, max_edge // divided_by + 1):
        all_resolution.append([i * divided_by, int(token_number // i * divided_by)])
    return all_resolution


def get_keep_crop_dict(video_frames, image_size):
    '''
    crop  by nearest ratio 
    '''
    all_res = get_all_resolution(image_size)
    N,C,H,W = video_frames.shape
    new_H = all_res[0][0]
    new_W = all_res[0][1]
    for i in range(len(all_res)):
        if H / W >= all_res[i][0] / all_res[i][1]:
            new_H = all_res[i][0]
            new_W = all_res[i][1]
    if H / W != new_H / new_W:
        crop_H = int(new_H / new_W * W)
        up = (H - crop_H)
    else:
        up = None
        crop_H = None
    return {
        "up": 0,
        "crop_H": crop_H,
        "left": 0,
        "crop_W": W,
        "new_H": new_H,
        "new_W": new_W,
    }

def get_data_idx_motionvideo(video_reader, max_num_frames, motion_frame_num=1, clip_first_frame_idx=9):
    motion_frames = 1 + 8 + 64 #73 motion frames are always 73 during training
    orig_fps = float(video_reader.get_avg_fps())
    video_num_frames = len(video_reader)
    duration = video_num_frames / orig_fps
    # number of frames for this video at 16FPS
    num_frames_16fps = int(round(duration * 16))
    if num_frames_16fps <= (max_num_frames + 9):     # fewer than 81 video frames + 9 motion frames = 90
        idx_16 = np.arange(num_frames_16fps, dtype=np.float32)
        t = idx_16 / 16

        # GT video starts from frame 1
        idx_gt = (t * orig_fps).astype(np.int64)[1:1+max_num_frames]
        idx_gt = np.clip(idx_gt, 0, video_num_frames - 1).tolist()
        # GT video is short: pad to max_num_frames (repeat last frame)
        pad_len = max_num_frames - len(idx_gt)
        if pad_len > 0:
            idx_gt += [idx_gt[-1]] * pad_len
        
        # motion video: repeat frame 0 to length 1+8+64=73
        idx_motionvideo = (t * orig_fps).astype(np.int64)[0]
        idx_motionvideo = np.clip(idx_motionvideo, 0, video_num_frames - 1).tolist()
        idx_motionvideo = [idx_motionvideo] * motion_frames
    else:
        start_16 = clip_first_frame_idx # start from frame 9
        # GT video: take 81 frames starting from frame start_16
        idx_16 = np.arange(start_16, start_16 + max_num_frames, dtype=np.float32)
        t = idx_16 / 16
        idx_gt = (t * orig_fps).astype(np.int64)
        idx_gt = np.clip(idx_gt, 0, video_num_frames - 1).tolist()

        assert motion_frame_num == 9 # first clip always uses 9 frames as motion frames; subsequent clips use previously generated frames
        idx_16 = np.arange(start_16 - motion_frame_num, start_16, dtype=np.float32)
        t = idx_16 / 16
        idx_motionvideo = (t * orig_fps).astype(np.int64)
        idx_motionvideo = np.clip(idx_motionvideo, 0, video_num_frames - 1).tolist()

        pad_len = motion_frames - len(idx_motionvideo) 
        idx_motionvideo = [idx_motionvideo[0]] * pad_len + idx_motionvideo  # repeat first frame of motion video
        
    return idx_gt, idx_motionvideo

def preprocess_video(video_path, depth_rgb3_path, geom_rgb_path, human_mesh_path, bbox_path, smplx_path, is_crop_bbox=False, test_image_size=640, clip_first_frame_idx=0, num_frames=81, specify_ID_profile_path=None, scene_memory_bank=None):
    from datasets.oneshot_data_utils import take_bboxes_by_idx, scale_bboxes_xyxy, crop_resize_by_bboxes_xyxy, overlay_green_box, generate_human_pos_maps, select_ref_index_based_orientation, smooth_bboxes_breathing, get_static_center_crop_bboxes, get_three_ref_images_fullseq, get_face_ref_image_fullseq, _find_img, _load_profile_as_tensor_letterbox
    if video_path is not None:

        if isinstance(video_path, str):
                video_path = Path(video_path)
        if isinstance(depth_rgb3_path, str):
            depth_rgb3_path = Path(depth_rgb3_path)
        if isinstance(geom_rgb_path, str):
            geom_rgb_path = Path(geom_rgb_path)
        if isinstance(human_mesh_path, str):
            human_mesh_path = Path(human_mesh_path)
        
        # read source video
        video_reader = decord.VideoReader(uri=video_path.as_posix())
        video_num_frames = len(video_reader)

        ### confirm data frame indices ###


        idx_motionvideo = None
        indices, idx_motionvideo = get_data_idx_motionvideo(video_reader, num_frames, motion_frame_num=9, clip_first_frame_idx=clip_first_frame_idx)
        
        ### read video by index ###
        # GT video
        frames = video_reader.get_batch(indices)
        frames = frames.permute(0, 3, 1, 2).contiguous()

        # read depth video
        if depth_rgb3_path is not None:
            depth_rgb_video_reader = decord.VideoReader(uri=depth_rgb3_path.as_posix())
            frames_depth_rgb = depth_rgb_video_reader.get_batch(indices)
            ref_pose_frames = frames_depth_rgb.permute(0,3,1,2).contiguous().float()
        # read layout video -> geom_rgb_path
        if geom_rgb_path is not None:
            geom_rgb_video_reader = decord.VideoReader(uri=geom_rgb_path.as_posix())
            frames_geom_rgb = geom_rgb_video_reader.get_batch(indices)
            layout_frames   = frames_geom_rgb.permute(0,3,1,2).contiguous().float()
        
        # local human mesh video
        if human_mesh_path is not None:
            human_mesh_video_reader = decord.VideoReader(uri=human_mesh_path.as_posix())
            human_mesh_frames = human_mesh_video_reader.get_batch(indices)
            human_mesh_frames = human_mesh_frames.permute(0, 3, 1, 2).contiguous().float()

        if idx_motionvideo is not None:
            motion_video = video_reader.get_batch(idx_motionvideo)
            motion_video = motion_video.permute(0,3,1,2).contiguous().float() # [N*3*H*W]
        else:
            motion_video = None

        H_gt,  W_gt  = frames.shape[-2], frames.shape[-1]
        H_rp,  W_rp  = ref_pose_frames.shape[-2], ref_pose_frames.shape[-1]

        inpainting_frames = torch.zeros_like(frames)
        inpainting_masks  = torch.ones_like(frames)

        _, _, H, W = frames.shape
        res_list = get_all_resolution(test_image_size)
        new_H = res_list[0][0]
        new_W = res_list[0][1]
        for i in range(len(res_list)):
            if H / W >= res_list[i][0] / res_list[i][1]:
                new_H = res_list[i][0]
                new_W = res_list[i][1]
        target_size = [new_H, new_W]
        print(f"[ONESHOT-RGBD test] target_size is {target_size}!!!\n")

        # === try loading bbox file (consistent with training rules) ===
        if bbox_path is not None and os.path.exists(bbox_path) and is_crop_bbox:
            
            bboxes_for_preprocess = np.load(bbox_path, allow_pickle=True)  # numpy: [F,4] or [F,5]
            bboxes_for_preprocess = np.array([bbox[0] if 0 in bbox else [-1, -1, -1, -1] for bbox in bboxes_for_preprocess])
            bboxes_for_preprocess = smooth_bboxes_breathing(
                                        bboxes_for_preprocess,
                                        min_valid_ratio=0.0,
                                        max_gap_interp=30,
                                        smooth_win=7,
                                        ema_alpha=0.7,
                                    )
            bboxes_cam_square = take_bboxes_by_idx(bboxes_for_preprocess, indices)

            # bbox enlarged 1.3x at ref_pose/layout resolution
            b_rp_orig = torch.tensor(bboxes_cam_square, dtype=torch.float32)
            b_rp, b_rp_orig_relative = get_static_center_crop_bboxes(b_rp_orig, H_rp, W_rp, wh_ratio=new_W/new_H)

            # map bbox from render resolution back to GT resolution for cropping
            b_gt = scale_bboxes_xyxy(b_rp, src_size=(H_rp, W_rp), dst_size=(H_gt, W_gt))

            H_crop, W_crop = b_rp[0,3] - b_rp[0,1], b_rp[0,2] - b_rp[0,0] # height and width of the final crop bbox
            # crop bbox must be the same size for the entire clip
            assert abs(H_crop - (b_rp[-1,3] - b_rp[-1,1]))<1e-4
            assert abs(W_crop - (b_rp[-1,2] - b_rp[-1,0]))<1e-4

            # scale human bbox to the new resolution
            b_rp_orig_relative = b_rp_orig_relative * torch.tensor([new_W/W_crop, new_H/H_crop, new_W/W_crop, new_H/H_crop])
            _, _, human_h_pos, human_w_pos = generate_human_pos_maps(b_rp_orig_relative, new_H, new_W, [4,16,16])

            # overlay colored box on original bbox for visualization
            overlay_green_box(ref_pose_frames, b_rp_orig)   # depth / ref pose
            overlay_green_box(layout_frames,   b_rp_orig)   # layout


            # 1) GT: map bbox from (H_rp,W_rp) to (H_gt,W_gt) and crop
            video_frames = crop_resize_by_bboxes_xyxy(frames, b_gt, target_size=target_size)
            if idx_motionvideo is not None:
                motion_video = crop_resize_by_bboxes_xyxy(motion_video, b_gt, target_size=target_size)
            # improved: select 3 frames with max orientation difference based on SMPLX orientation (covers diverse viewpoints)

            if specify_ID_profile_path is not None:
                profile_dir = str(specify_ID_profile_path)
                ref_paths = [
                    _find_img(profile_dir, "ref1"),
                    _find_img(profile_dir, "ref2"),
                    _find_img(profile_dir, "ref3"),
                ]
                face_path = _find_img(profile_dir, "face")

                ref_list = [_load_profile_as_tensor_letterbox(p, target_size, device="cpu", dtype=torch.float32) for p in ref_paths]
                ref_image = torch.cat(ref_list, dim=0)  # [3,3,Ht,Wt] OK
                face_ref  = _load_profile_as_tensor_letterbox(face_path, target_size, device="cpu", dtype=torch.float32)
            else:
                # extract full-body reference frames
                ref_image, three_angle_ref_global_idx, _ = get_three_ref_images_fullseq(
                        video_reader=video_reader,
                        bboxes_cam_square_full=bboxes_for_preprocess,
                        smplx_path=smplx_path,
                        H_rp=H_rp, W_rp=W_rp,
                        cur_resolution=target_size,
                        k=3,
                        select_fix_frame = 0,
                        ref_crop_human_out=True,
                    )
                # face reference frame
                face_ref, face_idx = get_face_ref_image_fullseq(
                        video_reader=video_reader,
                        video_path=video_path,
                        cur_resolution=target_size,
                        drop_prob=0.0,
                        min_face_pts=20,
                        topk=1,
                        enlarge_scale=1.25,
                        max_nose_offset_ratio=0.14,
                        max_roll_deg=10.0,
                        max_lr_vis_ratio=1.6,
                    )
            # background reference frames (naive_lastvideo: zeros for first clip, prev output for subsequent clips)
            if scene_memory_bank is None:
                bg_ref = torch.zeros((5, 3, target_size[0], target_size[1]), dtype=ref_image.dtype, device=ref_image.device)
            else:
                bg_ref = scene_memory_bank.to(ref_image.device, dtype=ref_image.dtype)
                assert bg_ref.shape[:2] == (5, 3), f"bg_ref shape wrong: {bg_ref.shape}"
                assert bg_ref.shape[-2:] == (target_size[0], target_size[1]), \
                    f"bg_ref HW mismatch: {bg_ref.shape[-2:]} vs {target_size}"
            ref_image = torch.cat([ref_image, bg_ref], dim=0)
            ref_image = torch.cat([ref_image,face_ref.to(ref_image.device)], dim=0)  # face ref appended at the end
            ref_human_img = ref_image

            # 2) ref_pose/layout: crop at its own resolution
            ref_pose_frames = crop_resize_by_bboxes_xyxy(ref_pose_frames, b_rp, target_size=target_size)
            layout_frames   = crop_resize_by_bboxes_xyxy(layout_frames,   b_rp, target_size=target_size)

            if human_mesh_path is not None and 'camera' in str(human_mesh_path):
                human_mesh_frames  = crop_resize_by_bboxes_xyxy(human_mesh_frames,   b_rp, target_size=target_size)

            # 4) inpainting_*: align with GT, crop using b_gt
            inpainting_frames = crop_resize_by_bboxes_xyxy(inpainting_frames, b_gt, target_size=target_size)
            inpainting_masks  = crop_resize_by_bboxes_xyxy(inpainting_masks,  b_gt, target_size=target_size)
            crop_info = {
                "pre_crop_box": None,
                "crop_info": {"up": 0, "left": 0, "crop_H": target_size[0], "crop_W": target_size[1], "new_H": target_size[0], "new_W": target_size[1]},
                "ori_video_shape": video_frames.shape[1:]
            }
        else: 
            print('error bbox_path:', bbox_path)
            raise NotImplementedError

        # 3) human_mesh: resize only, no crop
        new_H = res_list[0][0]
        new_W = res_list[0][1]
        for i in range(len(res_list)):
            if 512 / 512 >= res_list[i][0] / res_list[i][1]:
                new_H = res_list[i][0]
                new_W = res_list[i][1]
            target_size = [new_H, new_W]
        human_mesh_frames = torch.nn.functional.interpolate(
            human_mesh_frames, size=target_size, mode="bilinear", #bicubic",
            align_corners=True, antialias=True
        )

    else:
        video_frames = None
        ref_image = None
        ref_human_img = None
        inpainting_frames = None
        inpainting_masks = None
        ref_pose_frames = None
        layout_frames = None
        human_mesh_frames = None
        crop_info = None
    return video_frames, motion_video, ref_image, ref_human_img, inpainting_frames, inpainting_masks, ref_pose_frames, layout_frames, human_mesh_frames, crop_info, b_rp_orig_relative, human_h_pos, human_w_pos,


frame_transforms = transforms.Compose([transforms.Lambda(lambda x: x / 255.0 * 2.0 - 1.0)])
def video_transform(frames: torch.Tensor) -> torch.Tensor:
    return torch.stack([frame_transforms(f) for f in frames], dim=0)


def generate_input(data_dict, prompt_dict, is_crop_bbox=False, test_image_size=640, clip_first_frame_idx=0, num_frames=81, scene_memory_bank=None):
    data_root = args.train_data_root
    video, depth_rgb3_path, geom_rgb_path, bbox_path, human_mesh_path, smplx_param_path, prompt, specify_ID_profile_path = (
        data_dict["video_path"],
        data_dict["depth_rgb3_path"],
        data_dict["geom_rgb_path"],
        data_dict["human_bbox_path"],
        data_dict["human_mesh_path"],
        data_dict["smplx_param_path"],
        data_dict["prompt"],
        data_dict["specify_ID_profile_path"],
    )
    assert specify_ID_profile_path is not None
    
    video = os.path.join(data_root, video)
    depth_rgb3_path = os.path.join(data_root, depth_rgb3_path) if depth_rgb3_path else None
    geom_rgb_path = os.path.join(data_root, geom_rgb_path) if geom_rgb_path else None
    bbox_path = os.path.join(data_root, bbox_path) if bbox_path else None
    smplx_param_path = os.path.join(data_root, smplx_param_path) if smplx_param_path else None
    human_mesh_path = os.path.join(data_root, human_mesh_path) if human_mesh_path else None
    specify_ID_profile_path = os.path.join(data_root, specify_ID_profile_path) if specify_ID_profile_path else None
    if human_mesh_path is None: # pure scene (no human)
        print(f"[SceneOnly] no human mesh found at {human_mesh_path}. Using scene only.")
        human_mesh_path = None
        bbox_path = None
        smplx_param_path = None

    print(prompt)

    frames, motion_video, image, ref_human_img, inpainting_frames, inpainting_masks, ref_pose_frames, layout_frames, human_mesh_frames, crop_info, bboxes_cam_square, human_h_pos, human_w_pos = preprocess_video(
        video,
        depth_rgb3_path,
        geom_rgb_path,
        human_mesh_path,
        bbox_path,
        smplx_param_path,
        is_crop_bbox,
        test_image_size,
        clip_first_frame_idx,
        num_frames,
        specify_ID_profile_path=specify_ID_profile_path,
        scene_memory_bank=scene_memory_bank,
    )

    # prevent reshape side effects
    inpainting_masks[inpainting_masks > 0.] = 1
    # shape of image: [C, H, W]
    if inpainting_frames is None:
        inpainting_frames = copy.deepcopy(frames)
        inpainting_frames[inpainting_masks == 0.] = 255 // 2
    # replace ref img

    # process object image
    # assume object is never larger than the original video, pad accordingly
    assert ref_human_img.shape == image.shape
    ref_human_img_padding = torch.zeros_like(image)
    assert ref_human_img.shape == ref_human_img_padding.shape
    ref_human_img_padding[..., :ref_human_img.shape[-2], :ref_human_img.shape[-1]] = ref_human_img
    ref_human_img = ref_human_img_padding

    frames = video_transform(frames.float())
    inpainting_frames = video_transform(inpainting_frames.float())
    ref_pose_frames = video_transform(ref_pose_frames.float())
    layout_frames = video_transform(layout_frames.float())

    # Convert to [B, C, F, H, W]
    frames = frames.unsqueeze(0)
    frames = frames.permute(0, 2, 1, 3, 4).contiguous()
    inpainting_frames = inpainting_frames.unsqueeze(0)
    inpainting_frames = inpainting_frames.permute(0, 2, 1, 3, 4).contiguous()
    ref_pose_frames = ref_pose_frames.unsqueeze(0)
    ref_pose_frames = ref_pose_frames.permute(0, 2, 1, 3, 4).contiguous()
    layout_frames = layout_frames.unsqueeze(0)
    layout_frames = layout_frames.permute(0, 2, 1, 3, 4).contiguous()
    inpainting_masks = inpainting_masks.unsqueeze(0)
    inpainting_masks = inpainting_masks.permute(0, 2, 1, 3, 4).contiguous()
    if human_mesh_frames is not None:
        human_mesh_frames = video_transform(human_mesh_frames.float())
        human_mesh_frames = human_mesh_frames.unsqueeze(0)
        human_mesh_frames = human_mesh_frames.permute(0, 2, 1, 3, 4).contiguous()
    else:
        human_mesh_frames = None
    if motion_video is not None:
        motion_video = video_transform(motion_video.float())
        motion_video = motion_video.unsqueeze(0)
        motion_video = motion_video.permute(0, 2, 1, 3, 4).contiguous()

    assert frames.shape == inpainting_frames.shape == ref_pose_frames.shape == layout_frames.shape == inpainting_masks.shape, f"shape not match {frames.shape}, {inpainting_frames.shape}, {ref_pose_frames.shape}, {layout_frames.shape}, {inpainting_masks.shape}"


    return frames, image, ref_human_img, inpainting_frames, inpainting_masks, ref_pose_frames, layout_frames, human_mesh_frames, prompt, crop_info, bboxes_cam_square, motion_video, human_h_pos, human_w_pos

def export_single_video_only(output, prefix, save_name):
    """
    Save output video only, no concatenation.
    output: numpy array, shape [F, H, W, C]
    """
    import torch
    import logging
    from utils.video_io import save_video

    # 1. convert numpy to torch tensor [F, H, W, C]
    # assumes output is float in 0~1 or uint8 in 0~255
    output_tensor = torch.from_numpy(output).float()
    
    # normalize to 0~1 if values are in 0~255
    if output_tensor.max() > 1.5:
        output_tensor = output_tensor / 255.0

    # 2. rearrange dims to [1, C, F, H, W] as expected by save_video
    # original order: [F, H, W, C] -> permute -> [C, F, H, W] -> unsqueeze -> [1, C, F, H, W]
    vis = output_tensor.permute(3, 0, 1, 2).unsqueeze(0)

    # 3. convert to uint8 [0, 255]
    vis = (vis * 255.0).round().clamp(0, 255).to(torch.uint8)

    # 4. save
    logging.info(f"Saving single output video to: {prefix}/{save_name}")
    save_video(vis, f'{prefix}/{save_name}', fid_vis=False)


def build_scene_memory_bank_from_prev_output_bbox(
    prev_output_np,          # np [F,H,W,C], C=3/4
    prev_bboxes_ts,          # torch [T,4] in target_size xyxy (pixel coords)
    num_extra_frames: int,   # num_extra_frames=8
):
    """
    Returns: torch.float32 [5,3,H,W] in 0~255
    Logic: uniformly sample 5 frames from the last 20 frames of the previous clip (excluding extra tail), zero out human bbox region
    """
    # ---- 1) convert prev_output to uint8 0~255 ----
    vid = prev_output_np
    if vid.dtype != np.uint8:
        vid = vid.astype(np.float32)
        mn, mx = float(vid.min()), float(vid.max())
        # two common ranges: 0~1 or -1~1
        if mn >= -0.1 and mx <= 1.1:
            vid = vid * 255.0
        elif mn >= -1.1 and mx <= 1.1:
            vid = (vid + 1.0) * 0.5 * 255.0
        vid = np.clip(np.round(vid), 0, 255).astype(np.uint8)

    vid = vid[..., :3]  # drop alpha
    F, H, W, _ = vid.shape

    # ---- 2) drop extra tail, sample 5 frames from last 20 ----
    end = max(1, F - num_extra_frames)        # exclude extra tail
    start = max(0, end - 20)
    sample_fids = np.linspace(start, end - 1, 5).round().astype(int).tolist()

    # ---- 3) zero out human bbox per frame (no dilation) ----
    Tb = prev_bboxes_ts.shape[0]
    out = torch.zeros((5, 3, H, W), dtype=torch.float32)

    for k, fi in enumerate(sample_fids):
        bi = min(fi, Tb - 1)
        x0, y0, x1, y1 = prev_bboxes_ts[bi].round().to(torch.int64).tolist()

        # bbox may be -1 (invalid), skip
        if (x0 < 0) or (y0 < 0) or (x1 <= x0) or (y1 <= y0):
            fr = vid[fi]
        else:
            x0 = max(0, min(x0, W - 1))
            x1 = max(0, min(x1, W))
            y0 = max(0, min(y0, H - 1))
            y1 = max(0, min(y1, H))
            fr = vid[fi].copy()
            fr[y0:y1, x0:x1, :] = 0  # zero out human region

        out[k] = torch.from_numpy(fr).permute(2, 0, 1).float()  # 0~255 float

    return out  # [5,3,H,W] float32

def parse_args():
    parser = argparse.ArgumentParser(description='mogen evaluation')
    parser.add_argument('--train_data_root', type=str, default=None, help='train data path')
    parser.add_argument('--pretrain_path', type=str, default=os.environ.get("ONESHOT_MODEL_DIR", "/root/paddlejob/bosdata/yangfengyuan/PretrainModels/Wan/ONESHOT-14B-diffusers"), help='Merged ONESHOT model directory, including the fused transformer / vae / text_encoder / tokenizer / scheduler / preprocess. It can be set to a local copy via the ONESHOT_MODEL_DIR environment variable or this CLI argument.')
    parser.add_argument('--csv_path_name', type=str, default="list_rgbd_testset.csv", help='pretrain path')
    parser.add_argument('--test_image_size', type=int, default=640, help='')
    parser.add_argument('--scheduler', type=str, default='diffsynth', help='scheduler')
    parser.add_argument('--num_frames', type=int, default=81, help='num frames')
    parser.add_argument('--num_prevmotion_frames', type=int, default=5, help='num frames')
    parser.add_argument('--num_extra_frames', type=int, default=8, help='num frames')
    parser.add_argument('--save_path', type=str, default='')
    parser.add_argument('--num_inference_steps', type=int, default=4, help='num frames')
    parser.add_argument('--seed', type=int, default=42, help='seed')
    args = parser.parse_args()
    return args

def set_seed(seed: int, deterministic: bool = False):
    """
    Helper function for reproducible behavior to set the seed in `random`, `numpy`, `torch`.

    Args:
        seed (`int`):
            The seed to set.
        deterministic (`bool`, *optional*, defaults to `False`):
            Whether to use deterministic algorithms where available. Can slow down training.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.use_deterministic_algorithms(True)

if __name__ == '__main__':
    args = parse_args()

    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    device = local_rank
    _init_logging(rank)

    save_prefix = args.save_path

    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            rank=rank,
            world_size=world_size)
        device_ = torch.device(f"cuda:{device}")
    else:
        device_ = torch.device('cuda:0')

    num_frames = args.num_frames

    _csv_stem = os.path.splitext(os.path.basename(args.csv_path_name))[0]
    prefix_origin = save_prefix + '/' + f"{args.scheduler}_{_csv_stem}"
    if not os.path.exists(f'{prefix_origin}'):
        os.makedirs(f'{prefix_origin}', exist_ok=True)

    # oneshot uses CSV format
    csv_path = os.path.join(args.train_data_root, args.csv_path_name)
    with open(csv_path, 'r') as csvfile:
        test_list = list(csv.DictReader(csvfile))

    prompt_path = os.path.join(args.train_data_root, 'prompts.json')
    prompt_dict = json.load(open(prompt_path, 'r'))

    num_videos = len(test_list)
    num_per_rank = num_videos // world_size
    num_left = num_videos % world_size
    logging.info(f"{num_videos, num_per_rank, num_left} ")
    nums = [num_per_rank] * world_size
    if num_left > 0:
        for i in range(num_left):
            nums[i] += 1
    logging.info(nums)
    accumulate_nums = []
    cnt_sum = 0
    for i in range(len(nums)):
        accumulate_nums.append(cnt_sum)
        cnt_sum = cnt_sum + nums[i]
    logging.info(accumulate_nums)
    start = accumulate_nums[rank]
    end = start + nums[rank]
    logging.info(f"rank: {rank}, {start, end}")
    test_list_curr_rank = test_list[start:end]

    if world_size > 1: 
        logging.info(f'=========== Rank: {dist.get_rank()}, {test_list_curr_rank}')
    else:
        logging.info(f'=========== Rank: 0, {test_list_curr_rank}')

    model_id = args.pretrain_path

    vae = AutoencoderKLWan.from_pretrained(
        model_id, subfolder="vae", torch_dtype=torch.float32
    )

    transformer = WanOneshotTransformer3DModel.from_pretrained(
        model_id, subfolder="transformer", torch_dtype=torch.bfloat16,
    )

    shift = 5.0
    if args.scheduler == 'flowmatch':
        # diffsynth
        scheduler = FlowMatchScheduler(shift=shift, sigma_min=0.0, extra_one_step=True)
    elif args.scheduler == 'unipc':
        scheduler = FlowUniPCMultistepScheduler(shift=shift)
    elif args.scheduler == 'sde-dpm++':        
        scheduler = FlowDPMSolverMultistepScheduler(shift=shift, algorithm_type="sde-dpmsolver++")
    elif args.scheduler == 'dpm++':
        scheduler = FlowDPMSolverMultistepScheduler(shift=shift, algorithm_type="dpmsolver++")
    elif args.scheduler == 'lcm':
        scheduler = FlowMatchLCMScheduler(1000, shift=shift)
    else:
        raise ValueError(f"Unknown scheduler: {args.scheduler}")

    pipe = WanOneshotPipeline.from_pretrained(
        model_id,
        vae=vae,
        image_encoder=None,
        transformer=transformer,
        scheduler=scheduler,
        image_processor=None,
        torch_dtype=torch.bfloat16,
    )

    onload_device = device_
    offload_device = torch.device("cpu")
    pipe.transformer.enable_group_offload(
        onload_device=onload_device,
        offload_device=offload_device,
        offload_type="block_level",   
        num_blocks_per_group=1,       
        use_stream=True,             
    )


    pipe.text_encoder.to(onload_device)
    pipe.vae = pipe.vae.to(onload_device)

    pipe.transformer.requires_grad_(False)
    pipe.transformer.eval()
    pipe.vae.requires_grad_(False)
    pipe.vae.eval()

    for idx_video, x_ in enumerate(test_list_curr_rank): # each idx is one long video (>40s)
        try:
            curr_video_name = x_['video_path']
            save_name = '_'.join(x_['video_path'].split('/')).replace('.mp4', '_gen.mp4')
            save_name = save_name.replace('.mp4', f'_{int(time.time())}.mp4')
            save_name_gt = '_'.join(x_['video_path'].split('/')).replace('.mp4', '_gt.mp4')

            prefix = prefix_origin
            if not os.path.exists(prefix):
                os.makedirs(prefix, exist_ok=True)
            
            # read original video frames
            cap = cv2.VideoCapture(os.path.join(args.train_data_root, curr_video_name))
            all_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if all_length < num_frames: continue

            num_firstmotion_frames = 0    # number of GT frames used as motion frames for the first clip
            num_prevmotion_frames = args.num_prevmotion_frames    # number of frames from previous clip used as motion frames
            num_extra_frames = args.num_extra_frames

            prev_output = None
            prev_bboxes_ts = None

            for idx, start_id in enumerate(range(0, all_length, num_frames-num_prevmotion_frames)):   # stride = num_frames - num_prevmotion_frames; first prevmotion_frames are from the previous clip
                num_frames_ext = num_frames + num_extra_frames # generate extra frames beyond clip length
                if start_id + num_frames > all_length: # last clip shorter than num_frames, stop
                    print('end_id exceed, end of the video')
                    break
                save_name = '_'.join(x_['video_path'].split('/')).replace('.mp4', '_gen.mp4')
                save_name = save_name.replace('.mp4', f'_{start_id}-{start_id + num_frames-num_prevmotion_frames}_{int(time.time())}.mp4')
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_id)
                ret, ori_image = cap.read()
                ori_image = Image.fromarray(cv2.cvtColor(ori_image, cv2.COLOR_BGR2RGB))
                original_size = ori_image.size

                ############# update memory bank ##########
                if idx == 0:
                    scene_memory_bank = None
                else:
                    scene_memory_bank = build_scene_memory_bank_from_prev_output_bbox(
                        prev_output_np=prev_output,
                        prev_bboxes_ts=prev_bboxes_ts,
                        num_extra_frames=num_extra_frames,
                    )

                assert num_frames == 81
                assert num_frames_ext == 81 +num_extra_frames # extra frames generated beyond base length
                frames, image, ref_obj_img, inpainting_frames, inpainting_masks, ref_pose_frames, layout_frames, human_mesh_frames, prompt, crop_info, bboxes_cam_square, motion_video, human_h_pos, human_w_pos = generate_input(x_, prompt_dict, is_crop_bbox=True, test_image_size=args.test_image_size, clip_first_frame_idx=start_id - num_prevmotion_frames, num_frames=num_frames_ext, scene_memory_bank=scene_memory_bank)
                assert frames.shape[2] == num_frames_ext #all_length
                prompt = prompt.split('\\n')[idx]
                print('this prompt:', prompt)
                motion_video = last_motion_video_latent = None  # no motion frames
                if idx == 0:    # first clip: record reference, use first 9 frames as motion frames (repeat first frame to length 73, consistent with training)
                    fix_ref_all_long_videos = image
                    # use first prevmotion frames of GT as motion frames
                    inpainting_frames[:,:,:num_firstmotion_frames] = frames[:,:,:num_firstmotion_frames]
                    inpainting_masks[:,:,:num_firstmotion_frames] = 0.
                    motion_video = inpainting_frames[:,:,:num_firstmotion_frames]  # 0-1 -> -1,1
                else:           # subsequent clips: use reference from first clip; motion frames are the tail of the previous clip
                    image[:3] = fix_ref_all_long_videos[:3]
                    image[-1] = fix_ref_all_long_videos[-1]
                    ref_obj_img[:3] = fix_ref_all_long_videos[:3]
                    ref_obj_img[-1] = fix_ref_all_long_videos[-1]
                    # normalize previous clip to the same value range
                    output_tensor = torch.from_numpy(output).to(device)       # [F,H,W,C]
                    output_tensor = output_tensor.permute(3, 0, 1, 2)         # [C,F,H,W]
                    output_tensor = output_tensor.unsqueeze(0).float()        # [1,C,F,H,W]
                    output_tensor = output_tensor / 1.0 * 2.0 - 1.0
                    # use last prevmotion frames of previous clip as motion frames (accounting for extra frames)
                    inpainting_frames[:,:,:num_prevmotion_frames] = output_tensor[:,:,-num_prevmotion_frames-num_extra_frames:output_tensor.shape[2]-num_extra_frames]
                    inpainting_masks[:,:,:num_prevmotion_frames] = 0.
                    motion_video = inpainting_frames[:,:,:num_prevmotion_frames]  # 0-1 -> -1,1
                assert torch.allclose(image, ref_obj_img), "image ref_obj_img not the same"

                ref_image_num = image.shape[0]
                assert ref_image_num == 9  # 3 body + 5 scene memory + 1 face
                assert image.ndim == 4
                def _to_pil_u8_chw(x: torch.Tensor):
                    if torch.is_floating_point(x):
                        x = x.round().clamp(0, 255).to(torch.uint8)
                    elif x.dtype != torch.uint8:
                        x = x.clamp(0, 255).to(torch.uint8)
                    return tvF.to_pil_image(x)

                image = [_to_pil_u8_chw(image_i) for image_i in image]
                ref_obj_img = [_to_pil_u8_chw(ref_obj_img_i) for ref_obj_img_i in ref_obj_img]
                width, height = image[0].size

                # for single reference image:

                negative_prompt = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" 
                logging.info(f'Negative Prompt: {negative_prompt}')

                generator = torch.Generator(device_).manual_seed(args.seed)
                set_seed(args.seed)

                # initialize noise latent (begin)
                if idx == 0:
                    # generate noise latent
                    vae_scale_factor_temporal = 4
                    vae_scale_factor_spatial = 8
                    num_latent_frames_per_sample = (num_frames_ext - 1) // 4 + 1 + ref_image_num # 21 + 3
                    num_sample = all_length // (num_frames-num_prevmotion_frames) + 1
                    latent_height = height // vae_scale_factor_spatial
                    latent_width = width // vae_scale_factor_spatial

                    shape = (1, pipe.vae.config.z_dim, num_latent_frames_per_sample * num_sample, latent_height, latent_width)
                    stride = (num_frames-num_prevmotion_frames) // 4 # 19
                    latents_wholeseq = randn_tensor(shape, generator=generator, device=pipe._execution_device, dtype=torch.float32)
                    latents_thisseq = latents_wholeseq[:, :, idx*stride:idx*stride+num_latent_frames_per_sample, :, :]
                else:
                    latents_thisseq = latents_wholeseq[:, :, idx*stride:idx*stride+num_latent_frames_per_sample, :, :]

                # initialize noise latent (end)

                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                    with torch.no_grad():
                        output = pipe(
                            image=image,
                            ref_obj_img=ref_obj_img,
                            inpainting_frames=inpainting_frames,
                            inpainting_masks=inpainting_masks,
                            ref_pose_frames=ref_pose_frames,
                            layout_frames=layout_frames,
                            negative_prompt=negative_prompt, 
                            prompt=prompt,
                            height=height,
                            width=width,
                            num_frames=num_frames_ext,
                            guidance_scale=1.0,
                            generator=generator,
                            num_inference_steps=args.num_inference_steps,
                            human_mesh_frames=human_mesh_frames,
                                    bboxes_cam_square=bboxes_cam_square,
                            human_h_pos=human_h_pos.unsqueeze(0),
                            human_w_pos=human_w_pos.unsqueeze(0),
                                        latents=latents_thisseq,
                        ).frames[0]
                        assert output.shape[0] == 81 + num_extra_frames # 81 base frames + extra frames
                        # update memory preparation
                        prev_output = output
                        prev_bboxes_ts = bboxes_cam_square.detach().cpu() 
                

                export_single_video_only(output, prefix, save_name.replace('.mp4', '_ourGen81.mp4'))
        except Exception as e:
            logging.error(e)
            traceback.print_exc()
        


            
