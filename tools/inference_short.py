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
    divided_by = 32
    min_edge = int(image_size / 1.4)
    max_edge = int(image_size * 1.4)
    token_number = image_size * image_size / divided_by / divided_by
    for i in range(min_edge // divided_by, max_edge // divided_by + 1):
        all_resolution.append([i * divided_by, int(token_number // i * divided_by)])
    return all_resolution


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

def make_slow_indices(indices, slow_factor=2):
    # indices: list[int], len=T
    T = len(indices)
    out = []
    for t in range(T):
        src = min(int(t // slow_factor), T - 1)
        out.append(indices[src])
    return out

def preprocess_video(video_path, depth_rgb3_path, geom_rgb_path, human_mesh_path, bbox_path, smplx_path, test_image_size=640, clip_first_frame_idx=0, num_frames=81, specify_ID_profile_path=None, slow_background=False):
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

        indices_smpl = indices
        # background 2x slow-motion (each frame repeated once)
        if slow_background:
            indices_bg = make_slow_indices(indices_smpl, slow_factor=2)
        else:
            indices_bg = indices
        
        ### read video by index ###
        # GT video
        frames = video_reader.get_batch(indices)
        frames = frames.permute(0, 3, 1, 2).contiguous()

        # read depth video
        if depth_rgb3_path is not None:
            depth_rgb_video_reader = decord.VideoReader(uri=depth_rgb3_path.as_posix())
            frames_depth_rgb = depth_rgb_video_reader.get_batch(indices_bg)
            ref_pose_frames = frames_depth_rgb.permute(0,3,1,2).contiguous().float()
        # read layout video -> geom_rgb_path
        if geom_rgb_path is not None:
            geom_rgb_video_reader = decord.VideoReader(uri=geom_rgb_path.as_posix())
            frames_geom_rgb = geom_rgb_video_reader.get_batch(indices_bg)
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
        if bbox_path is not None and os.path.exists(bbox_path):
            
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
            b_rp_orig = torch.tensor(bboxes_cam_square, dtype=torch.float32)

            b_rp_orig_gt = b_rp_orig.clone()  # optional: keep a copy of original bbox for debugging
            cam_npz_a = os.path.join(os.path.dirname(smplx_path), "camera_and_humanmask.npz")
            cam_npz_b = os.path.join(os.path.dirname(geom_rgb_path.as_posix()), "camera_and_humanmask.npz")

            # degenerate case: same person + same camera (ID swap only; motion and scene from original video),
            # bbox should equal the original video bbox; skip recompute to avoid introducing drift.
            scene_smplx_path = os.path.join(os.path.dirname(geom_rgb_path.as_posix()),
                                            "smplx_pred_params_all.npz")
            same_motion_and_cam = (
                list(indices_smpl) == list(indices_bg)
                and os.path.realpath(cam_npz_a) == os.path.realpath(cam_npz_b)
                and os.path.exists(scene_smplx_path)
                and os.path.realpath(str(smplx_path)) == os.path.realpath(scene_smplx_path)
            )

            if same_motion_and_cam:
                print("[ONESHOT-RGBD] same motion & camera as scene -> keep original bbox, skip recompute")
                b_rp_orig = b_rp_orig.clone()
            else:
                b_rp_orig = recompute_square_bboxes_from_smplx_and_camera(
                    bboxes_anchor_xyxy=b_rp_orig,     # still use only frame 0 as anchor
                    indices_a=indices_smpl,           # person unchanged
                    indices_b=indices_bg,             # background slow-motion (camera slow-motion)
                    smplx_npz_a=smplx_path,
                    cam_npz_a=cam_npz_a,
                    cam_npz_b=cam_npz_b,
                    H_rp=H_rp, W_rp=W_rp,
                    human_height_m=1.75,
                    prefer_pelvis=True,
                    clip_to_image=False,
                    smooth_logL=True,
                )

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
                # ref_smplx auto-derive: prefer the smplx that lives next to video_path
                ref_smplx_candidate = os.path.join(os.path.dirname(str(video_path)),
                                                   'smplx_pred_params_all.npz')
                ref_smplx_path = ref_smplx_candidate if os.path.exists(ref_smplx_candidate) else smplx_path
                ref_image, three_angle_ref_global_idx, _ = get_three_ref_images_fullseq(
                        video_reader=video_reader,
                        bboxes_cam_square_full=bboxes_for_preprocess,
                        smplx_path=ref_smplx_path,
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
            # background reference frames (no scene memory, zero-padded)
            bg_ref = torch.zeros((5, 3, target_size[0], target_size[1]), dtype=ref_image.dtype, device=ref_image.device)
            ref_image = torch.cat([ref_image, bg_ref], dim=0)
            ref_image = torch.cat([ref_image,face_ref.to(ref_image.device)], dim=0)  # face ref appended at the end
            ref_human_img = ref_image

            # 2) ref_pose/layout: crop at its own resolution
            ref_pose_frames = crop_resize_by_bboxes_xyxy(ref_pose_frames, b_rp, target_size=target_size)
            layout_frames   = crop_resize_by_bboxes_xyxy(layout_frames,   b_rp, target_size=target_size)


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


def generate_input(data_dict, prompt_dict, test_image_size=640, clip_first_frame_idx=0, num_frames=81, slow_background=False):
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
    # specify_ID_profile_path may be None → inference will auto-extract 3-angle reference frames from video_path
    
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
        test_image_size,
        clip_first_frame_idx,
        num_frames,
        specify_ID_profile_path=specify_ID_profile_path,
        slow_background=slow_background
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


    return frames, image, ref_human_img, inpainting_frames, inpainting_masks, ref_pose_frames, layout_frames, human_mesh_frames, prompt, crop_info, bboxes_cam_square, motion_video, human_h_pos, human_w_pos, specify_ID_profile_path, smplx_param_path

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


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('true', '1'):
        return True
    elif v.lower() in ('false', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('需要输入布尔值（true/false/True/False/1/0）')

def _rt34_to_T44(rt34: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float32)
    T[:3, :4] = rt34.astype(np.float32)
    return T

def _invert_w2c_T44(T_w2c: np.ndarray) -> np.ndarray:
    R = T_w2c[:3, :3]
    t = T_w2c[:3, 3:4]
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R.T
    T[:3, 3:4] = -R.T @ t
    return T

def _load_camera_npz(cam_npz_path: str):
    cam = np.load(cam_npz_path, allow_pickle=True)
    intr = cam["intrinsics"].astype(np.float32)          # [N,3,3]
    extr = cam["extrinsics_w2c"].astype(np.float32)      # [N,3,4]
    return intr, extr

def _get_first_person_vec3(obj, person_index=0):
    arr = np.asarray(obj)
    # case: empty
    if arr.size == 0:
        return None
    # case: [3]
    if arr.ndim == 1 and arr.shape[0] == 3:
        return arr.astype(np.float32)
    # case: [N,3]
    if arr.ndim == 2 and arr.shape[1] == 3:
        if arr.shape[0] <= person_index:
            return None
        return arr[person_index].astype(np.float32)
    return None


def _load_smpl_center_cam_clip(smplx_npz_a: str, indices, person_index: int = 0, prefer_pelvis=True):
    smpl = np.load(smplx_npz_a, allow_pickle=True)
    transl_list = smpl["smpl_transl"]  # object list
    has_delta = prefer_pelvis and ("delta_hp" in smpl.files)
    delta_list = smpl["delta_hp"] if has_delta else None
    T = len(indices)
    out = [None] * T
    # read frame by frame (allow None)
    for i, fid in enumerate(indices):
        t = _get_first_person_vec3(transl_list[fid], person_index=person_index)

        if t is not None and has_delta:
            d = _get_first_person_vec3(delta_list[fid], person_index=person_index)
            if d is not None:
                t = t + d  # pelvis

        out[i] = t
    # if all None, raise a clearer error
    if all(v is None for v in out):
        raise RuntimeError(
            f"[SMPLX EMPTY TRACK] all frames have empty smpl_transl for person_index={person_index}. "
            f"smplx_npz={smplx_npz_a}"
        )
    # forward fill: replace gaps with nearest valid value
    last = None
    for i in range(T):
        if out[i] is None:
            out[i] = last
        else:
            last = out[i]
    # if leading Nones (forward fill cannot cover), back-fill with first valid value
    first_valid = next(v for v in out if v is not None)
    for i in range(T):
        if out[i] is None:
            out[i] = first_valid
        else:
            break
    missing = [fid for i,fid in enumerate(indices) if out[i] is None]
    if len(missing) > 0:
        print(f"[SMPLX WARN] missing transl frames: {missing[:10]}{'...' if len(missing)>10 else ''} (count={len(missing)})")
    return np.stack([v.astype(np.float32) for v in out], axis=0)  # [T,3]


@torch.no_grad()
def recompute_square_bboxes_from_smplx_and_camera(
    bboxes_anchor_xyxy: torch.Tensor,
    indices_a,                  # for SMPLX / cam_npz_a (person unchanged)
    indices_b,                  # for cam_npz_b (background slow-motion)
    smplx_npz_a: str,
    cam_npz_a: str,
    cam_npz_b: str,
    H_rp: int, W_rp: int,
    person_index: int = 0,
    human_height_m: float = 1.75,
    prefer_pelvis: bool = True,
    clip_to_image: bool = True,
    smooth_logL: bool = False,
):
    """
    Cross: SMPL motion from A, camera from B.
    Fix: anchor in B-cam0 using bbox0 + human height H, then translate A's relative motion to B-cam0,
           project with B's relative camera pose; bbox size uses fy*H/z (preserves body height).
    """
    indices_a = list(indices_a)
    indices_b = list(indices_b)
    T = len(indices_a)
    assert len(indices_b) == T, "indices_a / indices_b length mismatch"

    intr_a, extr_a = _load_camera_npz(cam_npz_a)
    intr_b, extr_b = _load_camera_npz(cam_npz_b)

    E_a = extr_a[indices_a].astype(np.float32)  # A camera (used to transform SMPL into A cam0)
    K_b = intr_b[indices_b].astype(np.float32)  # B intrinsics (slow-motion)
    E_b = extr_b[indices_b].astype(np.float32)  # B extrinsics (slow-motion)

    # SMPLX (person unchanged): use indices_a
    X_A_cam = _load_smpl_center_cam_clip(smplx_npz_a, indices_a, person_index=person_index, prefer_pelvis=prefer_pelvis)


    T_w2c_A0 = _rt34_to_T44(E_a[0])
    X_A0 = np.zeros((T, 3), dtype=np.float32)
    for t in range(T):
        T_w2c_At = _rt34_to_T44(E_a[t])
        T_c2w_At = _invert_w2c_T44(T_w2c_At)
        Xh = np.ones((4,1), dtype=np.float32); Xh[:3,0] = X_A_cam[t]
        X0 = (T_w2c_A0 @ (T_c2w_At @ Xh))  # in A cam0 coords
        X_A0[t] = X0[:3,0]

    # --- anchor bbox0 (pixel) ---
    b0 = bboxes_anchor_xyxy[0].float().cpu().numpy()
    x0, y0, x1, y1 = map(float, b0.tolist())
    cu0 = 0.5 * (x0 + x1)
    cv0 = 0.5 * (y0 + y1)
    L0_px = max(2.0, float(max(x1 - x0, y1 - y0)))

    # --- build B cam0 3D anchor from bbox0 + human height ---
    K0 = K_b[0]
    fx0, fy0 = float(K0[0,0]), float(K0[1,1])
    cx0, cy0 = float(K0[0,2]), float(K0[1,2])

    z0 = (fy0 * float(human_height_m)) / max(L0_px, 1e-6)
    x0_3d = (cu0 - cx0) * (z0 / fx0)
    y0_3d = (cv0 - cy0) * (z0 / fy0)
    P0_B_cam0 = np.array([x0_3d, y0_3d, z0], dtype=np.float32)

    # --- translate A0 trajectory into B cam0 coords (no scale, no rotation) ---
    X_B0 = (X_A0 - X_A0[0:1]) + P0_B_cam0[None, :]

    # --- precompute B relative transform cam0 -> camt ---
    T_w2c_B0 = _rt34_to_T44(E_b[0])
    T_c2w_B0 = _invert_w2c_T44(T_w2c_B0)

    bboxes = np.zeros((T,4), dtype=np.float32)

    # optional smoothing on log(L)
    logL_prev = np.log(L0_px)

    for t in range(T):
        Kt = K_b[t]
        fx, fy = float(Kt[0,0]), float(Kt[1,1])
        cx, cy = float(Kt[0,2]), float(Kt[1,2])

        T_w2c_Bt = _rt34_to_T44(E_b[t])

        Xh = np.ones((4,1), dtype=np.float32); Xh[:3,0] = X_B0[t]
        X_ct = (T_w2c_Bt @ (T_c2w_B0 @ Xh))  # in B cam_t
        x, y, z = float(X_ct[0,0]), float(X_ct[1,0]), max(float(X_ct[2,0]), 1e-6)

        u = fx * (x / z) + cx
        v = fy * (y / z) + cy

        # size from human height (keeps body height consistent)
        L = (fy * float(human_height_m)) / z
        L = max(2.0, float(L))

        if smooth_logL:
            alpha = 0.2
            logL = (1 - alpha) * logL_prev + alpha * np.log(L)
            logL_prev = logL
            L = float(np.exp(logL))

        half = 0.5 * L
        xmin = u - half
        xmax = u + half
        ymin = v - half
        ymax = v + half

        if clip_to_image:
            xmin = float(np.clip(xmin, 0.0, W_rp - 1.0))
            xmax = float(np.clip(xmax, 0.0, W_rp - 1.0))
            ymin = float(np.clip(ymin, 0.0, H_rp - 1.0))
            ymax = float(np.clip(ymax, 0.0, H_rp - 1.0))

        bboxes[t] = [xmin, ymin, xmax, ymax]

    # enforce exact bbox0
    bboxes[0] = np.array([x0, y0, x1, y1], dtype=np.float32)
    return torch.from_numpy(bboxes).float()


def export_to_video_with_reference_2col(
    output, prefix, save_name,
    ref_pose_frames, layout_frames, human_mesh_frames, frames,
    ref_image, motion_video
):
    import torch
    import logging
    import torchvision.transforms.functional as tvF
    from utils.video_io import save_video

    device = frames.device

    # -------------------------
    # helper: ref_image -> [C,H,W] in [-1,1]
    # -------------------------
    def _to_chw_minus1_1(img):
        if torch.is_tensor(img):
            t = img
            # allow [H,W,C] / [C,H,W]
            if t.ndim == 3 and t.shape[0] not in (1, 3) and t.shape[-1] in (1, 3):
                t = t.permute(2, 0, 1)
            t = t.to(device).float()
            # if likely uint8 0~255
            if t.max() > 1.5:
                t = t / 255.0
            return t * 2 - 1
        else:
            # PIL or numpy
            t = tvF.to_tensor(img).to(device)  # [C,H,W], 0~1
            return t * 2 - 1

    # -------------------------
    # 1) output -> [1,C,F,H,W] in [-1,1]
    # -------------------------
    output_tensor = torch.from_numpy(output).to(device)       # [F,H,W,C]
    output_tensor = output_tensor.permute(3, 0, 1, 2)         # [C,F,H,W]
    output_tensor = output_tensor.unsqueeze(0).float()        # [1,C,F,H,W]
    output_tensor = output_tensor / 1.0 * 2.0 - 1.0

    # -------------------------
    # 2) ref_image: single or list -> concat along width, then repeat over frames
    # -------------------------
    if isinstance(ref_image, (list, tuple)):
        ref_list = list(ref_image)
    else:
        ref_list = [ref_image]

    ref_tensors = [_to_chw_minus1_1(img) for img in ref_list]   # each [C,H,W] in [-1,1]

    # 对齐到同一 H/W（以第一张为基准；如果你想强制到 frames 的分辨率，也可以改这里）
    base_h, base_w = ref_tensors[0].shape[-2], ref_tensors[0].shape[-1]
    aligned = []
    for t in ref_tensors:
        if t.shape[-2:] != (base_h, base_w):
            t = torch.nn.functional.interpolate(
                t.unsqueeze(0), size=(base_h, base_w),
                mode="bilinear", align_corners=False
            ).squeeze(0)
        aligned.append(t)

    ref_concat = torch.cat(aligned, dim=-1)  # [C, H, sumW]
    vis_ref_img = (
        ref_concat.unsqueeze(0)                               # [1,C,H,sumW]
        .repeat(frames.shape[2], 1, 1, 1)                     # [F,C,H,sumW]
        .unsqueeze(0)                                         # [1,F,C,H,sumW]
        .permute(0, 2, 1, 3, 4).contiguous()                  # [1,C,F,H,sumW]
    )

    # -------------------------
    # 3) motion video vis
    # -------------------------
    motion_video_vis = torch.zeros_like(frames)
    motion_video_vis[:, :, :motion_video.shape[2]] = motion_video

    # -------------------------
    # 4) human mesh resize
    # -------------------------
    human_mesh_frames = torch.nn.functional.interpolate(
        human_mesh_frames.permute(0, 2, 1, 3, 4)[0],
        size=(layout_frames.shape[-2], layout_frames.shape[-2]),  # H=W=layoutH
        mode='bilinear',
        align_corners=False
    )[None, :].permute(0, 2, 1, 3, 4)

    # ---- 两行 & 右对齐 ----
    row1 = torch.cat([vis_ref_img, motion_video_vis, frames], dim=-1)
    row2 = torch.cat([ref_pose_frames, layout_frames, human_mesh_frames, output_tensor], dim=-1)

    if row1.shape[-1] < row2.shape[-1]:
        pad = (row2.shape[-1] - row1.shape[-1], 0, 0, 0, 0, 0)
        row1 = torch.nn.functional.pad(row1, pad)
    elif row2.shape[-1] < row1.shape[-1]:
        pad = (row1.shape[-1] - row2.shape[-1], 0, 0, 0, 0, 0)
        row2 = torch.nn.functional.pad(row2, pad)

    vis_video = torch.cat([row1, row2], dim=-2)

    # -------------------------
    # 5) save
    # -------------------------
    vis = (vis_video.clamp(-1, 1) + 1) / 2.0
    vis = (vis * 255.0).round().clamp(0, 255).to(torch.uint8)

    logging.info(f"Saving 2-row comparison video to: {prefix}/{save_name}")
    save_video(vis, f'{prefix}/{save_name}', fid_vis=True)


            


def parse_args():
    parser = argparse.ArgumentParser(description='mogen evaluation')
    parser.add_argument('--train_data_root', type=str, default=None, help='train data path')
    parser.add_argument('--pretrain_path', type=str, default=os.environ.get("ONESHOT_MODEL_DIR", "/root/paddlejob/bosdata/yangfengyuan/PretrainModels/Wan/ONESHOT-14B-diffusers"), help='Merged ONESHOT model directory, including the fused transformer / vae / text_encoder / tokenizer / scheduler / preprocess. It can be set to a local copy via the ONESHOT_MODEL_DIR environment variable or this CLI argument.')
    parser.add_argument('--csv_path_name', type=str, default="list_rgbd_testset.csv", help='pretrain path')
    parser.add_argument('--test_image_size', type=int, default=640, help='')
    parser.add_argument('--scheduler', type=str, default='diffsynth', help='scheduler')
    parser.add_argument('--num_frames', type=int, default=81, help='num frames')
    parser.add_argument('--save_path', type=str, default='')
    parser.add_argument('--slow_background', type=str2bool, default=False)
    parser.add_argument('--num_inference_steps', type=int, default=4, help='num frames')
    parser.add_argument('--seed', type=int, default=41, help='seed')
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
    if not os.path.exists(prefix_origin):
        os.makedirs(prefix_origin, exist_ok=True)

    # oneshot uses CSV format
    csv_path = os.path.join(args.train_data_root, args.csv_path_name)
    with open(csv_path, 'r') as csvfile:
        test_list = list(csv.DictReader(csvfile))

    # prompt is read from the CSV "prompt" column (see generate_input -> data_dict["prompt"])
    prompt_dict = {}

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

    for idx_video, x_ in enumerate(test_list_curr_rank): 
        try:
            curr_video_name = x_['video_path']  
            save_name = '_'.join(x_['video_path'].split('/')).replace('.mp4', '_gen.mp4')
            save_name = save_name.replace('.mp4', f'_{int(time.time())}.mp4')
            save_name_gt = '_'.join(x_['video_path'].split('/')).replace('.mp4', '_gt.mp4')

            # save_name encodes ID/SMPL/Scene info; all videos land in the same dir
            def _seq_name(p):
                # extract seq name (/.../<seq>/<clip>/foo.mp4 -> <seq>)
                return os.path.basename(os.path.dirname(os.path.dirname(p)))
            prefix = prefix_origin

            if not os.path.exists(prefix):
                os.makedirs(prefix, exist_ok=True)

            _id_token = x_['specify_ID_profile_path'].split('/')[-1] if x_.get('specify_ID_profile_path') else "fromVideo"
            save_name = f"ID_{_id_token}-SMPLX_{_seq_name(x_['human_mesh_path'])}-Scene_{_seq_name(x_['geom_rgb_path'])}_gen.mp4"
            save_name = save_name.replace('.mp4', f'_{int(time.time())}.mp4')

            assert num_frames == 81
            frames, image, ref_obj_img, inpainting_frames, inpainting_masks, ref_pose_frames, layout_frames, human_mesh_frames, prompt, crop_info, bboxes_cam_square, motion_video, human_h_pos, human_w_pos, specify_ID_profile_path, smplx_param_path = generate_input(x_, prompt_dict, test_image_size=args.test_image_size, clip_first_frame_idx=0, num_frames=num_frames, slow_background=args.slow_background)
            assert frames.shape[2] == num_frames #all_length
            print('this prompt:', prompt)
            motion_video = last_motion_video_latent = None  # no motion frames
            motion_video = inpainting_frames[:,:,:0]
            assert torch.allclose(image, ref_obj_img), "image and ref_obj_img not same!"


            
            logging.info(f'多图参考，共参考图: {image.shape[0]}')
            ref_image_num = image.shape[0]
            assert ref_image_num == 9  # 3 body refs + 5 bg refs (zeros) + 1 face ref
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

            negative_prompt = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走，飞动的物体，人物和物体交互不自然" #，人物和地面交互不自然
            logging.info(f'Negative Prompt: {negative_prompt}')

            generator = torch.Generator(device_).manual_seed(args.seed)
            set_seed(args.seed)


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
                        num_frames=num_frames,
                        guidance_scale=1.0,
                        generator=generator,
                        num_inference_steps=args.num_inference_steps,
                        human_mesh_frames=human_mesh_frames,
                        bboxes_cam_square=bboxes_cam_square,
                        human_h_pos=human_h_pos.unsqueeze(0),
                        human_w_pos=human_w_pos.unsqueeze(0),
                    ).frames[0]
                    assert output.shape[0] == 81 
            
            # export_to_video_with_reference_2col(output, prefix, save_name, ref_pose_frames*inpainting_masks, layout_frames*inpainting_masks, human_mesh_frames, frames, image, motion_video)
            export_single_video_only(output, prefix, save_name.replace('.mp4', '_ourGen81.mp4'))
            info_data = {
                "ID_profile": specify_ID_profile_path,
                "SMPLX_PARAM": smplx_param_path,
                "prompt": prompt
            }
            json_file_path = os.path.join(prefix, save_name.replace('.mp4', '_ourGen81.json'))
            with open(json_file_path, 'w', encoding='utf-8') as f:
                json.dump(info_data, f, ensure_ascii=False, indent=4)
            print(f"Saved info json to: {json_file_path}")


        except Exception as e:
            logging.error(e)
            traceback.print_exc()
        
