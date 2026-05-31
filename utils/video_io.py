"""/usr/bin/env python3"""

import os
import skvideo.io
import cv2
import torch
import numpy as np
from PIL import Image
from einops import rearrange
import decord
import tempfile
import PIL
from tqdm import tqdm
from concurrent.futures import (
    ProcessPoolExecutor, 
    ThreadPoolExecutor, 
    as_completed
)
import torchvision
import torchvision.transforms as transforms
import copy


def to_np(x):
    """Convert a tensor or an array into numpy"""
    return x.cpu().numpy()


def auto_scale(x):
    """Auto scale the input"""
    max_v = x.max()
    min_v = x.min()
    if max_v <= 1 and min_v < 0 and min_v >= -1:
        x = (x * 0.5 + 0.5) * 255
    elif max_v <= 1 and min_v >= 0:
        x = x * 255
    return x.astype(np.uint8)


def auto_reshape(x):
    """Reshape the input"""
    n_dim = len(x.shape)
    if isinstance(x, torch.Tensor):
        if n_dim == 3:
            if x.shape[0] == 3:
                return rearrange(x, "c h w -> 1 h w c")
        elif n_dim == 4:
            if x.shape[1] == 3:
                return rearrange(x, "f c h w -> f h w c")
        elif n_dim == 5:
            if x.shpae[1] == 3:
                return rearrange(x, "b c f h w -> (b f) h w c")
            elif x.shape[2] == 3:
                return rearrange(x, "b f c h w -> (b f ) h w c")
    elif isinstance(x, np.ndarray):
        if n_dim == 3:
            if x.shape[0] == 3:
                return x.transpose(1, 2, 0)
        elif n_dim == 4:
            if x.shape[1] == 3:
                return x.transpose(0, 2, 3, 1)
        elif n_dim == 5:
            print("x.shape:", x.shape)
            if x.shape[1] == 3:
                x = x.transpose(0, 2, 3, 4, 1)
                b, f, h, w, c = x.shape
                x = x.reshape((b * f, h, w, c))
                return x
            elif x.shape[2] == 3:
                x = x.transpose(0, 1, 3, 4, 2)
                b, f, h, w, c = x.shape
                x = x.reshape((b * f, h, w, c))
                return x
    return x


def save_image(image, out_path):
    """Save images"""
    if isinstance(image, torch.Tensor):
        # image = to_np(image)
        image = auto_scale(to_np(auto_reshape(image)))
        image = image[0]
        Image.fromarray(image).save(out_path)
    elif isinstance(image, np.ndarray):
        image = auto_scale(auto_reshape(image))
        image = image[0]
        Image.fromarray(image).save(out_path)
    elif isinstance(image, PIL.Image.Image):
        image.convert("RGB").save(out_path)
    else:
        raise ValueError(f"Unsupported type of image: {type(image)}")


def save_video(frames, video_out_path, fid_vis=False):
    """Save video"""
    if isinstance(frames, torch.Tensor):
        frames = to_np(frames)
        frames = auto_scale(auto_reshape(frames))
        frames = np.ascontiguousarray(frames)
    basedir = os.path.dirname(video_out_path)
    if basedir != "":
        os.makedirs(basedir, exist_ok=True)
    input_dict = {
        "-r": "16"
    }
    output_dict = {
        "-r": "16",
        "-pix_fmt": "yuv420p",
        "-crf": "9",
        "-vf": "colorspace=all=bt709:iall=bt601-6-625:fast=1",
        "-c:v": "libx264",
    }
    writer = skvideo.io.FFmpegWriter(
        video_out_path, inputdict=input_dict, outputdict=output_dict, verbosity=1
    )
    fid = 0
    for frame in frames:
        if fid_vis:
            frame = frame.astype(np.uint8)
            H, W = frame.shape[:2]
            # print("frame:", frame.shape)
            cv2.putText(
                frame,
                str(fid).zfill(4),
                (int(H * 0.1), int(W * 0.1)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (255, 0, 0),
                2,
            )
            fid += 1
        writer.writeFrame(frame)
    writer.close()


def save_video_as_grid_and_mp4(video_batch: torch.Tensor, save_path: str):
    """Save videos as grid and mp4 format"""
    gif_frames = []
    count = 0
    for i, vid in enumerate(video_batch):
        if isinstance(vid, torch.Tensor):
            vid = to_np(vid)
            vid = auto_scale(auto_reshape(vid))
            vid = np.ascontiguousarray(vid)
        for frame in vid:
            # frame = rearrange(frame, "c h w -> h w c")
            # frame = (255.0 * frame).cpu().numpy().astype(np.uint8)
            # frame = frame.copy()
            cv2.putText(
                frame,
                str(count),
                (int(100), int(100)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 0, 0),
                thickness=1,
            )
            count += 1
            gif_frames.append(frame)
    print("gif_frames:", len(gif_frames))
    name = 0
    now_save_path = save_path + "_" + str(name) + ".mp4"
    while os.path.exists(now_save_path):
        name += 1
        now_save_path = save_path + "_" + str(name) + ".mp4"

    save_video(gif_frames, now_save_path, fid_vis=False)


def load_image(image_path):
    """Load image"""
    img = Image.open(image_path)
    return np.array(img)


def load_video(vid_path, inds=None, return_array=True):
    """Load video"""
    vr = decord.VideoReader(vid_path)
    if inds is None:
        inds = range(0, len(vr))
    out = []
    for i in inds:
        frame = vr[i].asnumpy()
        out.append(frame)
    if return_array:
        return np.array(out)
    else:
        return out


def export_to_video(
    video_frames, output_video_path, fid_vis=False, save_img=False, fps=16, insert_mode=False
) -> str:
    """Export a list of images or videos into an mp4 file."""
    if output_video_path is None:
        output_video_path = tempfile.NamedTemporaryFile(suffix=".mp4").name
    
    if insert_mode:
        save_img = False

    if save_img:
        image_path = output_video_path.replace(".mp4", "")
        if not os.path.exists(image_path):
            os.makedirs(image_path, exist_ok=True)

    if isinstance(video_frames[0], np.ndarray):
        if save_img:
            for ind, frame in enumerate(video_frames):
                frame = Image.fromarray(frame).convert("RGB")
                frame.save(image_path + f"/{'%06d' % ind}.png")
        video_frames = [(frame * 255).astype(np.uint8) for frame in video_frames]

    elif isinstance(video_frames[0], PIL.Image.Image):
        if save_img:
            # for ind, frame in enumerate(video_frames):
            #     frame = frame.convert("RGB")
            #     frame.save(image_path + f"/{'%06d' % ind}.png")
            task_pool = set()
            with ProcessPoolExecutor(max_workers=16) as executor:
                for index in range(len(video_frames)):
                    task_pool.add(
                        executor.submit(
                            save_image,
                            video_frames[index],
                            f"{image_path}/{'%06d' % index}.png",
                        )
                    )
                pbar = tqdm(total=len(task_pool), desc="save img")  # Init pbar
                for task in as_completed(task_pool):
                    pbar.update(1)
                    task.result()
                pbar.close()
        video_frames = [np.array(frame) for frame in video_frames]

    basedir = os.path.dirname(output_video_path)

    if basedir != "":
        os.makedirs(basedir, exist_ok=True)
    input_dict = {"-r": f"{fps}"}  # 设置输入帧率为 25 FPS
    if not insert_mode:
        output_dict = {
            "-pix_fmt": "yuv420p",
            "-crf": "9",
            "-vf": "colorspace=all=bt709:iall=bt601-6-625:fast=1",
            "-c:v": "libx264",
        }
    else:
        output_dict = {
            "-pix_fmt": "yuv420p",
            "-crf": "9",
            "-vf": "colorspace=all=bt709:iall=bt601-6-625:fast=1",
            "-c:v": "libx264",
            "-g": "10",
        }
    writer = skvideo.io.FFmpegWriter(
        output_video_path, inputdict=input_dict, outputdict=output_dict, verbosity=1
    )
    fid = 0
    for frame in video_frames:
        if fid_vis:
            frame = frame.astype(np.uint8)
            H, W = frame.shape[:2]
            cv2.putText(
                frame,
                str(fid).zfill(4),
                (int(H * 0.1), int(W * 0.1)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (255, 0, 0),
                2,
            )
            fid += 1
        writer.writeFrame(frame)
    writer.close()
    return output_video_path


def inverse_paste(image, ref_frame, crop_info):
    """
    video: Image
    ref_frame: Image
    """
    # print("======== Paste to Original Video =========")
    # print(f"ref_frame: {ref_frame.size}")
    ori_video_shape = crop_info["ori_video_shape"]
    crop_resize_info = crop_info["crop_info"]
    
    up, left, crop_H, crop_W, new_H, new_W = [
        crop_resize_info[k]
        for k in ["up", "left", "crop_H", "crop_W", "new_H", "new_W"]
    ]
    # print(crop_resize_info)
    image = torchvision.transforms.functional.resize(
        image, size=(crop_H, crop_W), interpolation=torchvision.transforms.InterpolationMode.BICUBIC
    )
    image_out = copy.deepcopy(ref_frame)
    image_out.paste(image, (left, up))

    # 新增：将图像向下padding至1920高度
    target_height = 1920
    current_height = image_out.height
    if current_height < target_height:
        # 创建一个全黑的背景图像
        padded_image = Image.new('RGB', (image_out.width, target_height), (0, 0, 0))
        # 将原始图像粘贴到背景图像的顶部
        padded_image.paste(image_out, (0, 0))
        image_out = padded_image

    return image_out