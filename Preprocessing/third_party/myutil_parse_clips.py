import os, glob, cv2, tempfile, shutil, subprocess, concurrent.futures
from typing import List, Tuple, Optional

def parse_and_chunk_seq(
    p,
    clip_len_s: int = 10,
    stride_s: float   = 5.0,
    min_total_s: int = 15,
    target_fps: Optional[float] = None,   # <<< [新增] 不传则不影响原逻辑
):
    """
    返回 clips 列表；元素字段与原实现完全一致。
    - 视频：优先 ffmpeg 抽帧；失败再回退到 OpenCV+线程写盘
    - 文件夹：直接切片（默认 fps=30.0）
    """
    IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")  # 可按需增减

    def _read_first_size_from_folder(folder):
        imlist = sorted([x for x in glob.glob(os.path.join(folder, "*")) if os.path.splitext(x)[1].lower() in IMG_EXTS])
        if len(imlist) == 0:
            raise ValueError(f"No images in folder: {folder}")
        img0 = cv2.imread(imlist[0], cv2.IMREAD_UNCHANGED)
        if img0 is None:
            raise ValueError(f"Failed to read first image in folder: {imlist[0]}")
        H0, W0 = img0.shape[:2]
        return H0, W0, imlist

    # ---------- Fast path: ffmpeg 抽帧 ----------
    def _extract_with_ffmpeg(video_path, target_fps=None):
        """
        用 ffmpeg 一次性解全帧到 tmpdir，命名为 frame_%06d.jpg
        返回：((H0,W0), fps, img_paths, tmpdir)
        """
        # 优先用 imageio_ffmpeg 自带的 ffmpeg（与 oneshot env 一起打包，免外部 PATH 依赖）；
        # 找不到再回退到 PATH 上的 "ffmpeg"。ffprobe 没有自带的，PATH 找不到就静默失败，
        # 后续会从首帧 cv2.imread 补 H/W、target_fps 补 fps。
        try:
            import imageio_ffmpeg
            FFMPEG_BIN = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            FFMPEG_BIN = "ffmpeg"
        FFPROBE_BIN = "ffprobe"

        tmpdir = tempfile.mkdtemp()
        # 1) 读元信息
        def _probe(fmt):
            try:
                out = subprocess.check_output(fmt, stderr=subprocess.STDOUT)
                return out.decode("utf-8", "ignore")
            except Exception:
                return ""
        # fps
        probefps = _probe([FFPROBE_BIN,"-v","error","-select_streams","v:0","-show_entries","stream=r_frame_rate",
                           "-of","default=noprint_wrappers=1:nokey=1", video_path]).strip()
        fps = None
        if probefps and "/" in probefps:
            a,b = probefps.split("/")
            try:
                a=float(a); b=float(b); fps=(a/b if b!=0 else 0.0)
            except Exception:
                fps=None
        # 宽高
        probesz = _probe([FFPROBE_BIN,"-v","error","-select_streams","v:0","-show_entries","stream=width,height",
                          "-of","csv=s=x:p=0", video_path]).strip()
        W0=H0=0
        if "x" in probesz:
            try:
                W0, H0 = map(int, probesz.split("x"))
            except Exception:
                W0=H0=0

        # 2) 解帧（-vsync 0 避免重复帧 / -q:v 2 高质JPEG）
        # cmd = [
        #     "ffmpeg","-hide_banner","-loglevel","error",
        #     "-i", video_path, "-vsync","0","-q:v","2","-start_number","0", 
        #     os.path.join(tmpdir, "frame_%06d.jpg")
        # ]
        cmd = [
            FFMPEG_BIN,"-hide_banner","-loglevel","error",
            "-i", video_path
        ]
        if target_fps is not None and float(target_fps) > 0:
            cmd += ["-vf", f"fps={float(target_fps)}"]   # <<< [新增] 固定抽帧
        cmd += [
            "-vsync","0","-q:v","2","-start_number","0",
            os.path.join(tmpdir, "frame_%06d.jpg")
        ]

        try:
            subprocess.check_call(cmd)
        except Exception:
            shutil.rmtree(tmpdir, ignore_errors=True)
            raise

        img_paths = sorted(glob.glob(os.path.join(tmpdir, "frame_*.jpg")))
        if not img_paths:
            shutil.rmtree(tmpdir, ignore_errors=True)
            raise ValueError("ffmpeg extracted 0 frames.")

        # 回退保障
        # if not fps or fps <= 0: fps = 30.0
        if target_fps is not None and float(target_fps) > 0:
            fps = float(target_fps)   # <<< [新增] 让后续切 clip 的 fps 也一致
        else:
            if not fps or fps <= 0: fps = 30.0
        if H0 == 0 or W0 == 0:
            # 读首帧补元数据
            im0 = cv2.imread(img_paths[0])
            if im0 is None: 
                shutil.rmtree(tmpdir, ignore_errors=True)
                raise ValueError("Failed to read first extracted frame.")
            H0, W0 = im0.shape[:2]

        return (H0, W0), float(fps), img_paths, tmpdir

    # ---------- 回退：OpenCV + 线程写 ----------
    def _extract_video_to_tmpdir_cv(video_path, target_fps=None, jpeg_quality=90, max_inflight=256):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Error opening video file {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        W0  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H0  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if not fps or fps <= 0 or W0 <= 0 or H0 <= 0:
            # 再尝试读一帧拿尺寸
            ret, frame = cap.read()
            if not ret:
                cap.release()
                raise ValueError(f"Error: Video metadata invalid (fps/W/H) and cannot read first frame: {video_path}")
            H0, W0 = frame.shape[:2]
            fps = 30.0
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        tmpdir = tempfile.mkdtemp()
        # 预估帧数（可能不准，但能给个大概容量）
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0 else None
        img_paths = []

        # 顺序稳定 + 有限 in-flight
        max_workers = max(4, (os.cpu_count() or 8))
        write_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]

        def _write_one(i, fr):
            fp = os.path.join(tmpdir, f"frame_{i:06d}.jpg")
            ok = cv2.imwrite(fp, fr, write_params)
            return fp if ok else None

        futures = []
        use_target = (target_fps is not None and float(target_fps) > 0 and fps and fps > 0)
        if use_target:
            step = float(fps) / float(target_fps)
            if step < 1.0:
                step = 1.0
                target_fps = float(fps)
            next_pick = 0.0
            src_i = 0
        idx = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if use_target:
                    if src_i + 1e-6 < next_pick:
                        src_i += 1
                        continue
                    next_pick += step
                    src_i += 1
                # 限制 in-flight futures，防止占用过多内存
                if len(futures) >= max_inflight:
                    fp = futures.pop(0).result()
                    if fp is not None:
                        img_paths.append(fp)
                futures.append(ex.submit(_write_one, idx, frame.copy()))
                idx += 1
            # 收尾
            for fu in futures:
                fp = fu.result()
                if fp is not None:
                    img_paths.append(fp)

        cap.release()
        if not img_paths:
            shutil.rmtree(tmpdir, ignore_errors=True)
            raise ValueError(f"No frames extracted or all writes failed for {video_path}")

        img_paths.sort()  # 保险：按文件名排序即时间顺序
        fps_out = float(target_fps) if use_target else float(fps)
        return (H0, W0), fps_out, img_paths, tmpdir

    # ---------- 公共切片逻辑 ----------
    def _slice_to_clips(all_imgs, fps, H0, W0, tmpdir, src_kind):
        clips = []
        total_frames = len(all_imgs)
        total_s = total_frames / fps

        if total_s <= min_total_s:
            clips.append(dict(
                img_paths=all_imgs, fps=fps, orig_size=(H0, W0),
                tmpdir=tmpdir, src_kind=src_kind, clip_tag=f"000-{int(round(total_s)):03d}"
            ))
            return clips

        L = int(round(clip_len_s * fps))
        S = int(round(stride_s   * fps))
        if total_frames < L:  # 防止极端小视频
            clips.append(dict(
                img_paths=all_imgs, fps=fps, orig_size=(H0, W0),
                tmpdir=tmpdir, src_kind=src_kind, clip_tag=f"000-{int(round(total_s)):03d}"
            ))
            return clips

        starts = list(range(0, total_frames - L + 1, S))
        for st in starts:
            ed = st + L
            t0 = int(round(st / fps))
            t1 = int(round(ed / fps))
            clips.append(dict(
                img_paths=all_imgs[st:ed],
                fps=fps, orig_size=(H0, W0),
                tmpdir=tmpdir, src_kind=src_kind, clip_tag=f"{t0:03d}-{t1:03d}"
            ))
        if starts:
            last_start_frame = starts[-1]
            next_potential_start = last_start_frame + S
            if next_potential_start < total_frames:
                remaining_frames = total_frames - next_potential_start
                if (remaining_frames / fps) > stride_s:
                    st, ed = next_potential_start, total_frames
                    t0, t1 = int(round(st/fps)), int(round(ed/fps))
                    clips.append(dict(
                        img_paths=all_imgs[st:ed], fps=fps, orig_size=(H0, W0),
                        tmpdir=tmpdir, src_kind=src_kind, clip_tag=f"{t0:03d}-{t1:03d}"
                    ))
        return clips

    # ================= 主流程 =================
    if os.path.isdir(p):
        # 文件夹序列
        H0, W0, all_imgs = _read_first_size_from_folder(p)
        fps = 30.0  # 文件夹默认 30

        # ===== [新增] 可选：把 folder 从 30fps 抽到 target_fps =====
        if target_fps is not None and float(target_fps) > 0:
            tfps = float(target_fps)
            step = fps / tfps  # 30/16 = 1.875
            if step < 1.0:
                step = 1.0   # 不做补帧，只最多到 30
                tfps = fps

            picked = []
            next_pick = 0.0
            i = 0
            for k, path in enumerate(all_imgs):
                if k + 1e-6 >= next_pick:
                    picked.append(path)
                    next_pick += step
            all_imgs = picked
            fps = tfps
        # ===== [新增结束] =====
        
        return _slice_to_clips(all_imgs, fps, H0, W0, None, 'folder')
    else:
        # 视频路径：ffmpeg 抽帧（imageio_ffmpeg 自带二进制，env 自包含）。
        # 历史上这里有 "失败回退 OpenCV" 的注释 + 一行 ipdb.set_trace() 调试残留——
        # 实际并没有 OpenCV fallback；ffmpeg 抽帧失败时直接抛错，避免无声 hang 在 ipdb。
        (H0, W0), fps, all_imgs, tmpdir = _extract_with_ffmpeg(p, target_fps=target_fps)
        return _slice_to_clips(all_imgs, fps, H0, W0, tmpdir, 'video')
