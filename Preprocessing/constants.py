"""Preprocessing 模块常量与路径。

代码与小数据文件全部 vendor 在 ./third_party/ 下；大模型 ckpt（human3r.pth、
DA3、SMPL/SMPLX body model、DINOv2 hub 源码）统一放在 ``PREPROCESS_ROOT`` 下，
默认指向训练好的合并模型同级目录的 ``preprocess/`` 子目录。各子路径可单独用环境
变量覆盖。

PREPROCESS_ROOT 期望布局（推荐用 symlink 指向 BOS 真实文件）::

    <PREPROCESS_ROOT>/
        human3r.pth                       # HUMAN3R_CKPT  (4.4G)
        DA3NESTED-GIANT-LARGE-1.1/        # DA3_CKPT_DIR  (6.3G)
        smpl_models/                      # SMPL_MODELS_DIR (3.2G, 含 smpl/ smplx/)
        torch_hub/
            facebookresearch_dinov2_main/ # 由 dinov2_pack.tar.gz 解压而来

setup_torch_hub() 会把 torch.hub 缓存目录指向 ``<PREPROCESS_ROOT>/torch_hub``，
让 ``torch.hub.load('facebookresearch/dinov2', ...)`` 完全离线工作。
"""
from __future__ import annotations

import os

# 本目录 / vendor 出来的第三方代码目录（git tracked）
PREPROC_DIR = os.path.dirname(os.path.abspath(__file__))
THIRD_PARTY_DIR = os.path.join(PREPROC_DIR, "third_party")

# ---------------------------------------------------------------------------
# 一站式预处理资产根目录
#
# 优先级：
#   1. 环境变量 PREPROCESS_ROOT 直接指定 preprocess/ 目录
#   2. 否则从 ONESHOT_MODEL_DIR/preprocess 派生（与 inference 脚本共享同一根）
#   3. 否则用 BOS 默认路径
#
# 用户在本地下载了 ONESHOT-14B-diffusers/ 后，只需 ::
#     export ONESHOT_MODEL_DIR=/path/to/ONESHOT-14B-diffusers
# 即可同时切换 inference 和 preprocessing 的资产路径。
# ---------------------------------------------------------------------------
_DEFAULT_ONESHOT_MODEL_DIR = (
    "/root/paddlejob/bosdata/yangfengyuan/PretrainModels/Wan/"
    "ONESHOT-14B-diffusers"
)
ONESHOT_MODEL_DIR = os.environ.get("ONESHOT_MODEL_DIR", _DEFAULT_ONESHOT_MODEL_DIR)
PREPROCESS_ROOT = os.environ.get(
    "PREPROCESS_ROOT",
    os.path.join(ONESHOT_MODEL_DIR, "preprocess"),
)

# Human3R 主 ckpt（4.4GB）
HUMAN3R_CKPT = os.environ.get(
    "HUMAN3R_CKPT",
    os.path.join(PREPROCESS_ROOT, "human3r.pth"),
)

# DA3 (Depth-Anything-3) 模型目录
DA3_CKPT_DIR = os.environ.get(
    "DA3_CKPT_DIR",
    os.path.join(PREPROCESS_ROOT, "DA3NESTED-GIANT-LARGE-1.1"),
)

# SMPL/SMPLX body model 目录（含 smpl/ 与 smplx/ 子目录，3.2GB）。
# dust3r/smpl_model.py 期望 third_party/models/{smpl,smplx} 存在；
# ensure_smpl_symlinks() 在运行时把它们 symlink 到这里。
SMPL_MODELS_DIR = os.environ.get(
    "SMPL_MODELS_DIR",
    os.path.join(PREPROCESS_ROOT, "smpl_models"),
)

# torch.hub 缓存目录；预解压的 facebookresearch_dinov2_main/ 必须在它下面
TORCH_HUB_DIR = os.environ.get(
    "TORCH_HUB_DIR",
    os.path.join(PREPROCESS_ROOT, "torch_hub"),
)


def ensure_smpl_symlinks() -> None:
    """确保 third_party/models/{smpl,smplx} symlink 指向 SMPL_MODELS_DIR。

    幂等且对并发安全：多个 preprocess 进程同时调用时，TOCTOU race 不会抛错。
    """
    target_dir = os.path.join(THIRD_PARTY_DIR, "models")
    os.makedirs(target_dir, exist_ok=True)
    for name in ("smpl", "smplx"):
        link = os.path.join(target_dir, name)
        src = os.path.join(SMPL_MODELS_DIR, name)
        if os.path.islink(link):
            try:
                if os.readlink(link) == src:
                    continue
                os.unlink(link)
            except FileNotFoundError:
                pass  # 别的进程刚刚清掉了，继续走 symlink
        elif os.path.exists(link):
            continue  # 真目录已存在，不动
        if not os.path.exists(src):
            continue
        try:
            os.symlink(src, link)
        except FileExistsError:
            # 并发：另一进程刚刚建好。校验一下指向是否一致，不一致才报。
            if os.path.islink(link) and os.readlink(link) == src:
                continue
            raise


def setup_torch_hub() -> None:
    """把 torch.hub 缓存目录指向 ``TORCH_HUB_DIR``，使 dinov2 等离线加载。

    要求 ``<TORCH_HUB_DIR>/facebookresearch_dinov2_main/`` 已经预先解压好（由
    ``dinov2_pack.tar.gz`` 直接解压得到）。必须在第一次 ``torch.hub.load`` 之前调用。

    Fail-fast：目录或 hubconf.py 缺失时直接抛错，避免 mhmr/blocks/dinov2.py 里
    ``torch.hub.load('facebookresearch/dinov2', ...)`` 落到联网下载/校验分支
    （内网无外网时会无声 hang）。
    """
    import torch
    os.makedirs(TORCH_HUB_DIR, exist_ok=True)
    torch.hub.set_dir(TORCH_HUB_DIR)

    dinov2_dir = os.path.join(TORCH_HUB_DIR, "facebookresearch_dinov2_main")
    hubconf = os.path.join(dinov2_dir, "hubconf.py")
    if not os.path.isfile(hubconf):
        raise FileNotFoundError(
            "[setup_torch_hub] dinov2 hub source missing: "
            f"{hubconf} not found.\n"
            f"  TORCH_HUB_DIR     = {TORCH_HUB_DIR}\n"
            f"  PREPROCESS_ROOT   = {PREPROCESS_ROOT}\n"
            f"  ONESHOT_MODEL_DIR = {ONESHOT_MODEL_DIR}\n"
            "Expected layout: <TORCH_HUB_DIR>/facebookresearch_dinov2_main/hubconf.py\n"
            "Fix: extract dinov2_pack.tar.gz into <TORCH_HUB_DIR>/ "
            "(produces facebookresearch_dinov2_main/), or point ONESHOT_MODEL_DIR / "
            "PREPROCESS_ROOT / TORCH_HUB_DIR to a directory that already contains it. "
            "Refusing to proceed because torch.hub.load would otherwise try to fetch "
            "from github.com and hang on offline machines."
        )

    # 顺便写好 trusted_list，跳过 torch.hub 的 api.github.com 校验
    trusted_list = os.path.join(TORCH_HUB_DIR, "trusted_list")
    try:
        existing = ""
        if os.path.isfile(trusted_list):
            with open(trusted_list, "r") as f:
                existing = f.read()
        if "facebookresearch/dinov2" not in existing:
            with open(trusted_list, "a") as f:
                f.write("facebookresearch/dinov2\n")
    except OSError:
        pass  # 只读 BOS 挂载等场景下写不进，忽略；hubconf 已存在不会触发联网下载


# 切 clip 默认参数
DEFAULT_CLIP_LEN_S = 6
DEFAULT_STRIDE_S = 5
DEFAULT_MIN_TOTAL_S = 6  # < 这个长度不切，整段当一个 clip
