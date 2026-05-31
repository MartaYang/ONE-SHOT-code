"""Compatibility shim used by `preprocess_video.py`.

历史上需要把 Human3R_DA3/src（与 ckpt 同目录）插到 sys.path，让 `import dust3r`
落到 BOS 上的代码副本。现在 `dust3r/` 与 `croco/` 已 vendor 在
`Preprocessing/third_party/`（已被加入 sys.path），所以默认不再插入外部路径。

如需回退到旧行为（例如本地修改了 BOS 上的 dust3r 代码想直接跑），可设置环境变量
`PREPROC_USE_EXTERNAL_DUST3R=1`，此时仍会按 ckpt dirname 插 sys.path。
"""
import os
import sys


def add_path_to_dust3r(ckpt: str) -> None:
    if os.environ.get("PREPROC_USE_EXTERNAL_DUST3R") == "1":
        sys.path.insert(0, os.path.dirname(os.path.abspath(ckpt)))
