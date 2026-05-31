# Multi-HMR
# Copyright (c) 2024-present NAVER Corp.
# CC BY-NC-SA 4.0 license

import os

import torch
from torch import nn

class Dinov2Backbone(nn.Module):
    def __init__(self, name='dinov2_vitb14', pretrained=False, *args, **kwargs):
        super().__init__()
        self.name = name
        # Force source='local' so torch.hub never touches github.com
        # (no api.github.com trust validation, no source download). Resolves
        # the pre-extracted directory under torch.hub.get_dir(); setup_torch_hub()
        # already guarantees <hub_dir>/facebookresearch_dinov2_main/hubconf.py exists.
        local_repo = os.path.join(torch.hub.get_dir(), "facebookresearch_dinov2_main")
        if not os.path.isfile(os.path.join(local_repo, "hubconf.py")):
            raise FileNotFoundError(
                f"[Dinov2Backbone] dinov2 hub source not found at {local_repo}. "
                "Call Preprocessing.constants.setup_torch_hub() first, and ensure "
                "<TORCH_HUB_DIR>/facebookresearch_dinov2_main/ is pre-extracted."
            )
        self.encoder = torch.hub.load(local_repo, self.name, source='local', pretrained=pretrained)
        self.patch_size = self.encoder.patch_size
        self.embed_dim = self.encoder.embed_dim

    def forward(self, x):
        """
        Encode a RGB image using a ViT-backbone
        Args:
            - x: torch.Tensor of shape [bs,3,w,h]
        Return:
            - y: torch.Tensor of shape [bs,k,d] - image in patchified mode
        """
        assert len(x.shape) == 4
        y = self.encoder.get_intermediate_layers(x)[0] # ViT-L+896x896: [bs,4096,1024] - [bs,nb_patches,emb]
        return y

