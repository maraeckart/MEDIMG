from __future__ import annotations

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download

from src.models.baselines import _CheXpertBase
from src.data.chexpert_datamodule import PATHOLOGIES
from src.models.eva_x import EVA_X, checkpoint_filter_fn


class EVAXModule(_CheXpertBase):
    """EVA-X fine-tuned on CheXpert (U-Ones policy).

    Backbone pretrained on 520k chest X-rays via masked image modelling.
    """

    def __init__(
        self,
        pretrained: bool = True,
        freeze_encoder: bool = False,
        dropout: float = 0.0,
        num_classes: int = 14,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.05,
        uncertainty_policy: str = "u_ones",
        pathologies: list[str] = PATHOLOGIES,
    ):
        super().__init__(
            num_classes=num_classes,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            uncertainty_policy=uncertainty_policy,
            pathologies=pathologies,
        )

        self.encoder = EVA_X(
            img_size=224,
            patch_size=16,
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4 * 2 / 3,
            swiglu_mlp=True,
            use_rot_pos_emb=True,
            ref_feat_shape=(14, 14),
            num_classes=0,
            global_pool="avg",
            qkv_fused=False,
            scale_mlp=True
        )

        if pretrained:
            ckpt_path = hf_hub_download(
                repo_id="MapleF/eva_x",
                filename="eva_x_base_patch16_merged520k_mim.pt",
            )
            state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            state_dict = checkpoint_filter_fn(state_dict, self.encoder)
            self.encoder.load_state_dict(state_dict, strict=False)

        self.head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(768, num_classes),
        )

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)  # (B, 384)
        return self.head(features)  # (B, 14)

    def configure_optimizers(self):
        head_params = list(self.head.parameters())
        head_ids = {id(p) for p in head_params}
        backbone_params = [p for p in self.encoder.parameters()
                           if id(p) not in head_ids]

        optimizer = torch.optim.AdamW([
            {"params": backbone_params, "lr": self.lr * 0.1},
            {"params": head_params,     "lr": self.lr},
        ], weight_decay=self.weight_decay)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.trainer.max_epochs, eta_min=1e-7
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/auroc_macro",
            },
        }
