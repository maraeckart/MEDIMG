from __future__ import annotations

from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import densenet121, DenseNet121_Weights

from src.models.baselines import _CheXpertBase
from src.data.chexpert_datamodule import PATHOLOGIES


class MeanTeacherModule(_CheXpertBase):
    """DenseNet121 Mean Teacher for semi-supervised CheXpert classification.

    This extends the DenseNet121 U-Ignore baseline with:
    - a student model trained with gradients
    - a teacher model updated as EMA of the student
    - a consistency loss between student and teacher predictions

    Validation and testing use the teacher model.
    """

    def __init__(
        self,
        pretrained: bool = True,
        freeze_encoder: bool = False,
        dropout: float = 0.0,
        num_classes: int = 14,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        uncertainty_policy: str = "u_ignore",
        pathologies: list[str] = PATHOLOGIES,
        ema_decay: float = 0.999,
        consistency_weight: float = 1.0,
    ):
        super().__init__(
            num_classes=num_classes,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            uncertainty_policy=uncertainty_policy,
            pathologies=pathologies,
        )

        self.save_hyperparameters(ignore=["pathologies"])

        self.ema_decay = ema_decay
        self.consistency_weight = consistency_weight

        self.student = self._build_densenet(
            pretrained=pretrained,
            freeze_encoder=freeze_encoder,
            dropout=dropout,
            num_classes=num_classes,
        )

        self.teacher = deepcopy(self.student)

        # Teacher is updated manually through EMA, not by backpropagation.
        for param in self.teacher.parameters():
            param.requires_grad = False

    def _build_densenet(
        self,
        pretrained: bool,
        freeze_encoder: bool,
        dropout: float,
        num_classes: int,
    ) -> nn.Module:
        weights = DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = densenet121(weights=weights)

        in_features = backbone.classifier.in_features
        backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )

        if freeze_encoder:
            for name, param in backbone.named_parameters():
                if "classifier" not in name:
                    param.requires_grad = False

        return backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Used by validation_step/test_step inherited from _CheXpertBase.
        # Therefore validation and test are done with the EMA teacher.
        return self.teacher(x)

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        # 1. Unpack the two individual streams passed from CheXpertSSLDataModule
        labeled_batch = batch["labeled"]
        unlabeled_batch = batch["unlabeled"]

        # 2. Extract components for the Supervised Loss (from labeled data)
        lbl_images = labeled_batch["image"]
        targets = labeled_batch["label"]
        mask = labeled_batch["mask"]

        # 3. Extract components for the Unsupervised Consistency Loss
        unlbl_images = unlabeled_batch["image"]

        # 4. Supervised Forward Pass (Student only evaluates labeled images)
        student_labeled_logits = self.student(lbl_images)
        supervised_loss = self._masked_loss(
            logits=student_labeled_logits,
            targets=targets,
            mask=mask,
        )

        # 5. Unsupervised Forward Pass (Both pass the unlabeled images)
        student_unlabeled_logits = self.student(unlbl_images)
        
        self.teacher.eval()
        with torch.no_grad():
            teacher_unlabeled_logits = self.teacher(unlbl_images)

        # 6. Compute Consistency Regularization (on unlabeled data predictions)
        consistency_loss = F.mse_loss(
            torch.sigmoid(student_unlabeled_logits),
            torch.sigmoid(teacher_unlabeled_logits),
        )

        # 7. Total Loss Calculation
        loss = supervised_loss + self.consistency_weight * consistency_loss

        # Update train metrics with student supervised predictions
        self.train_metrics.update(student_labeled_logits.detach(), targets, mask)

        # Logging
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/supervised_loss", supervised_loss, on_step=False, on_epoch=True)
        self.log("train/consistency_loss", consistency_loss, on_step=False, on_epoch=True)

        return loss
    
    def validation_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        # 1. Extract elements from the validation batch dictionary
        images = batch["image"]
        targets = batch["label"]
        mask = batch["mask"]

        # 2. Forward pass through the EMA teacher (handled by your forward method)
        logits = self.forward(images)

        # 3. Calculate paper-accurate U-Ignore loss using the sample mask
        loss = self._masked_loss(logits=logits, targets=targets, mask=mask)

        # 4. Update validation metrics while strictly honoring the U-Ignore policy.
        # This tells TorchMetrics to mask out the -1 and -2 sentinels, 
        # avoiding denominator collapse and realistic tracking.
        self.val_metrics.update(logits, targets, mask)

        # 5. Log the corrected validation loss
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss

    @torch.no_grad()
    def on_train_batch_end(self, outputs, batch, batch_idx) -> None:
        """Update teacher weights after every student update."""
        self._update_teacher()

    @torch.no_grad()
    def _update_teacher(self) -> None:
        """EMA update: teacher = alpha * teacher + (1 - alpha) * student."""
        for teacher_param, student_param in zip(
            self.teacher.parameters(),
            self.student.parameters(),
        ):
            teacher_param.data.mul_(self.ema_decay)
            teacher_param.data.add_(
                student_param.data,
                alpha=1.0 - self.ema_decay,
            )

        # Copy buffers such as BatchNorm running mean/variance.
        for teacher_buffer, student_buffer in zip(
            self.teacher.buffers(),
            self.student.buffers(),
        ):
            teacher_buffer.copy_(student_buffer)

    def configure_optimizers(self):
        # Same idea as DenseNet baseline: lower LR for pretrained backbone,
        # full LR for classifier head.
        head_params = list(self.student.classifier.parameters())
        head_ids = {id(p) for p in head_params}

        backbone_params = [
            p for p in self.student.parameters()
            if id(p) not in head_ids and p.requires_grad
        ]

        head_params = [p for p in head_params if p.requires_grad]

        optimizer = torch.optim.Adam(
            [
                {"params": backbone_params, "lr": self.lr * 0.1},
                {"params": head_params, "lr": self.lr},
            ],
            weight_decay=self.weight_decay,
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs,
            eta_min=1e-7,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/auroc_macro",
            },
        }