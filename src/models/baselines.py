"""
Baseline LightningModules

Baselines:
  1. RandomClassifier  – sanity-check; outputs class-frequency prior
  2. DenseNet121Module – reproduces the Stanford CheXpert baseline
     configurable with U-Ignore or U-Ones uncertainty policy

"""

from __future__ import annotations

import torch
import torch.nn as nn
from lightning import LightningModule
from torchvision.models import densenet121, DenseNet121_Weights

from src.utils.metrics import CheXpertMetrics
from src.data.chexpert_datamodule import PATHOLOGIES

import csv
from pathlib import Path

# Shared LightningModule base

class _CheXpertBase(LightningModule):
    """Shared training / val / test logic."""

    def __init__(
        self,
        num_classes: int = 14,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        uncertainty_policy: str = "u_ignore",  # only used for loss masking
        pathologies: list[str] = PATHOLOGIES,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["pathologies"])

        self.num_classes = num_classes
        self.lr = learning_rate
        self.weight_decay = weight_decay
        self.uncertainty_policy = uncertainty_policy

        # Metrics 
        self.train_metrics = CheXpertMetrics(num_classes, prefix="train/",
                                              pathologies=pathologies)
        self.val_metrics   = CheXpertMetrics(num_classes, prefix="val/",
                                              pathologies=pathologies)
        self.test_metrics  = CheXpertMetrics(num_classes, prefix="test/",
                                              pathologies=pathologies)

        # BCE with logits (handles multi-label naturally)
        # applies sigmoid internally, computes binary cross-entropy for each of the 14 classes independently
        # Reduction=none so we can apply the uncertainty mask per-element
        self.criterion = nn.BCEWithLogitsLoss(reduction="none")

        self.pathologies = pathologies
        self._test_records: list = []


    # Loss with optional mask
    def _masked_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Element-wise BCE masked by the u_ignore policy."""
        loss = self.criterion(logits, targets)  # (B, C)
        if self.uncertainty_policy == "u_ignore":
            loss = loss * mask
            denom = mask.sum().clamp(min=1)
            return loss.sum() / denom
        return loss.mean()

    # Shared step (across train, val, test)
    def _step(self, batch: dict, metrics: CheXpertMetrics) -> torch.Tensor:
        images  = batch["image"]
        targets = batch["label"]
        mask    = batch["mask"]

        logits = self(images) # calss forward
        loss   = self._masked_loss(logits, targets, mask)
        metrics.update(logits.detach(), targets, mask)
        return loss

    # Hooks for lightning
    def training_step(self, batch, batch_idx):
        loss = self._step(batch, self.train_metrics)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def on_train_epoch_end(self):
        self.log_dict(self.train_metrics.compute(), on_epoch=True)
        self.train_metrics.reset()

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch, self.val_metrics)
        self.log("val/loss", loss, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self):
        computed = self.val_metrics.compute()
        self.log_dict(computed, on_epoch=True)
        self.val_metrics.reset()

    def test_step(self, batch, batch_idx):
        images = batch["image"]
        targets = batch["label"]
        mask = batch["mask"]
        paths = batch["path"]
        
        logits = self(images)
        
        loss = self._masked_loss(logits, targets, mask)
        self.log("test/loss", loss, on_epoch=True)

        self.test_metrics.update(logits.detach(), targets, mask)        
        
        probs = torch.sigmoid(logits).detach().cpu()
        for i in range(len(paths)):
            self._test_records.append({
                "path":    paths[i],
                "probs":   probs[i].tolist(),
                "targets": targets[i].cpu().tolist(),
                "mask":    mask[i].cpu().tolist(),
            })

    def on_test_epoch_end(self):

        self.log_dict(self.test_metrics.compute())
        self.test_metrics.reset()

        out_path = Path("outputs/misclassified.csv")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = ["img_path"] + [
            col
            for name in self.pathologies
            for col in (f"{name}_true", f"{name}_prob", f"{name}_wrong")
        ]

        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for rec in self._test_records:
                row = {"img_path": rec["path"]}
                for j, name in enumerate(self.pathologies):
                    true_val = rec["targets"][j]
                    prob     = rec["probs"][j]
                    valid    = bool(rec["mask"][j])
                    row[f"{name}_true"] = int(true_val) if valid else ""
                    row[f"{name}_prob"] = f"{prob:.4f}"
                    row[f"{name}_wrong"] = int((prob > 0.5) != bool(true_val)) if valid else ""
                writer.writerow(row)

        self._test_records.clear()


# Baseline 1 – Random Classifier

class RandomClassifier(_CheXpertBase):
    """Outputs class-frequency priors regardless of the input image.

    Useful to sanity-check that the model is actually learning something.
    The 'prior' is estimated from the training set label frequencies.
    """

    def __init__(
        self,
        num_classes: int = 14,
        pathologies: list[str] = PATHOLOGIES,
        # Class-prior logits – will be updated on first forward pass if None
        class_prior_logits: list[float] | None = None,
    ):
        super().__init__(
            num_classes=num_classes,
            learning_rate=0.0,      # no training needed
            pathologies=pathologies,
        )

        self.automatic_optimization = False
        
        if class_prior_logits is None:
            # Start from equal probability (logit = 0)
            class_prior_logits = [0.0] * num_classes

        self.logit_bias = nn.Parameter(
            torch.tensor(class_prior_logits, dtype=torch.float32),
            requires_grad=False,  # frozen
        )
        # Dummy parameter so LightningModule doesn't complain
        self._dummy = nn.Parameter(torch.zeros(1), requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Broadcast bias to (B, C)
        return self.logit_bias.unsqueeze(0).expand(x.size(0), -1)

    def configure_optimizers(self):
        # No actual training; return a no-op optimizer
        return torch.optim.SGD([self._dummy], lr=0.0)

    def training_step(self, batch, batch_idx):
        loss = self._step(batch, self.train_metrics)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)

    @classmethod
    def from_label_frequencies(
        cls, pos_fractions: list[float], num_classes: int = 14, **kwargs
    ) -> "RandomClassifier":
        """Create from per-class positive-label frequencies.
            converts a probability back to the raw logit value that 
            sigmoid would map to that probability

        Args:
            pos_fractions: List of P(y=1) per class, length == num_classes.
        """
        import math
        logits = [
            math.log(p / (1 - p)) if 0 < p < 1 else 0.0
            for p in pos_fractions
        ]
        return cls(num_classes=num_classes, class_prior_logits=logits, **kwargs)


# Baseline 2 – DenseNet121

class DenseNet121Module(_CheXpertBase):
    """DenseNet121 multi-label classifier (Stanford CheXpert baseline).

    Reproduces the architecture from:
      Irvin et al. "CheXpert: A Large Chest Radiograph Dataset with
      Uncertainty Labels and Expert Comparison." AAAI 2019.

    The final FC layer is replaced with a linear layer of size `num_classes`.
    ImageNet pre-training is used by default.

    Args:
        pretrained:         Load ImageNet weights.
        freeze_encoder:     Freeze DenseNet feature extractor (linear probe). true = only the new head is trained, the backbone is frozen.
        dropout:            Dropout rate before classifier head.
        num_classes:        Number of output pathologies.
        uncertainty_policy: How the DataModule handled uncertain labels.
                            If 'u_ignore', loss masking is applied.
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
    ):
        super().__init__(
            num_classes=num_classes,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            uncertainty_policy=uncertainty_policy,
            pathologies=pathologies,
        )

        weights = DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = densenet121(weights=weights)

        # Replace classifier
        # original DenseNet121 classifier maps 1024 features to 1000 ImageNet classes. 
        # replace it with a linear layer mapping to 14 pathologies.
        in_features = backbone.classifier.in_features
        backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )
        self.model = backbone

        if freeze_encoder:
            for name, param in self.model.named_parameters():
                if "classifier" not in name:
                    param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)  # raw logits (B, 14)

    def configure_optimizers(self):
        # Separate LR for head vs backbone (because backbone already trained on imagenet)
        head_params = list(self.model.classifier.parameters())
        head_ids = {id(p) for p in head_params}
        backbone_params = [p for p in self.model.parameters()
                           if id(p) not in head_ids]

        optimizer = torch.optim.Adam([
            {"params": backbone_params, "lr": self.lr * 0.1},
            {"params": head_params,    "lr": self.lr},
        ], weight_decay=self.weight_decay)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=10, eta_min=1e-7
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/auroc_macro",
            },
        }
