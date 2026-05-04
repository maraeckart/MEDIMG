"""
Evaluation metrics

Metrics implemented (all per-class + macro averages):
  • AUROC              – primary metric (matches original Stanford study)
  • Average Precision  – area under PR curve
  • F1 @ threshold     – default 0.5
  • Sensitivity / Specificity @ threshold
  • Calibration (ECE)  – reliability of probability outputs

Usage in a LightningModule::

    from src.utils.metrics import CheXpertMetrics

    self.train_metrics = CheXpertMetrics(num_classes=14, prefix="train/")
    self.val_metrics   = CheXpertMetrics(num_classes=14, prefix="val/")

    # in validation_step:
    self.val_metrics.update(preds, targets)

    # in on_validation_epoch_end:
    self.log_dict(self.val_metrics.compute())
    self.val_metrics.reset()
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torchmetrics import (
    AUROC,
    AveragePrecision,
    CalibrationError,
    F1Score,
    MetricCollection,
    Specificity,
)
from torchmetrics.classification import MultilabelAUROC, MultilabelAveragePrecision

from src.data.chexpert_datamodule import PATHOLOGIES

# any sub-metric registered as an attribute gets automatically moved to the right device
class CheXpertMetrics(nn.Module):
    """
    Wraps torchmetrics into a per-class + macro collection.

    Args:
        num_classes: Number of target pathologies (default 14).
        threshold:   Decision threshold for binary metrics.
        prefix:      String prefix added to every logged key.
        pathologies: Names of pathologies (for per-class logging).
    """

    def __init__(
        self,
        num_classes: int = 14,
        threshold: float = 0.5,
        prefix: str = "",
        pathologies: list[str] = PATHOLOGIES,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.threshold = threshold
        self.prefix = prefix
        self.pathologies = pathologies

        # Macro metrics (scalar)
        # average="macro": compute the metric independently for each class, take unweighted mean. 
        self.macro_collection = MetricCollection(
            {
                "auroc_macro": MultilabelAUROC(
                    num_labels=num_classes, average="macro", thresholds=None
                ),
                "auprc_macro": MultilabelAveragePrecision(
                    num_labels=num_classes, average="macro", thresholds=None
                ),
                "f1_macro": F1Score(
                    task="multilabel",
                    num_labels=num_classes,
                    average="macro",
                    threshold=threshold,
                ),
                "specificity_macro": Specificity(
                    task="multilabel",
                    num_labels=num_classes,
                    average="macro",
                    threshold=threshold,
                ),
            },
            prefix=prefix,
        )

        # Per-class AUROC (14 scalars)
        self.perclass_auroc = MultilabelAUROC(
            num_labels=num_classes, average="none", thresholds=None
        )

        # Calibration (ECE: (Expected Calibration Error)) 
        # torchmetrics ECE for multilabel: evaluate per-class average
        # measures whether predicted probabilities are trustworthy
        self._ece_metrics = nn.ModuleList(
            [CalibrationError(task="binary", n_bins=15) for _ in range(num_classes)]
        )

    #
    def update(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> None:
        """Accumulate predictions.

        Args:
            preds:   Raw logits or probabilities, shape (B, C).
            targets: Binary ground truth, shape (B, C).
            mask:    Optional (B, C) float tensor; 0 means ignore this
                     sample/class combination (u_ignore policy).
        """
        # Convert logits → probabilities if necessary
        if preds.min() < 0 or preds.max() > 1:
            probs = torch.sigmoid(preds)
        else:
            probs = preds

        # torchmetrics multilabel metrics want integer targets
        targets_int = targets.long()

        # Apply mask: replace masked entries with a neutral value so they
        # contribute minimally.
        self.macro_collection.update(probs, targets_int)
        self.perclass_auroc.update(probs, targets_int)

        # Per-class ECE
        for c in range(self.num_classes):
            if mask is not None:
                valid = mask[:, c].bool()
                if valid.sum() == 0:
                    continue
                self._ece_metrics[c].update(probs[valid, c], targets_int[valid, c])
            else:
                self._ece_metrics[c].update(probs[:, c], targets_int[:, c])

    # called once at epoch end, takes everything that was accumulated in update 
    # across all batches and calculates the final metric values.
    def compute(self) -> dict[str, torch.Tensor]:
        results: dict[str, torch.Tensor] = {}

        # Macro scalars
        results.update(self.macro_collection.compute())

        # Per-class AUROC
        per_class_auroc = self.perclass_auroc.compute()  # (14,)
        for i, name in enumerate(self.pathologies[: self.num_classes]):
            safe_name = name.replace(" ", "_").lower()
            results[f"{self.prefix}auroc_{safe_name}"] = per_class_auroc[i]

        # ECE (mean over classes)
        ece_vals = []
        for c in range(self.num_classes):
            try:
                ece_vals.append(self._ece_metrics[c].compute())
            except Exception:
                pass
        if ece_vals:
            results[f"{self.prefix}ece_mean"] = torch.stack(ece_vals).mean()

        return results

    # called after logging, clears accumulated predictions
    def reset(self) -> None:
        self.macro_collection.reset()
        self.perclass_auroc.reset()
        for m in self._ece_metrics:
            m.reset()

    
    def forward(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        self.update(preds, targets, mask)
        return self.compute()
