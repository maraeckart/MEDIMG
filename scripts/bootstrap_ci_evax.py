from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score
from torch.utils.data import DataLoader
from tqdm import tqdm

import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.models.evax_module import EVAXModule
from src.data.chexpert_datamodule import CheXpertDataModule, PATHOLOGIES


def collect_predictions(ckpt_path: str, data_root: str, batch_size: int = 32) -> tuple:
    """Run inference on test set, return (probs, targets, masks) as numpy arrays."""
    model = EVAXModule.load_from_checkpoint(ckpt_path)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    dm = CheXpertDataModule(data_dir=data_root, batch_size=batch_size)
    dm.setup(stage="test")
    loader = dm.test_dataloader()

    all_probs, all_targets, all_masks = [], [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Inference"):
            images  = batch["image"].to(device)
            targets = batch["label"]
            mask    = batch["mask"]

            logits = model(images)
            probs  = torch.sigmoid(logits).cpu()

            all_probs.append(probs.numpy())
            all_targets.append(targets.numpy())
            all_masks.append(mask.numpy())

    return (
        np.concatenate(all_probs,   axis=0),
        np.concatenate(all_targets, axis=0),
        np.concatenate(all_masks,   axis=0),
    )


def bootstrap_ci(
    probs: np.ndarray,
    targets: np.ndarray,
    masks: np.ndarray,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict:
    """Bootstrap 95% CI for macro AUROC, per-class AUROC, and macro AUPRC."""
    rng = np.random.default_rng(seed)
    n = len(probs)
    lo_pct = (1 - confidence) / 2 * 100
    hi_pct = (1 + confidence) / 2 * 100
    n_classes = probs.shape[1]

    boot_auroc_macro = []
    boot_auprc_macro = []
    boot_auroc_per   = [[] for _ in range(n_classes)]

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        b_probs   = probs[idx]
        b_targets = targets[idx]
        b_masks   = masks[idx]

        class_aurocs, class_auprcs = [], []
        for c in range(n_classes):
            valid = b_masks[:, c].astype(bool)
            if valid.sum() < 2:
                continue
            y_true = b_targets[valid, c]
            y_prob = b_probs[valid, c]
            # need both classes present
            if y_true.sum() == 0 or (1 - y_true).sum() == 0:
                continue
            auroc = roc_auc_score(y_true, y_prob)
            auprc = average_precision_score(y_true, y_prob)
            class_aurocs.append(auroc)
            class_auprcs.append(auprc)
            boot_auroc_per[c].append(auroc)

        if class_aurocs:
            boot_auroc_macro.append(np.mean(class_aurocs))
            boot_auprc_macro.append(np.mean(class_auprcs))

    def ci(vals):
        if not vals:
            return {"mean": None, "ci_low": None, "ci_high": None}
        return {
            "mean":    float(np.mean(vals)),
            "ci_low":  float(np.percentile(vals, lo_pct)),
            "ci_high": float(np.percentile(vals, hi_pct)),
        }

    results = {
        "auroc_macro": ci(boot_auroc_macro),
        "auprc_macro": ci(boot_auprc_macro),
    }
    for c, name in enumerate(PATHOLOGIES):
        safe = name.replace(" ", "_").lower()
        results[f"auroc_{safe}"] = ci(boot_auroc_per[c])

    return results


def print_results(results: dict, confidence: float = 0.95):
    pct = int(confidence * 100)
    print(f"\n{'Metric':<35} {'Mean':>8}  {pct}% CI")
    print("-" * 60)
    for key, val in results.items():
        if val["mean"] is None:
            print(f"{key:<35} {'N/A':>8}")
        else:
            print(f"{key:<35} {val['mean']:>8.4f}  [{val['ci_low']:.4f}, {val['ci_high']:.4f}]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",        required=True,  help="Path to .ckpt file")
    parser.add_argument("--data_root",   required=True,  help="CheXpert data root dir")
    parser.add_argument("--batch_size",  type=int, default=32)
    parser.add_argument("--n_bootstrap", type=int, default=1000)
    parser.add_argument("--output_dir",  default="outputs/bootstrap_ci")
    args = parser.parse_args()

    probs, targets, masks = collect_predictions(args.ckpt, args.data_root, args.batch_size)
    print(f"Collected predictions: {probs.shape[0]} samples, {probs.shape[1]} classes")

    results = bootstrap_ci(probs, targets, masks, n_bootstrap=args.n_bootstrap)
    print_results(results)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "bootstrap_ci_evax.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved → {out / 'bootstrap_ci_evax.json'}")
