"""
CheXpert DataModule for lightning-hydra-template.

Dataset: CheXpert-v1.0-small (Kaggle)
Labels: 1 (positive) | 0 (negative) | -1 → NaN in CSV (uncertain)

"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Literal, Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms


# Constants
PATHOLOGIES = [
    "No Finding",
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Lung Lesion",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
]

NUM_CLASSES = len(PATHOLOGIES)  # 14

UncertaintyPolicy = Literal["u_ignore", "u_ones", "u_zeros"]

class CheXpertDataset(Dataset):
    """Reads images and multi-hot labels from CheXpert CSV files.

    Args:
        csv_path:           Path to train.csv / valid.csv.
        data_root:          Root directory that contains the image paths
                            listed in the CSV (usually the dataset root).
        transform:          torchvision transform pipeline.
        uncertainty_policy: How to handle uncertain labels (-1 / NaN).
        pathologies:        Subset of PATHOLOGIES to use as targets.
    """

    def __init__(
        self,
        csv_path: str | Path,
        data_root: str | Path,
        transform: Optional[Callable] = None,
        uncertainty_policy: UncertaintyPolicy = "u_ignore",
        pathologies: list[str] = PATHOLOGIES,
        frontal_only: bool = True
    ):
        self.data_root = Path(data_root)
        self.transform = transform
        self.uncertainty_policy = uncertainty_policy
        self.pathologies = pathologies

        df = pd.read_csv(csv_path)

        # Normalise column names (strip whitespace)
        df.columns = df.columns.str.strip()

        # filter for frontal view
        if frontal_only:
            df = df[df["Frontal/Lateral"] == "Frontal"]

        # store Image paths 
        self.image_paths: list[Path] = [self.data_root / p for p in df["Path"]]

        # Labels 
        label_df = df[self.pathologies].copy()

        # NaN means "not mentioned in the report" — no information at all.
        # We use -2 as a second sentinel so we can distinguish:
        #   -1  → uncertain mention (u-label)
        #   -2  → not mentioned (NaN)
        label_df = label_df.fillna(-2)


        # Apply uncertainty policy to -1 entries
        if uncertainty_policy == "u_ones":
            label_df = label_df.replace(-1, 1)
        elif uncertainty_policy == "u_zeros":
            label_df = label_df.replace(-1, 0)
        elif uncertainty_policy == "u_ignore":
            # Keep -1 as a sentinel
            pass
        else:
            raise ValueError(f"Unknown uncertainty_policy: {uncertainty_policy}")

        self.labels: np.ndarray = label_df.values.astype(np.float32)


    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> dict:
        img_path = self.image_paths[idx]
        # convert to rgb for densenet and eva
        image = Image.open(img_path).convert("RGB")

        # apply transform pipeline
        if self.transform:
            image = self.transform(image)

        label = torch.tensor(self.labels[idx], dtype=torch.float32)

        # mask=1 means: include this class in the loss (gradient flows).
        # mask=0 means: zero out the loss for this class (no gradient).
        # -2 (NaN/"not mentioned") is always masked — for every policy.
        # -1 (uncertain) is also masked when using u_ignore.
        if self.uncertainty_policy == "u_ignore":
            mask = ((label != -1) & (label != -2)).float()
        else:
            # u_ones / u_zeros: uncertain labels were already remapped above,
            # but "not mentioned" entries (-2) still carry no signal → mask them.
            mask = (label != -2).float()

 
        # Clamp replaces remaining sentinels with 0 so that BCEWithLogitsLoss
        # receives valid targets in [0, 1]. The actual value does not matter
        # since loss contribution is multiplied by 0.
        label = label.clamp(min=0)  

        return {"image": image, "label": label, "mask": mask, "idx": idx}



class CheXpertDataModule(LightningDataModule):
    """
    Lightning DataModule for CheXpert-v1.0-small.

    train.csv  – ~220 000 studies
    valid.csv  – 234 studies (radiologist-labelled, used as test set)

    patient-level validation split out of train.csv

    Args:
        data_dir:            Path to the CheXpert-v1.0-small root directory.
        uncertainty_policy:  "u_ignore" | "u_ones" | "u_zeros"
        val_fraction:        Fraction of patients held out for validation.
        image_size:          Target image resolution (square).
        batch_size:          Mini-batch size.
        num_workers:         DataLoader workers.
        pin_memory:          Pin memory for GPU transfers.
        use_weighted_sampler: Oversample minority classes via WeightedRandomSampler.
        pathologies:         Subset of the 14 pathologies (default: all).
    """

    def __init__(
        self,
        data_dir: str = "data/CheXpert-v1.0-small",
        uncertainty_policy: UncertaintyPolicy = "u_ignore",
        val_fraction: float = 0.1,
        image_size: int = 224,
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True,
        use_weighted_sampler: bool = False,
        pathologies: list[str] = PATHOLOGIES,
        frontal_only: bool = True,

        # Ablation flags — set to False to isolate the effect of each augmentation
        aug_horizontal_flip: bool = True,
        aug_rotation: bool = True,
        aug_color_jitter: bool = True,
    ):
        super().__init__()
        pathologies = list(pathologies)  # convert to list
        self.save_hyperparameters() # for wandb logging

        self.data_dir = Path(data_dir)
        self.uncertainty_policy = uncertainty_policy
        self.val_fraction = val_fraction
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.use_weighted_sampler = use_weighted_sampler
        self.pathologies = pathologies
        self.frontal_only = frontal_only
        self.aug_horizontal_flip = aug_horizontal_flip
        self.aug_rotation = aug_rotation
        self.aug_color_jitter = aug_color_jitter

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def _train_transforms(self) -> transforms.Compose:
        # Build augmentation pipeline conditionally so individual augmentations
        # can be toggled via config for ablation studies.
        # Base pipeline (always applied):
        pipeline = [transforms.Resize((self.image_size, self.image_size))]
 
        if self.aug_horizontal_flip:
            pipeline.append(transforms.RandomHorizontalFlip())
        if self.aug_rotation:
            pipeline.append(transforms.RandomRotation(10))
        if self.aug_color_jitter:
            pipeline.append(transforms.ColorJitter(brightness=0.2, contrast=0.2))
 
        # Always applied after augmentations:
        pipeline += [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ]
        return transforms.Compose(pipeline)
    
    # for validation set
    def _val_transforms(self) -> transforms.Compose:
        return transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    # called by trainer right before training or testing
    def setup(self, stage: Optional[str] = None) -> None:
        train_csv = self.data_dir / "train.csv"
        valid_csv = self.data_dir / "valid.csv"

        df_train_full = pd.read_csv(train_csv)
        df_train_full.columns = df_train_full.columns.str.strip()

        # Patient-level train/val split
        # Extract patient ID from path
        # to know which rows in the csv belong to the same patient
        df_train_full["patient_id"] = (
            df_train_full["Path"]
            .str.split("/")
            .str[2]  # index depends on path depth; adjust if needed
        )

        # same patient data never splits to train and val
        all_patients = df_train_full["patient_id"].unique()
        rng = np.random.default_rng(42) # keeps the patient split identical across all experiment runs
        rng.shuffle(all_patients)
        n_val = max(1, int(len(all_patients) * self.val_fraction))
        val_patients = set(all_patients[:n_val])

        df_val = df_train_full[df_train_full["patient_id"].isin(val_patients)]
        df_train = df_train_full[~df_train_full["patient_id"].isin(val_patients)]

        # Write temporary CSVs
        _tmp = self.data_dir / "_splits"
        _tmp.mkdir(exist_ok=True)
        df_train.to_csv(_tmp / "train_split.csv", index=False)
        df_val.to_csv(_tmp / "val_split.csv", index=False)

        if stage in ("fit", None):
            self.train_dataset = CheXpertDataset(
                csv_path=_tmp / "train_split.csv",
                data_root=self.data_dir,
                transform=self._train_transforms(),
                uncertainty_policy=self.uncertainty_policy,
                pathologies=self.pathologies,
                frontal_only=self.frontal_only
            )
            self.val_dataset = CheXpertDataset(
                csv_path=_tmp / "val_split.csv",
                data_root=self.data_dir,
                transform=self._val_transforms(),
                uncertainty_policy=self.uncertainty_policy,
                pathologies=self.pathologies,
                frontal_only=self.frontal_only
            )

        if stage in ("test", None):
            # Official validation set (radiologist labels) → our test set
            self.test_dataset = CheXpertDataset(
                csv_path=valid_csv,
                data_root=self.data_dir,
                transform=self._val_transforms(),
                uncertainty_policy="u_ones",  # always use hard labels for test
                pathologies=self.pathologies,
                frontal_only=self.frontal_only
            )

    # DataLoaders 

    # turn on if rare classes stagnating in per-class AUROC.
    def _make_sampler(self, dataset: CheXpertDataset) -> Optional[WeightedRandomSampler]:
        """Compute per-sample weights to balance positive/negative ratio."""
        if not self.use_weighted_sampler:
            return None
        labels = dataset.labels  # (N, 14); already has -1 → 0
        pos_counts = (labels == 1).sum(axis=0)  # (14,)
        neg_counts = (labels == 0).sum(axis=0)
        # weight for class c = 1/pos_c if positive, 1/neg_c if negative
        # aggregate over all classes by summing weights
        pos_weight = np.where(pos_counts > 0, 1.0 / pos_counts, 0)
        neg_weight = np.where(neg_counts > 0, 1.0 / neg_counts, 0)
        sample_weights = (
            (labels == 1) * pos_weight + (labels == 0) * neg_weight
        ).sum(axis=1)
        return WeightedRandomSampler(
            weights=torch.tensor(sample_weights, dtype=torch.float),
            num_samples=len(dataset),
            replacement=True,
        )

    def train_dataloader(self) -> DataLoader:
        sampler = self._make_sampler(self.train_dataset)
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=(sampler is None),
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    # Helpers
    @property
    def num_classes(self) -> int:
        return len(self.pathologies)

    def __repr__(self) -> str:
        return (
            f"CheXpertDataModule("
            f"policy={self.uncertainty_policy}, "
            f"val_frac={self.val_fraction}, "
            f"img_size={self.image_size}, "
            f"batch={self.batch_size})"
        )