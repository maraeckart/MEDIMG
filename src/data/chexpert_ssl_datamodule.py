# src/data/chexpert_ssl_datamodule.py

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from src.data.chexpert_datamodule import (
    CheXpertDataModule,
    CheXpertDataset,
    PATHOLOGIES,
    UncertaintyPolicy,
)


class CheXpertSSLDataModule(CheXpertDataModule):
    """Semi-supervised CheXpert datamodule for Mean Teacher training.

    Returns two train loaders:
    - labeled: used for supervised BCE loss
    - unlabeled: used for student-teacher consistency loss
    """

    def __init__(
        self,
        data_dir: str = "data",
        uncertainty_policy: UncertaintyPolicy = "u_ignore",
        val_fraction: float = 0.1,
        labeled_fraction: float = 0.2,
        image_size: int = 224,
        batch_size_labeled: int = 16,
        batch_size_unlabeled: int = 16,
        num_workers: int = 4,
        pin_memory: bool = True,
        pathologies: list[str] = PATHOLOGIES,
        frontal_only: bool = True,
        aug_horizontal_flip: bool = True,
        aug_rotation: bool = True,
        aug_color_jitter: bool = True,
        aug_gaussian_blur: bool = False,
        seed: int = 42,
    ):
        super().__init__(
            data_dir=data_dir,
            uncertainty_policy=uncertainty_policy,
            val_fraction=val_fraction,
            image_size=image_size,
            batch_size=batch_size_labeled + batch_size_unlabeled,
            num_workers=num_workers,
            pin_memory=pin_memory,
            use_weighted_sampler=False,
            pathologies=pathologies,
            frontal_only=frontal_only,
            aug_horizontal_flip=aug_horizontal_flip,
            aug_rotation=aug_rotation,
            aug_color_jitter=aug_color_jitter,
            aug_gaussian_blur=aug_gaussian_blur
        )

        self.labeled_fraction = labeled_fraction
        self.batch_size_labeled = batch_size_labeled
        self.batch_size_unlabeled = batch_size_unlabeled
        self.seed = seed

        self.labeled_dataset = None
        self.unlabeled_dataset = None

    def setup(self, stage: Optional[str] = None) -> None:
        train_csv = self.data_dir / "train.csv"
        valid_csv = self.data_dir / "valid.csv"

        df_train_full = pd.read_csv(train_csv)
        df_train_full.columns = df_train_full.columns.str.strip()

        if self.frontal_only:
            df_train_full = df_train_full[df_train_full["Frontal/Lateral"] == "Frontal"]

        df_train_full["patient_id"] = (
            df_train_full["Path"]
            .str.split("/")
            .str[2]
        )

        rng = np.random.default_rng(self.seed)

        all_patients = df_train_full["patient_id"].unique()
        rng.shuffle(all_patients)

        n_val = max(1, int(len(all_patients) * self.val_fraction))
        val_patients = set(all_patients[:n_val])
        train_patients = np.array(all_patients[n_val:])

        df_val = df_train_full[df_train_full["patient_id"].isin(val_patients)]
        df_train = df_train_full[df_train_full["patient_id"].isin(train_patients)]

        # split remaining train patients into labelled and unlabelled patients
        rng.shuffle(train_patients)
        n_labeled = max(1, int(len(train_patients) * self.labeled_fraction))
        labeled_patients = set(train_patients[:n_labeled])

        df_labeled = df_train[df_train["patient_id"].isin(labeled_patients)]
        df_unlabeled = df_train[~df_train["patient_id"].isin(labeled_patients)]

        split_dir = self.data_dir / "_splits"
        split_dir.mkdir(exist_ok=True)

        labeled_csv = split_dir / "ssl_labeled_split.csv"
        unlabeled_csv = split_dir / "ssl_unlabeled_split.csv"
        val_csv = split_dir / "ssl_val_split.csv"

        df_labeled.to_csv(labeled_csv, index=False)
        df_unlabeled.to_csv(unlabeled_csv, index=False)
        df_val.to_csv(val_csv, index=False)

        if stage in ("fit", None):
            self.labeled_dataset = CheXpertDataset(
                csv_path=labeled_csv,
                data_root=self.data_dir,
                transform=self._train_transforms(),
                uncertainty_policy=self.uncertainty_policy,
                pathologies=self.pathologies,
                frontal_only=False,  # already filtered above
            )

            self.unlabeled_dataset = CheXpertDataset(
                csv_path=unlabeled_csv,
                data_root=self.data_dir,
                transform=self._train_transforms(),
                uncertainty_policy=self.uncertainty_policy,
                pathologies=self.pathologies,
                frontal_only=False,
            )

            self.val_dataset = CheXpertDataset(
                csv_path=val_csv,
                data_root=self.data_dir,
                transform=self._val_transforms(),
                uncertainty_policy=self.uncertainty_policy,
                pathologies=self.pathologies,
                frontal_only=False,
            )

        if stage in ("test", None):
            self.test_dataset = CheXpertDataset(
                csv_path=valid_csv,
                data_root=self.data_dir,
                transform=self._val_transforms(),
                uncertainty_policy="u_ones",
                pathologies=self.pathologies,
                frontal_only=self.frontal_only,
            )

    def train_dataloader(self):
        labeled_loader = DataLoader(
            self.labeled_dataset,
            batch_size=self.batch_size_labeled,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
        )

        unlabeled_loader = DataLoader(
            self.unlabeled_dataset,
            batch_size=self.batch_size_unlabeled,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
        )

        return {
            "labeled": labeled_loader,
            "unlabeled": unlabeled_loader,
        }