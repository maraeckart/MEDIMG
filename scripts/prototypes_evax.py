from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms

import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.models.evax_module import EVAXModule
from src.data.chexpert_datamodule import CheXpertDataModule, PATHOLOGIES
from src.utils.prototypes import PrototypeExplainer, MisclassificationAnalyzer


def _unnormalize(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    imgs = (tensor * std + mean).clamp(0, 1).permute(0, 2, 3, 1).numpy()
    return [imgs[i] for i in range(len(imgs))]


def extract_features(model, dataloader, device):
    all_features, all_labels, all_masks, all_images = [], [], [], []
    with torch.no_grad():
        for batch in dataloader:
            feats = model.encoder(batch["image"].to(device)).cpu().numpy()
            all_features.append(feats)
            all_labels.append(batch["label"].numpy())
            all_masks.append(batch["mask"].numpy())
            all_images.extend(_unnormalize(batch["image"]))
    return (
        np.concatenate(all_features),
        np.concatenate(all_labels),
        np.concatenate(all_masks),
        all_images,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--data_dir", default="data/CheXpert-v1.0-small")
    parser.add_argument("--misclassified", default="outputs/misclassified.csv")
    parser.add_argument("--output_dir", default="outputs/prototypes")
    parser.add_argument("--n_prototypes", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = EVAXModule.load_from_checkpoint(args.ckpt, map_location=device)
    model.to(device).eval()

    dm = CheXpertDataModule(
        data_dir=args.data_dir,
        uncertainty_policy="u_ones",
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=False,
    )
    dm.setup(stage="fit")

    print("Extracting training features...")
    train_features, train_labels, train_masks, train_images = extract_features(
        model, dm.train_dataloader(), device
    )

    df_wrong = pd.read_csv(args.misclassified)

    val_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    for j, pathology in enumerate(PATHOLOGIES):
        wrong_col = f"{pathology}_wrong"
        true_col  = f"{pathology}_true"
        prob_col  = f"{pathology}_prob"

        if wrong_col not in df_wrong.columns:
            continue

        valid = train_masks[:, j] == 1
        if valid.sum() == 0:
            continue

        explainer = PrototypeExplainer(n_prototypes=args.n_prototypes)
        explainer.fit(
            train_features[valid],
            train_labels[valid, j].astype(int),
            [train_images[i] for i in np.where(valid)[0]],
        )

        wrong_rows = df_wrong[df_wrong[wrong_col] == 1]
        if len(wrong_rows) == 0:
            print(f"{pathology}: no misclassifications, skipping")
            continue

        test_feats, test_true, test_pred, test_imgs = [], [], [], []
        for _, row in wrong_rows.iterrows():
            img = Image.open(row["img_path"]).convert("RGB")
            tensor = val_tf(img).unsqueeze(0).to(device)
            with torch.no_grad():
                feat = model.encoder(tensor).cpu().numpy()[0]
            test_feats.append(feat)
            test_true.append(int(row[true_col]))
            test_pred.append(int(float(row[prob_col]) > 0.5))
            test_imgs.append(np.array(img.resize((224, 224))) / 255.0)

        analyzer = MisclassificationAnalyzer(explainer)
        results  = analyzer.analyze(
            np.array(test_feats),
            np.array(test_true),
            test_imgs,
            np.array(test_pred),
        )

        save_path = output_dir / f"{pathology.replace(' ', '_')}_prototypes.png"
        analyzer.plot(results, save_path=save_path)
        print(f"{pathology}: {len(results)} misclassifications → {save_path}")


if __name__ == "__main__":
    main()
