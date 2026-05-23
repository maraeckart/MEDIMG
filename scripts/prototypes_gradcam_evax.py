from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torchvision import transforms

import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

sys.path.insert(0, str(Path(__file__).parent))
from gradcam_evax import reshape_transform

from src.models.evax_module import EVAXModule
from src.data.chexpert_datamodule import CheXpertDataModule, PATHOLOGIES
from src.utils.prototypes_gradcam import PrototypeExplainer, MisclassificationAnalyzer


def extract_features(model, dataloader, device):
    all_features, all_labels, all_masks, all_paths = [], [], [], []
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader, 1):
            if batch_idx % 100 == 0:
                print(f"  batch {batch_idx}/{len(dataloader)}", flush=True)
            feats = model.encoder(batch["image"].to(device)).cpu().numpy()
            all_features.append(feats)
            all_labels.append(batch["label"].numpy())
            all_masks.append(batch["mask"].numpy())
            all_paths.extend(batch["path"])
    return (
        np.concatenate(all_features),
        np.concatenate(all_labels),
        np.concatenate(all_masks),
        all_paths,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",          required=True)
    parser.add_argument("--data_dir",      default="data/CheXpert-v1.0-small")
    parser.add_argument("--misclassified", default="outputs/misclassified.csv")
    parser.add_argument("--output_dir",    default="outputs/prototypes_gradcam")
    parser.add_argument("--n_prototypes",  type=int, default=3)
    parser.add_argument("--batch_size",    type=int, default=64)
    parser.add_argument("--num_workers",   type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = EVAXModule.load_from_checkpoint(args.ckpt, map_location=device)
    model.to(device).eval()

    target_layers = [model.encoder.blocks[-1].norm1]
    cam = GradCAM(model=model, target_layers=target_layers, reshape_transform=reshape_transform)

    dm = CheXpertDataModule(
        data_dir=args.data_dir,
        uncertainty_policy="u_ones",
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=False,
    )
    dm.setup(stage="fit")

    print("Extracting training features...")
    train_features, train_labels, train_masks, train_paths = extract_features(
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

        idx_pos = np.where(valid & (train_labels[:, j] == 1))[0]
        idx_neg = np.where(valid & (train_labels[:, j] == 0))[0]

        if len(idx_pos) == 0 or len(idx_neg) == 0:
            print(f"{pathology}: skipping — one class has no training samples")
            continue

        rng = np.random.default_rng(42)
        idx_pos = rng.choice(idx_pos, size=min(2500, len(idx_pos)), replace=False) if len(idx_pos) > 2500 else idx_pos
        idx_neg = rng.choice(idx_neg, size=min(2500, len(idx_neg)), replace=False) if len(idx_neg) > 2500 else idx_neg
        idx = np.concatenate([idx_pos, idx_neg])

        proto_images = []
        for i in idx:
            img = Image.open(train_paths[i]).convert("RGB")
            proto_images.append(np.array(img.resize((224, 224))) / 255.0)

        explainer = PrototypeExplainer(n_prototypes=args.n_prototypes)
        explainer.fit(
            train_features[idx],
            train_labels[idx, j].astype(int),
            proto_images,
        )

        wrong_rows = df_wrong[df_wrong[wrong_col] == 1]
        if len(wrong_rows) == 0:
            print(f"{pathology}: no misclassifications, skipping")
            continue

        test_feats, test_true, test_pred, test_imgs, gradcam_overlays = [], [], [], [], []
        for _, row in wrong_rows.iterrows():
            img    = Image.open(row["img_path"]).convert("RGB")
            rgb_np = np.array(img.resize((224, 224))) / 255.0
            tensor = val_tf(img).unsqueeze(0).to(device)

            with torch.no_grad():
                feat = model.encoder(tensor).cpu().numpy()[0]

            grayscale = cam(input_tensor=tensor, targets=[ClassifierOutputTarget(j)])[0]
            overlay   = show_cam_on_image(rgb_np.astype(np.float32), grayscale, use_rgb=True)

            test_feats.append(feat)
            test_true.append(int(row[true_col]))
            test_pred.append(int(float(row[prob_col]) > 0.5))
            test_imgs.append(rgb_np)
            gradcam_overlays.append(overlay)

        analyzer = MisclassificationAnalyzer(explainer)
        results  = analyzer.analyze(
            np.array(test_feats),
            np.array(test_true),
            test_imgs,
            np.array(test_pred),
        )

        for result, overlay in zip(results, gradcam_overlays):
            result["gradcam_overlay"] = overlay

        save_path = output_dir / f"{pathology.replace(' ', '_')}_prototypes.png"
        analyzer.plot(results, save_path=save_path)
        print(f"{pathology}: {len(results)} misclassifications → {save_path}")


if __name__ == "__main__":
    main()
