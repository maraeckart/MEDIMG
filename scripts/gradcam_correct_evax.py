from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
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
from src.data.chexpert_datamodule import PATHOLOGIES


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--misclassified", default="outputs/misclassified.csv")
    parser.add_argument("--output_dir", default="outputs/gradcam_correct")
    parser.add_argument("--n_samples", type=int, default=30)
    parser.add_argument("--data_dir", default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = EVAXModule.load_from_checkpoint(args.ckpt, map_location=device)
    model.to(device).eval()

    target_layers = [model.encoder.blocks[-1].norm1]
    cam = GradCAM(model=model, target_layers=target_layers, reshape_transform=reshape_transform)

    val_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    df = pd.read_csv(args.misclassified)

    if args.data_dir:
        df["img_path"] = df["img_path"].str.replace(
            "/home/jovyan/MEDIMG/data/CheXpert-v1.0-small", args.data_dir, regex=False
        )


    for j, pathology in enumerate(PATHOLOGIES):
        wrong_col = f"{pathology}_wrong"
        true_col  = f"{pathology}_true"

        if wrong_col not in df.columns:
            continue

        correct_pos = df[(df[wrong_col] == 0) & (df[true_col] == 1)]
        if len(correct_pos) == 0:
            print(f"{pathology}: no correctly classified positives, skipping")
            continue

        samples = correct_pos.sample(n=min(args.n_samples, len(correct_pos)), random_state=42)
        n_show  = len(samples)

        fig, axes = plt.subplots(n_show, 2, figsize=(6, n_show * 3))
        if n_show == 1:
            axes = axes[np.newaxis, :]

        fig.suptitle(f"{pathology} — correctly classified (true positive)\nLeft: original  |  Right: GradCAM", fontsize=10, y=1.01)

        for i, (_, row) in enumerate(samples.iterrows()):
            img    = Image.open(row["img_path"]).convert("RGB")
            rgb_np = np.array(img.resize((224, 224))) / 255.0
            tensor = val_tf(img).unsqueeze(0).to(device)

            grayscale = cam(input_tensor=tensor, targets=[ClassifierOutputTarget(j)])[0]
            overlay   = show_cam_on_image(rgb_np.astype(np.float32), grayscale, use_rgb=True)

            axes[i, 0].imshow(rgb_np)
            axes[i, 0].set_title("Original", fontsize=7)
            axes[i, 0].axis("off")

            axes[i, 1].imshow(overlay)
            axes[i, 1].set_title("GradCAM", fontsize=7)
            axes[i, 1].axis("off")

        plt.tight_layout()
        save_path = output_dir / f"{pathology.replace(' ', '_')}_correct.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"{pathology}: {n_show} samples → {save_path}")


if __name__ == "__main__":
    main()
