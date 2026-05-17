from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM, EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torchvision import transforms

import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.models.evax_module import EVAXModule
from src.data.chexpert_datamodule import PATHOLOGIES


def reshape_transform(tensor, height=14, width=14):
    """Reshape ViT token activations → spatial feature map for Grad-CAM.
    
    EVA-X with 224px and patch_size=16 → 14×14 = 196 patch tokens.
    Input:  (B, 197, 384)
    Output: (B, 384, 14, 14)
    """
    result = tensor[:, 1:, :]  # skip CLS → (B, 196, 384)
    result = result.reshape(result.size(0), height, width, result.size(2))
    result = result.transpose(2, 3).transpose(1, 2)  # (B, C, H, W)
    return result


def load_image(path: str, image_size: int = 224):
    image = Image.open(path).convert("RGB")
    rgb_img = np.array(image.resize((image_size, image_size))) / 255.0

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return transform(image).unsqueeze(0), rgb_img.astype(np.float32)


def run_gradcam(
    ckpt_path: str,
    image_path: str,
    output_dir: str = "outputs/gradcam",
    pathology_idx: int | None = None,
    use_eigen: bool = False,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = EVAXModule.load_from_checkpoint(ckpt_path)
    model.eval()

    # Last transformer block — norm1 is applied before self-attention
    target_layers = [model.encoder.blocks[-1].norm1]

    input_tensor, rgb_img = load_image(image_path)

    with torch.no_grad():
        probs = torch.sigmoid(model(input_tensor))[0]

    if pathology_idx is None:
        pathology_idx = int(probs.argmax())

    print(f"Pathology: {PATHOLOGIES[pathology_idx]}  "
          f"(predicted prob = {probs[pathology_idx]:.3f})")

    CAMClass = EigenCAM if use_eigen else GradCAM
    cam = CAMClass(
        model=model,
        target_layers=target_layers,
        reshape_transform=reshape_transform,
    )

    grayscale_cam = cam(
        input_tensor=input_tensor,
        targets=[ClassifierOutputTarget(pathology_idx)],
    )[0]

    overlay = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(rgb_img)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(grayscale_cam, cmap="jet")
    axes[1].set_title("Grad-CAM heatmap")
    axes[1].axis("off")

    axes[2].imshow(overlay)
    label = PATHOLOGIES[pathology_idx]
    axes[2].set_title(f"{label}  (p={probs[pathology_idx]:.3f})")
    axes[2].axis("off")

    plt.tight_layout()
    stem = Path(image_path).stem
    out = output_dir / f"{stem}_{label.replace(' ', '_')}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output_dir", default="outputs/gradcam")
    parser.add_argument("--pathology", type=int, default=None,
                        help="0-13 index; default = top predicted class")
    parser.add_argument("--eigen", action="store_true",
                        help="Use EigenCAM (no backprop, faster, more stable)")
    args = parser.parse_args()

    run_gradcam(args.ckpt, args.image, args.output_dir, args.pathology, args.eigen)
