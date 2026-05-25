# MEDIMG — Multi-Label Chest X-Ray Pathology Classification

Multi-label classification of 14 chest pathologies from chest X-rays using the CheXpert dataset.

Two modeling approaches are explored and compared: a fully supervised fine-tuning of the EVA-X foundation model, and a self-supervised Mean Teacher framework for semi-supervised learning. Training is managed with PyTorch Lightning and Hydra for reproducible configuration.

Post-hoc interpretability is provided via GradCAM heatmaps and prototype analysis, giving visual insight into which image regions drive each prediction. Robustness is assessed through ablation studies across uncertainty label strategies, data augmentation schemes, and pretrained vs. random initialisation, with statistical validation via bootstrap confidence intervals.

> This project was bootstrapped from the [lightning-hydra-template](https://github.com/ashleve/lightning-hydra-template) by ashleve.

---

## Labels

The model predicts the following 14 observations derived from radiology reports via an automated rule-based labeler. Each label is either positive (1), uncertain (u), or negative (0):

| # | Finding |
|---|---------|
| 1 | No Finding |
| 2 | Enlarged Cardiomediastinum |
| 3 | Cardiomegaly |
| 4 | Lung Opacity |
| 5 | Lung Lesion |
| 6 | Edema |
| 7 | Consolidation |
| 8 | Pneumonia |
| 9 | Atelectasis |
| 10 | Pneumothorax |
| 11 | Pleural Effusion |
| 12 | Pleural Other |
| 13 | Fracture |
| 14 | Support Devices |

---

## Project Structure

```
├── configs/          # Hydra config files (train, eval, model, data, …)
├── data/             # CheXpert CSVs and data files
├── logs/             # Training checkpoints and logs
├── notebooks/        # Exploratory notebooks
├── outputs/          # Evaluation outputs (e.g. misclassified.csv, GradCAM images)
├── scripts/          # Analysis scripts (GradCAM, prototypes)
├── src/
│   ├── data/         # CheXpert datamodule
│   ├── models/       # Model module
│   ├── utils/        # Shared utilities
│   ├── train.py      # Training entry point
│   └── eval.py       # Evaluation entry point
├── tests/
├── requirements.txt
└── environment.yaml
```

---

## Setup

**Option A — pip**
```bash
pip install -r requirements.txt
```

---

## Running the Project

### Train

```bash
python src/train.py
```

Override any config value on the command line via Hydra:
```bash
python src/train.py trainer.max_epochs=50 seed=42
```

### Evaluate

```bash
python src/eval.py ckpt_path=logs/best.ckpt
```

---

## Explainability Scripts

Run these after training with a saved checkpoint.

**GradCAM on misclassified samples**
```bash
python scripts/gradcam_evax.py --ckpt logs/best.ckpt
```

**GradCAM on correctly classified samples**
```bash
python scripts/gradcam_correct_evax.py --ckpt logs/best.ckpt
```

**Prototype analysis**
```bash
python scripts/prototypes_evax.py --ckpt logs/best.ckpt
```

**Prototype analysis with GradCAM overlay**
```bash
python scripts/prototypes_gradcam_evax.py --ckpt logs/best.ckpt
```

Outputs are saved to `outputs/` by default.

---

## Ablation Studies

To assess model robustness, the following ablations are evaluated:

- **Uncertainty label policy** — U-Ones vs. U-Zeros vs. U-Ignore
- **Initialisation** — pretrained EVA-X vs. random initialisation
- **Data augmentation** — none, flips/rotations, medical noise
- **Reproducibility** — multiple runs with different random seeds
- **Statistical validation** — 95% bootstrap confidence intervals on AUROC

---

## Acknowledgements

Project structure based on the [lightning-hydra-template](https://github.com/ashleve/lightning-hydra-template) by ashleve.

