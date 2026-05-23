"""
Prototype-based interpretability utilities with GradCAM support.

Extends prototypes.py with a 4-column plot that includes a GradCAM overlay.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import pairwise_distances


def _kmedoids(
    X: np.ndarray,
    n_clusters: int,
    max_iter: int = 300,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(random_state)
    D = pairwise_distances(X, metric="euclidean")
    medoid_idx = rng.choice(len(X), size=n_clusters, replace=False)
    for _ in range(max_iter):
        labels = np.argmin(D[:, medoid_idx], axis=1)
        new_medoids = medoid_idx.copy()
        for k in range(n_clusters):
            members = np.where(labels == k)[0]
            if len(members) == 0:
                continue
            sub_D = D[np.ix_(members, members)]
            new_medoids[k] = members[np.argmin(sub_D.sum(axis=1))]
        if np.array_equal(np.sort(new_medoids), np.sort(medoid_idx)):
            break
        medoid_idx = new_medoids
    labels = np.argmin(D[:, medoid_idx], axis=1)
    return medoid_idx, labels


class PrototypeExplainer:
    def __init__(self, n_prototypes: int = 3, random_state: int = 42) -> None:
        self.n_prototypes = n_prototypes
        self.random_state = random_state
        self._proto_features: np.ndarray | None = None
        self._proto_labels: np.ndarray | None = None
        self._proto_images: list | None = None

    def fit(self, features, labels, images) -> PrototypeExplainer:
        features = np.asarray(features)
        labels   = np.asarray(labels)
        proto_features, proto_labels, proto_images = [], [], []
        for cls in np.unique(labels):
            cls_indices = np.where(labels == cls)[0]
            cls_feats   = features[cls_indices]
            n_clusters  = min(self.n_prototypes, len(cls_feats))
            med_local, _ = _kmedoids(cls_feats, n_clusters, random_state=self.random_state)
            for local_idx in med_local:
                proto_features.append(cls_feats[local_idx])
                proto_labels.append(cls)
                proto_images.append(images[cls_indices[local_idx]])
        self._proto_features = np.array(proto_features)
        self._proto_labels   = np.array(proto_labels)
        self._proto_images   = proto_images
        return self

    def explain(self, query_feature):
        self._check_fitted()
        dists = np.linalg.norm(self._proto_features - query_feature, axis=1)
        best  = int(np.argmin(dists))
        return int(self._proto_labels[best]), self._proto_images[best], float(dists[best])

    def explain_class(self, query_feature, cls):
        self._check_fitted()
        mask = self._proto_labels == cls
        if not mask.any():
            raise ValueError(f"Class {cls} has no prototypes. Call fit() with data containing this class.")
        cls_feats = self._proto_features[mask]
        cls_imgs  = [img for img, m in zip(self._proto_images, mask) if m]
        dists     = np.linalg.norm(cls_feats - query_feature, axis=1)
        best      = int(np.argmin(dists))
        return cls, cls_imgs[best], float(dists[best])

    def _check_fitted(self):
        if self._proto_features is None:
            raise RuntimeError("PrototypeExplainer has not been fitted. Call fit() first.")


class MisclassificationAnalyzer:
    def __init__(self, explainer: PrototypeExplainer) -> None:
        self.explainer = explainer

    def analyze(self, test_features, test_labels, test_images, predicted_labels) -> list[dict]:
        test_features    = np.asarray(test_features)
        test_labels      = np.asarray(test_labels)
        predicted_labels = np.asarray(predicted_labels)
        results = []
        for i in np.where(predicted_labels != test_labels)[0]:
            q_feat   = test_features[i]
            true_cls = int(test_labels[i])
            pred_cls = int(predicted_labels[i])
            _, pred_proto_img, pred_dist = self.explainer.explain_class(q_feat, pred_cls)
            _, true_proto_img, true_dist = self.explainer.explain_class(q_feat, true_cls)
            results.append({
                "query_image":          test_images[i],
                "true_label":           true_cls,
                "pred_label":           pred_cls,
                "pred_prototype_image": pred_proto_img,
                "pred_prototype_dist":  pred_dist,
                "true_prototype_image": true_proto_img,
                "true_prototype_dist":  true_dist,
            })
        return results

    def plot(self, results, max_samples=12, save_path: Optional[str | Path] = None) -> None:
        n_show = min(max_samples, len(results))
        if n_show == 0:
            return

        has_gradcam = "gradcam_overlay" in results[0]
        cols = 4 if has_gradcam else 3
        fig, axes = plt.subplots(n_show, cols, figsize=(cols * 2.8, n_show * 2.8))
        if n_show == 1:
            axes = axes[np.newaxis, :]

        fig.suptitle(
            "Misclassification Explanation via Nearest Prototype\n"
            "Col 1: query  |  Col 2: nearest prototype of PREDICTED class  "
            "|  Col 3: nearest prototype of TRUE class"
            + ("  |  Col 4: GradCAM overlay" if has_gradcam else ""),
            fontsize=10, y=1.01,
        )

        for i, rec in enumerate(results[:n_show]):
            ax = axes[i]
            true_cls = rec["true_label"]
            pred_cls = rec["pred_label"]

            ax[0].imshow(rec["query_image"], cmap="gray_r")
            ax[0].set_title(f"Query\ntrue={true_cls}  pred={pred_cls}", fontsize=8, color="red")
            ax[0].axis("off")

            ax[1].imshow(rec["pred_prototype_image"], cmap="gray_r")
            ax[1].set_title(f"Pred proto (cls {pred_cls})\nd={rec['pred_prototype_dist']:.1f}", fontsize=8, color="darkorange")
            ax[1].axis("off")
            for spine in ax[1].spines.values():
                spine.set_edgecolor("darkorange")
                spine.set_linewidth(3)

            ax[2].imshow(rec["true_prototype_image"], cmap="gray_r")
            ax[2].set_title(f"True proto (cls {true_cls})\nd={rec['true_prototype_dist']:.1f}", fontsize=8, color="green")
            ax[2].axis("off")

            if has_gradcam:
                ax[3].imshow(rec["gradcam_overlay"])
                ax[3].set_title("GradCAM\n(predicted class)", fontsize=8, color="purple")
                ax[3].axis("off")

        plt.tight_layout()
        if save_path is not None:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close()
        else:
            plt.show()
