"""
Prototype-based interpretability utilities.

Provides K-Medoids prototype selection and misclassification explanation
via nearest-prototype lookup in embedding space.

Classes:
  PrototypeExplainer       – builds per-class K-Medoids prototypes from training features
  MisclassificationAnalyzer – explains misclassifications via nearest-prototype distances

Usage::

    from src.utils.prototypes import PrototypeExplainer, MisclassificationAnalyzer

    explainer = PrototypeExplainer(n_prototypes=3)
    explainer.fit(train_features, train_labels, train_images)

    analyzer = MisclassificationAnalyzer(explainer)
    results = analyzer.analyze(test_features, test_labels, test_images, predicted_labels)
    analyzer.plot(results, save_path="outputs/misclassification_prototypes.png")
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
    """Pure NumPy K-Medoids with Euclidean distance.

    Args:
        X:            Feature matrix, shape (N, D).
        n_clusters:   Number of medoids to find.
        max_iter:     Maximum number of swap iterations.
        random_state: Seed for reproducible initialisation.

    Returns:
        Tuple of (medoid_indices, labels) where medoid_indices index into X
        and labels assign each sample to its nearest medoid.
    """
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
    """Builds per-class K-Medoids prototypes and answers nearest-prototype queries.

    After calling fit(), each class is represented by ``n_prototypes`` medoid
    samples selected from the training set in embedding space. Explanations
    are produced by finding the prototype with the smallest Euclidean distance
    to a query embedding.

    Args:
        n_prototypes: Number of prototypes per class.
        random_state: Seed for K-Medoids initialisation.
    """

    def __init__(self, n_prototypes: int = 3, random_state: int = 42) -> None:
        self.n_prototypes = n_prototypes
        self.random_state = random_state

        self._proto_features: np.ndarray | None = None
        self._proto_labels: np.ndarray | None = None
        self._proto_images: list | None = None

    def fit(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        images: np.ndarray,
    ) -> PrototypeExplainer:
        """Build K-Medoids prototypes from training data.

        Args:
            features: Embedding matrix, shape (N, D). Already scaled/embedded
                      by the model's feature extractor — no preprocessing applied.
            labels:   Integer class labels, shape (N,).
            images:   Array or sequence of images, one per sample. Passed
                      directly to ``matplotlib.pyplot.imshow``; the caller is
                      responsible for shape and value range.

        Returns:
            self, to allow method chaining.
        """
        features = np.asarray(features)
        labels = np.asarray(labels)

        proto_features: list = []
        proto_labels: list = []
        proto_images: list = []

        for cls in np.unique(labels):
            cls_indices = np.where(labels == cls)[0]
            cls_feats = features[cls_indices]

            n_clusters = min(self.n_prototypes, len(cls_feats))
            med_local, _ = _kmedoids(cls_feats, n_clusters, random_state=self.random_state)

            for local_idx in med_local:
                proto_features.append(cls_feats[local_idx])
                proto_labels.append(cls)
                proto_images.append(images[cls_indices[local_idx]])

        self._proto_features = np.array(proto_features)
        self._proto_labels = np.array(proto_labels)
        self._proto_images = proto_images
        return self

    def explain(self, query_feature: np.ndarray) -> tuple[int, object, float]:
        """Find the nearest prototype across all classes.

        Args:
            query_feature: Embedding vector, shape (D,).

        Returns:
            Tuple of ``(predicted_class, prototype_image, distance)``.
        """
        self._check_fitted()
        dists = np.linalg.norm(self._proto_features - query_feature, axis=1)
        best = int(np.argmin(dists))
        return int(self._proto_labels[best]), self._proto_images[best], float(dists[best])

    def explain_class(
        self, query_feature: np.ndarray, cls: int
    ) -> tuple[int, object, float]:
        """Find the nearest prototype restricted to a single class.

        Args:
            query_feature: Embedding vector, shape (D,).
            cls:           Class label to restrict the search to.

        Returns:
            Tuple of ``(cls, prototype_image, distance)``.
        """
        self._check_fitted()
        mask = self._proto_labels == cls
        if not mask.any():
            raise ValueError(
                f"Class {cls} has no prototypes. Call fit() with data containing this class."
            )
        cls_feats = self._proto_features[mask]
        cls_imgs = [img for img, m in zip(self._proto_images, mask) if m]
        dists = np.linalg.norm(cls_feats - query_feature, axis=1)
        best = int(np.argmin(dists))
        return cls, cls_imgs[best], float(dists[best])

    def _check_fitted(self) -> None:
        if self._proto_features is None:
            raise RuntimeError(
                "PrototypeExplainer has not been fitted. Call fit() first."
            )


class MisclassificationAnalyzer:
    """Explains misclassified test samples using nearest-prototype distances.

    For each misclassified sample, records the nearest prototype of the
    predicted class (explaining why the model chose that class) and the
    nearest prototype of the true class (showing what the correct class
    looks like in training data).

    Args:
        explainer: A fitted PrototypeExplainer instance.
    """

    def __init__(self, explainer: PrototypeExplainer) -> None:
        self.explainer = explainer

    def analyze(
        self,
        test_features: np.ndarray,
        test_labels: np.ndarray,
        test_images: np.ndarray,
        predicted_labels: np.ndarray,
    ) -> list[dict]:
        """Collect nearest-prototype explanations for every misclassified sample.

        Args:
            test_features:    Embeddings for the test set, shape (N, D).
            test_labels:      True integer labels, shape (N,).
            test_images:      Images for the test set, one per sample.
            predicted_labels: Predicted integer labels from the classifier, shape (N,).

        Returns:
            List of dicts, one per misclassified sample. Each dict contains:
            ``query_image``, ``true_label``, ``pred_label``,
            ``pred_prototype_image``, ``pred_prototype_dist``,
            ``true_prototype_image``, ``true_prototype_dist``.
        """
        test_features = np.asarray(test_features)
        test_labels = np.asarray(test_labels)
        predicted_labels = np.asarray(predicted_labels)

        results: list[dict] = []
        wrong_indices = np.where(predicted_labels != test_labels)[0]

        for i in wrong_indices:
            q_feat = test_features[i]
            true_cls = int(test_labels[i])
            pred_cls = int(predicted_labels[i])

            _, pred_proto_img, pred_dist = self.explainer.explain_class(q_feat, pred_cls)
            _, true_proto_img, true_dist = self.explainer.explain_class(q_feat, true_cls)

            results.append({
                "query_image": test_images[i],
                "true_label": true_cls,
                "pred_label": pred_cls,
                "pred_prototype_image": pred_proto_img,
                "pred_prototype_dist": pred_dist,
                "true_prototype_image": true_proto_img,
                "true_prototype_dist": true_dist,
            })

        return results

    def plot(
        self,
        results: list[dict],
        max_samples: int = 12,
        save_path: Optional[str | Path] = None,
    ) -> None:
        """Plot misclassification explanations as a 3-column grid.

        Each row shows: query image | nearest prototype of the predicted class
        | nearest prototype of the true class. The orange border on the centre
        panel highlights why the model made its prediction.

        Args:
            results:     Output of analyze().
            max_samples: Maximum number of rows to render.
            save_path:   File path for saving the figure (dpi=150). If None,
                         the figure is shown interactively.
        """
        n_show = min(max_samples, len(results))
        if n_show == 0:
            return

        cols = 3
        fig, axes = plt.subplots(n_show, cols, figsize=(cols * 2.8, n_show * 2.8))
        if n_show == 1:
            axes = axes[np.newaxis, :]

        fig.suptitle(
            "Misclassification Explanation via Nearest Prototype\n"
            "Left: query  |  Centre: nearest prototype of PREDICTED class  "
            "|  Right: nearest prototype of TRUE class",
            fontsize=10,
            y=1.01,
        )

        for i, rec in enumerate(results[:n_show]):
            ax0, ax1, ax2 = axes[i]
            true_cls = rec["true_label"]
            pred_cls = rec["pred_label"]

            ax0.imshow(rec["query_image"], cmap="gray_r")
            ax0.set_title(
                f"Query\ntrue={true_cls}  pred={pred_cls}",
                fontsize=8,
                color="red",
            )
            ax0.axis("off")

            ax1.imshow(rec["pred_prototype_image"], cmap="gray_r")
            ax1.set_title(
                f"Pred proto (cls {pred_cls})\nd={rec['pred_prototype_dist']:.1f}",
                fontsize=8,
                color="darkorange",
            )
            ax1.axis("off")
            for spine in ax1.spines.values():
                spine.set_edgecolor("darkorange")
                spine.set_linewidth(3)

            ax2.imshow(rec["true_prototype_image"], cmap="gray_r")
            ax2.set_title(
                f"True proto (cls {true_cls})\nd={rec['true_prototype_dist']:.1f}",
                fontsize=8,
                color="green",
            )
            ax2.axis("off")

        plt.tight_layout()

        if save_path is not None:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close()
        else:
            plt.show()
