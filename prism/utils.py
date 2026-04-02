"""Internal helpers for the prism package."""

import math
import numpy as np


def _weighted_mean(values, weights):
    values, weights = np.asarray(values), np.asarray(weights)
    return (values * weights).sum() / weights.sum() if weights.sum() else math.nan


def _entropy(counts):
    """Shannon entropy of a Counter (base-e)."""
    total = sum(counts.values())
    if total == 0:
        return 0.0
    probs = np.array(list(counts.values()), dtype=float) / total
    probs = probs[probs > 0]
    return -(probs * np.log(probs)).sum()


def _flatten_clusters(clusters: dict, text_key: str = "Message_en"):
    """
    Flatten clusters dict into texts list, labels array, and cluster index lists.
    Returns:
      texts: List[str]
      labels: np.ndarray of shape (n_samples,)
      cluster_idxs: List[List[int]] mapping each cluster to its text indices
    """
    texts = []
    labels = []
    cluster_idxs = []
    idx = 0
    for cluster_id in sorted(clusters):
        items = clusters[cluster_id]
        n = len(items)
        cluster_idxs.append(list(range(idx, idx + n)))
        for item in items:
            texts.append(item[text_key])
            labels.append(cluster_id)
        idx += n
    return texts, np.array(labels, dtype=int), cluster_idxs
