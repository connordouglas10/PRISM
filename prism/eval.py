"""Evaluation utilities for sentence embeddings and binary classification."""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_curve,
    auc,
    roc_auc_score,
)
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from sentence_transformers.evaluation import SentenceEvaluator
from sentence_transformers import InputExample


class BinaryROCEvaluator(SentenceEvaluator):
    """
    Evaluator for binary classification on sentence embeddings:
    accuracy, precision/recall/f1 at threshold=0.5, and ROC AUC.
    """

    def __init__(
        self,
        examples: list,
        batch_size: int = 32,
        name: str = "",
    ):
        super().__init__()
        self.sents1 = [e["sentence1"] for e in examples]
        self.sents2 = [e["sentence2"] for e in examples]
        self.labels = np.array([int(e["label"]) for e in examples])
        self.batch_size = batch_size
        self.name = name or "binary"

    def __call__(
        self,
        model: SentenceTransformer,
        output_path: str = None,
        epoch: int = -1,
        steps: int = -1,
    ) -> float:
        emb1 = model.encode(
            self.sents1,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        emb2 = model.encode(
            self.sents2,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        metrics = self._compute_metrics(emb1, emb2)
        return metrics[f"{self.name}_roc_auc"]

    def _compute_metrics(self, emb1: np.ndarray, emb2: np.ndarray) -> dict:
        scores = expit(np.sum(emb1 * emb2, axis=1))
        preds = (scores >= 0.5).astype(int)
        acc = accuracy_score(self.labels, preds)
        prec, rec, f1, _ = precision_recall_fscore_support(
            self.labels, preds, average="binary"
        )
        roc_auc = roc_auc_score(self.labels, scores)
        return {
            f"{self.name}_accuracy": acc,
            f"{self.name}_precision": prec,
            f"{self.name}_recall": rec,
            f"{self.name}_f1": f1,
            f"{self.name}_roc_auc": roc_auc,
        }


def plot_ROCs(text_1, text_2, labels, models, model_names=None):
    """Plot ROC curves for one or more models on sentence pairs."""
    fprs = []
    tprs = []
    aucs = []
    for model in models:
        embeddings_1 = model.encode(text_1)
        embeddings_2 = model.encode(text_2)
        sims = [
            cosine_similarity([embeddings_1[i], embeddings_2[i]])[0][1]
            for i in range(len(embeddings_1))
        ]
        fpr, tpr, _ = roc_curve(labels, sims)
        auc_val = auc(fpr, tpr)
        fprs.append(fpr)
        tprs.append(tpr)
        aucs.append(auc_val)

    if model_names is None:
        model_names = ["" for _ in models]

    plt.figure()
    for i in range(len(models)):
        plt.plot(
            fprs[i],
            tprs[i],
            lw=2,
            label="%s ROC curve (area = %0.2f)" % (model_names[i], aucs[i]),
        )
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic")
    plt.legend(loc="lower right")
    plt.show()
