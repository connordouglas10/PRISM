"""Core PRISM class for narrative clustering and model adaptation."""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    SimilarityFunction,
    losses,
    util,
)
from sentence_transformers.evaluation import (
    BinaryClassificationEvaluator,
    EmbeddingSimilarityEvaluator,
)
from sentence_transformers.training_args import BatchSamplers

from .eval import BinaryROCEvaluator
from .teachers import OpenAITeacher, LocalTeacher
from .utils import _flatten_clusters


class Prism:
    """
    Main entry point for narrative clustering and adaptation.
    """

    def __init__(
        self,
        data: Sequence[str] | np.ndarray | pd.DataFrame,
        model_name: str = "all-mpnet-base-v2",
        teacher: OpenAITeacher | LocalTeacher | None = None,
    ):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.MODEL: SentenceTransformer = SentenceTransformer(model_name, device=device)
        print(f"SentenceTransformers model loaded on {self.MODEL.device}")

        self.TEACHER = teacher or LocalTeacher()
        self.clusters: List[List[int]] | None = None
        self.embeddings: np.ndarray | None = None
        self.data_df: pd.DataFrame | None = None
        self.texts: Sequence[str] | None = None
        self.emb_train_iter = 0
        self.comp_train_iter = 0
        self.model_copies: List[SentenceTransformer] = []

        if isinstance(data, pd.DataFrame):
            self.data_df = data
            self.texts = data["Message_en"].to_numpy()
        elif isinstance(data, np.ndarray):
            self.texts = list(data)
        elif isinstance(data, list):
            self.texts = data
        else:
            raise TypeError("data must be of type DataFrame, ndarray, or list")

    # --- embeddings & clustering -------------------------------------------------

    def embed_texts(self) -> np.ndarray:
        """Embeds texts using the underlying SentenceTransformer model."""
        if self.texts is None:
            raise ValueError("No texts available to embed.")
        print("Embedding texts")
        self.embeddings = self.MODEL.encode(self.texts)
        return self.embeddings

    def cluster_texts(
        self,
        distance_threshold: float,
        min_community_size: int,
        recompute_embeddings: bool = False,
        new_embeddings: np.ndarray | None = None,
    ) -> List[List[int]]:
        """Cluster texts via sentence-transformers community detection."""
        if new_embeddings is None:
            if self.embeddings is None or recompute_embeddings:
                self.embed_texts()
            embeddings = self.embeddings
        else:
            embeddings = new_embeddings
        clusters = util.community_detection(
            embeddings,
            min_community_size=min_community_size,
            threshold=distance_threshold,
        )
        self.clusters = clusters
        return clusters

    def cluster_texts_agg(
        self,
        distance_threshold: float,
        min_community_size: int,
        recompute_embeddings: bool = False,
        new_embeddings: np.ndarray | None = None,
    ) -> List[List[int]]:
        """Cluster texts using AgglomerativeClustering on cosine distance."""
        X = (
            new_embeddings
            if new_embeddings is not None
            else (self.embed_texts() or self.embeddings)
            if (getattr(self, "embeddings", None) is None or recompute_embeddings)
            else self.embeddings
        )
        X = np.asarray(X.detach().cpu().numpy() if hasattr(X, "detach") else X)
        if X.size == 0:
            self.clusters = []
            return []

        thr = (
            1.0 - float(distance_threshold)
            if 0.0 <= float(distance_threshold) <= 1.0
            else float(distance_threshold)
        )

        labels = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=thr,
            metric="cosine",
            linkage="average",
            compute_full_tree=True,
        ).fit_predict(X)

        clusters: List[List[int]] = [
            np.where(labels == l)[0].tolist() for l in np.unique(labels)
        ]
        clusters = [c for c in clusters if len(c) >= min_community_size]
        clusters.sort(key=len, reverse=True)
        self.clusters = clusters
        return clusters

    def cluster_texts_kmeans(
        self,
        k_or_threshold: float,
        min_community_size: int,
        recompute_embeddings: bool = False,
        new_embeddings: np.ndarray | None = None,
    ) -> List[List[int]]:
        """Cluster texts using KMeans on L2-normalised embeddings."""
        X = (
            new_embeddings
            if new_embeddings is not None
            else (self.embed_texts() or self.embeddings)
            if (getattr(self, "embeddings", None) is None or recompute_embeddings)
            else self.embeddings
        )
        X = np.asarray(X.detach().cpu().numpy() if hasattr(X, "detach") else X)
        if X.size == 0:
            self.clusters = []
            return []

        n = X.shape[0]
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        Xn = X / norms

        if isinstance(k_or_threshold, (int, np.integer)) and k_or_threshold >= 1:
            k = int(k_or_threshold)
        else:
            k = int(np.ceil(np.sqrt(n)))
        k = max(1, min(k, n))

        labels = KMeans(n_clusters=k, n_init=10, random_state=None).fit_predict(Xn)
        clusters: List[List[int]] = [
            np.where(labels == l)[0].tolist() for l in np.unique(labels)
        ]
        clusters = [c for c in clusters if len(c) >= min_community_size]
        clusters.sort(key=len, reverse=True)
        self.clusters = clusters
        return clusters

    # --- cluster utilities -------------------------------------------------------

    def get_clusters_as_dict(self, clusters: List[List[int]] | None = None):
        """Return cluster contents as a dict of DataFrame rows."""
        if clusters is None:
            clusters = self.clusters
        if clusters is None or self.data_df is None:
            raise ValueError("Clusters and data_df must be available.")

        cluster_dict: Dict[int, List[Dict[str, Any]]] = {}
        cluster_keys = range(len(clusters))
        for cluster_index in cluster_keys:
            cluster_dict[cluster_index] = []
            for post_index in clusters[cluster_index]:
                cluster_dict[cluster_index].append(
                    self.data_df.iloc[post_index].to_dict()
                )
        return cluster_dict

    def get_flat_clusters(self, clusters: List[List[int]] | None = None) -> np.ndarray:
        """Flatten cluster labels into an array indexed like texts."""
        if clusters is None:
            clusters = self.clusters
        if clusters is None or self.embeddings is None:
            raise ValueError("Clusters and embeddings must be available.")

        flat_clusters = np.ones(len(self.embeddings)) * -1
        cluster_keys = range(len(clusters))
        for cluster_index in cluster_keys:
            for post_index in clusters[cluster_index]:
                flat_clusters[post_index] = cluster_index
        return flat_clusters.astype("int32")

    # --- comparison sampling -----------------------------------------------------

    @staticmethod
    def compute_central_most_embedding(embeddings: np.ndarray) -> int:
        """Index of embedding closest to the centroid."""
        centroid = np.mean(embeddings, axis=0)
        closest_index = np.argmin(
            [cosine_similarity([e, centroid])[0][1] for e in embeddings]
        )
        return int(closest_index)

    @staticmethod
    def compute_farthest_points(embeddings: np.ndarray) -> Tuple[int, int]:
        """Indices of the two farthest points in the embeddings."""
        max_distance = -1.0
        point_1, point_2 = 0, 0
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                dist = cosine_similarity([embeddings[i], embeddings[j]])[0][1]
                if dist > max_distance:
                    max_distance = dist
                    point_1, point_2 = i, j
        return int(point_1), int(point_2)

    def get_comparisons_from_clusters(
        self, clusters: List[List[int]] | None = None
    ) -> List[List[int]]:
        """Select central and extreme pairs within clusters for labelling."""
        if clusters is None:
            clusters = self.clusters
        if clusters is None:
            raise ValueError(
                "Cluster labels must be present on PRISM or passed explicitly."
            )
        if self.embeddings is None:
            raise ValueError("PRISM requires embeddings to generate comparisons.")

        comparisons: List[List[int]] = []
        for cluster_indices in clusters:
            embeddings_in_cluster = self.embeddings[cluster_indices]
            central_point = self.compute_central_most_embedding(embeddings_in_cluster)
            extrema_1, extrema_2 = self.compute_farthest_points(embeddings_in_cluster)
            points = list(set([central_point, extrema_1, extrema_2]))
            for i in range(1, len(points)):
                for j in range(0, i):
                    i_idx = cluster_indices[points[i]]
                    j_idx = cluster_indices[points[j]]
                    sim = cosine_similarity(
                        [self.embeddings[i_idx], self.embeddings[j_idx]]
                    )[0][1]
                    comparisons.append([i_idx, j_idx, sim])
        return comparisons

    # --- teacher-based datasets --------------------------------------------------

    def generate_comparison_dataset(self, comparisons):
        """Generate labelled sentence-pair dataset using the teacher model."""
        if comparisons is None:
            raise ValueError("Comparisons must be provided")
        if self.texts is None:
            raise ValueError("No texts available.")

        comparison_dataset: Dict[str, List[Any]] = {
            "sentence1": [],
            "sentence2": [],
            "label": [],
        }
        for comparison in comparisons:
            sentence1 = self.texts[comparison[0]]
            sentence2 = self.texts[comparison[1]]
            teacher_label, _ = self.TEACHER.compare(sentence1, sentence2)
            label = 1 if teacher_label == 1 else 0
            comparison_dataset["sentence1"].append(sentence1)
            comparison_dataset["sentence2"].append(sentence2)
            comparison_dataset["label"].append(label)
        return Dataset.from_dict(comparison_dataset)

    def get_teacher_embeddings(self, texts: Sequence[str]):
        """Fetch embeddings from the teacher model."""
        return self.TEACHER.encode(texts)

    def generate_embedding_sim_dataset(self, texts, teacher_embeddings=None):
        """Generate pairwise similarity dataset from teacher embeddings."""
        if teacher_embeddings is None:
            teacher_embeddings = self.get_teacher_embeddings(texts)
        else:
            if len(teacher_embeddings) != len(texts):
                raise ValueError("Embeddings must be the same length as texts")

        print(len(teacher_embeddings))
        embedding_dataset: Dict[str, List[Any]] = {
            "sentence1": [],
            "sentence2": [],
            "score": [],
        }
        pairs = []
        for i in range(len(texts)):
            for j in range(i):
                pairs.append((i, j))
        for i_idx, j_idx in pairs:
            embedding_dataset["sentence1"].append(texts[i_idx])
            embedding_dataset["sentence2"].append(texts[j_idx])
            embedding_dataset["score"].append(
                cosine_similarity([teacher_embeddings[i_idx], teacher_embeddings[j_idx]])[
                    0
                ][1]
            )
        return Dataset.from_dict(embedding_dataset)

    # --- fine-tuning loops -------------------------------------------------------

    def fine_tune_model_embedding(
        self,
        train_dataset: Dataset,
        train_parameters: Dict[str, Any] | None = None,
        eval_dataset: Dataset | None = None,
        eval_type: str | None = None,
    ):
        train_parameters = train_parameters or {}
        batch_size = train_parameters.get("batch_size", 16)
        num_train_epochs = train_parameters.get("num_train_epochs", 20)
        learning_rate = train_parameters.get("learning_rate", 3e-6)
        per_device_train_batch_size = train_parameters.get(
            "per_device_train_batch_size", 16
        )
        per_device_eval_batch_size = train_parameters.get(
            "per_device_eval_batch_size", 16
        )
        logging_steps = train_parameters.get("logging_steps", 10)

        print(batch_size)
        loss = losses.CoSENTLoss(self.MODEL)
        if eval_dataset is not None:
            if eval_type == "embedding":
                embedding_evaluator = EmbeddingSimilarityEvaluator(
                    sentences1=eval_dataset["sentence1"],
                    sentences2=eval_dataset["sentence2"],
                    scores=eval_dataset["score"],
                    main_similarity=SimilarityFunction.COSINE,
                    name="embed",
                )
                embedding_evaluator(self.MODEL)
                eval_strategy = "epoch"
            elif eval_type == "comparison":
                embedding_evaluator = BinaryROCEvaluator(
                    examples=eval_dataset,
                    batch_size=64,
                    name="dev-binary",
                )
                embedding_evaluator(self.MODEL)
                eval_strategy = "epoch"
            else:
                embedding_evaluator = None
                eval_strategy = "no"
        else:
            embedding_evaluator = None
            eval_strategy = "no"

        args = SentenceTransformerTrainingArguments(
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
            learning_rate=learning_rate,
            warmup_ratio=0.1,
            fp16=True,
            bf16=False,
            batch_sampler=BatchSamplers.NO_DUPLICATES,
            eval_strategy=eval_strategy,
            save_strategy="epoch",
            save_total_limit=2,
            logging_steps=logging_steps,
            run_name=f"mpnet-base-iter{self.emb_train_iter}",
        )
        trainer = SentenceTransformerTrainer(
            model=self.MODEL,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            loss=loss,
            evaluator=embedding_evaluator,
        )
        emb_history = trainer.train()
        self.emb_train_iter += 1
        return emb_history

    def fine_tune_model_comparison(
        self,
        train_dataset: Dataset,
        train_parameters: Dict[str, Any] | None = None,
        eval_dataset: Dataset | None = None,
        eval_type: str | None = None,
    ):
        train_parameters = train_parameters or {}
        batch_size = train_parameters.get("batch_size", 16)
        num_train_epochs = train_parameters.get("num_train_epochs", 20)
        learning_rate = train_parameters.get("learning_rate", 3e-6)
        per_device_train_batch_size = train_parameters.get(
            "per_device_train_batch_size", 16
        )
        per_device_eval_batch_size = train_parameters.get(
            "per_device_eval_batch_size", 16
        )
        logging_steps = train_parameters.get("logging_steps", 10)

        loss = losses.OnlineContrastiveLoss(
            self.MODEL,
            distance_metric=SimilarityFunction.to_similarity_fn("cosine"),
        )
        if eval_dataset is not None:
            embedding_evaluator = BinaryClassificationEvaluator(
                sentences1=eval_dataset["sentence1"],
                sentences2=eval_dataset["sentence2"],
                labels=eval_dataset["label"],
                name="embed",
            )
            embedding_evaluator(self.MODEL)
            eval_strategy = "epoch"
        else:
            embedding_evaluator = None
            eval_strategy = "no"

        args = SentenceTransformerTrainingArguments(
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
            learning_rate=learning_rate,
            warmup_ratio=0.1,
            fp16=True,
            bf16=False,
            batch_sampler=BatchSamplers.NO_DUPLICATES,
            eval_strategy=eval_strategy,
            save_strategy="epoch",
            save_total_limit=2,
            logging_steps=logging_steps,
            run_name=f"mpnet-base-iter{self.comp_train_iter}",
        )
        trainer = SentenceTransformerTrainer(
            model=self.MODEL,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            loss=loss,
            evaluator=embedding_evaluator,
        )
        emb_history = trainer.train()
        self.comp_train_iter += 1
        return emb_history

