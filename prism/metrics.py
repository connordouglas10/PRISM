"""Cluster and topic evaluation metrics for PRISM."""

from collections import Counter, defaultdict
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from gensim.corpora import Dictionary
from gensim.models.coherencemodel import CoherenceModel
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score

from .utils import _entropy, _weighted_mean


def author_coherence(clusters: Dict[int, List[Dict[str, Any]]]) -> float:
    """Author-based coherence: higher when clusters group posts by author."""
    all_authors = Counter(
        post["User Name"] for posts in clusters.values() for post in posts
    )
    global_entropy = _entropy(all_authors)
    if global_entropy == 0:
        return 1.0

    entropies, sizes = [], []
    for posts in clusters.values():
        counts = Counter(p["User Name"] for p in posts)
        entropies.append(_entropy(counts))
        sizes.append(len(posts))

    weighted_within = _weighted_mean(entropies, sizes)
    return 1.0 - (weighted_within / global_entropy)


def time_coherence(clusters: Dict[int, List[Dict[str, Any]]]) -> float:
    """Time-based coherence: higher when clusters group posts by time."""

    def _to_seconds(t):
        try:
            return pd.to_datetime(t).value / 1e9
        except Exception:
            return None

    all_times = [
        _to_seconds(p["Datetime"])
        for posts in clusters.values()
        for p in posts
    ]
    all_times = [t for t in all_times if t is not None and not np.isnan(t)]
    if len(all_times) <= 1:
        return 1.0

    global_var = np.var(all_times)
    if global_var == 0:
        return 1.0

    vars_, sizes = [], []
    for posts in clusters.values():
        ts = [_to_seconds(p["Datetime"]) for p in posts]
        ts = [t for t in ts if t is not None and not np.isnan(t)]
        if len(ts) > 1:
            vars_.append(np.var(ts))
        elif len(ts) == 1:
            vars_.append(0.0)
        else:
            vars_.append(global_var)
        sizes.append(len(posts))

    weighted_within = _weighted_mean(vars_, sizes)
    return 1.0 - (weighted_within / global_var)


def get_topic_accuracy(
    true_labels: Sequence[Any],
    topic_ids: Sequence[int],
    skip_unassigned: bool = True,
) -> Tuple[float, int]:
    """
    Cluster-prediction accuracy given topic assignments.
    A value of -1 in topic_ids means "no cluster assignment".
    Returns (accuracy, n_skipped).
    """
    if len(topic_ids) != len(true_labels):
        raise ValueError("`topic_ids` and `true_labels` must have equal length.")

    kept_indices = (
        [i for i, t in enumerate(topic_ids) if t != -1]
        if skip_unassigned
        else list(range(len(topic_ids)))
    )

    if not kept_indices:
        return 1.0, len(topic_ids)

    topic_to_indices: Dict[int, List[int]] = {}
    for idx in kept_indices:
        topic = topic_ids[idx]
        topic_to_indices.setdefault(topic, []).append(idx)

    topic_to_label: Dict[int, Any] = {
        topic: Counter(true_labels[i] for i in idxs).most_common(1)[0][0]
        for topic, idxs in topic_to_indices.items()
    }

    pred_labels = [topic_to_label[topic_ids[i]] for i in kept_indices]
    kept_true = [true_labels[i] for i in kept_indices]

    accuracy = accuracy_score(kept_true, pred_labels)
    n_skipped = len(topic_ids) - len(kept_indices)
    return float(accuracy), int(n_skipped)


def count_misclassified_posts(
    true_labels: Sequence[Any],
    topic_ids: Sequence[int],
) -> int:
    """Count misclassified documents among those with assigned topics (topic_id != -1)."""
    if len(topic_ids) != len(true_labels):
        raise ValueError("`topic_ids` and `true_labels` must have equal length.")

    kept_indices = [i for i, t in enumerate(topic_ids) if t != -1]
    if not kept_indices:
        return 0

    topic_to_indices: Dict[int, List[int]] = {}
    for idx in kept_indices:
        topic = topic_ids[idx]
        topic_to_indices.setdefault(topic, []).append(idx)

    topic_to_label: Dict[int, Any] = {
        topic: Counter(true_labels[i] for i in idxs).most_common(1)[0][0]
        for topic, idxs in topic_to_indices.items()
    }

    n_misclassified = sum(
        1
        for i in kept_indices
        if true_labels[i] != topic_to_label[topic_ids[i]]
    )
    return int(n_misclassified)


def get_topic_coherence(documents, doc_topics, top_n_words: int = 30) -> float:
    """C_v coherence over topic top words."""
    tokenized_docs = [doc.lower().split() for doc in documents]

    topic_to_docs: Dict[int, List[List[str]]] = defaultdict(list)
    for doc, topic in zip(tokenized_docs, doc_topics):
        if topic != -1:
            topic_to_docs[topic].append(doc)

    topic_words: List[List[str]] = []
    vectorizer = CountVectorizer()
    for topic, docs in topic_to_docs.items():
        joined_docs = [" ".join(doc) for doc in docs]
        if not joined_docs or all(doc.strip() == "" for doc in joined_docs):
            continue
        try:
            X = vectorizer.fit_transform(joined_docs)
            if X.shape[1] == 0:
                continue
            word_freq = np.asarray(X.sum(axis=0)).flatten()
            vocab = np.array(vectorizer.get_feature_names_out())
            top_indices = word_freq.argsort()[::-1][:top_n_words]
            top_words = vocab[top_indices].tolist()
            topic_words.append(top_words)
        except ValueError:
            continue

    if not topic_words:
        raise ValueError("No valid topics with top words were found.")

    dictionary = Dictionary(tokenized_docs)
    coherence_model = CoherenceModel(
        topics=topic_words,
        texts=tokenized_docs,
        dictionary=dictionary,
        coherence="c_v",
    )
    return float(coherence_model.get_coherence())


def get_topic_diversity(documents, doc_topics, top_n_words: int = 30) -> float:
    """
    Diversity of top words across topics:
    unique_top_words / (n_topics * top_n_words).
    """
    tokenized_docs = [doc.lower().split() for doc in documents]

    topic_to_docs: Dict[int, List[List[str]]] = defaultdict(list)
    for doc, topic in zip(tokenized_docs, doc_topics):
        if topic != -1:
            topic_to_docs[topic].append(doc)

    topic_words: List[List[str]] = []
    vectorizer = CountVectorizer()
    for topic, docs in topic_to_docs.items():
        joined_docs = [" ".join(doc) for doc in docs]
        if not joined_docs or all(doc.strip() == "" for doc in joined_docs):
            continue
        try:
            X = vectorizer.fit_transform(joined_docs)
            if X.shape[1] == 0:
                continue
            word_freq = np.asarray(X.sum(axis=0)).flatten()
            vocab = np.array(vectorizer.get_feature_names_out())
            top_indices = word_freq.argsort()[::-1][:top_n_words]
            top_words = vocab[top_indices].tolist()
            topic_words.append(top_words)
        except ValueError:
            continue

    if not topic_words:
        return 0.0

    all_top_words = sum(topic_words, [])
    unique_top_words = set(all_top_words)
    diversity = len(unique_top_words) / (len(topic_words) * top_n_words)
    return float(diversity)

