"""PRISM: narrative clustering and adaptation toolkit."""

from .core import Prism
from .teachers import OpenAITeacher, LocalTeacher
from .metrics import (
    author_coherence,
    time_coherence,
    get_topic_accuracy,
    count_misclassified_posts,
    get_topic_coherence,
    get_topic_diversity,
)
from .eval import BinaryROCEvaluator, plot_ROCs

# Backwards-compatible alias
NarrativeParser = Prism

__all__ = [
    "Prism",
    "NarrativeParser",
    "OpenAITeacher",
    "LocalTeacher",
    "author_coherence",
    "time_coherence",
    "get_topic_accuracy",
    "count_misclassified_posts",
    "get_topic_coherence",
    "get_topic_diversity",
    "BinaryROCEvaluator",
    "plot_ROCs",
]

