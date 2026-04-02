"""
Backwards-compatible shim for the old `narrative_parser` module.

All functionality now lives in the `prism` package. Prefer:

    from prism import Prism, OpenAITeacher, LocalTeacher

The old `NarrativeParser` name is kept as an alias for PRISM.
"""

from prism import (  # noqa: F401
    Prism,
    NarrativeParser,
    OpenAITeacher,
    LocalTeacher,
    author_coherence,
    time_coherence,
    get_topic_accuracy,
    count_misclassified_posts,
    get_topic_coherence,
    get_topic_diversity,
    BinaryROCEvaluator,
    plot_ROCs,
)

