"""Smoke tests for the prism package."""

import pandas as pd


def test_prism_import_and_cluster():
    from prism import Prism, LocalTeacher  # noqa: F401

    df = pd.DataFrame(
        {
            "Message_en": [
                "This is a test sentence.",
                "Another example sentence.",
                "Yet another test sentence.",
            ]
        }
    )

    teacher = LocalTeacher()
    prism = Prism(df, teacher=teacher)
    embeddings = prism.embed_texts()
    clusters = prism.cluster_texts(distance_threshold=0.8, min_community_size=1)

    assert embeddings is not None
    assert isinstance(clusters, list)

