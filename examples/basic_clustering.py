"""Minimal example for using PRISM on a small dataset."""

import pandas as pd

from prism import Prism, LocalTeacher


def main():
    df = pd.DataFrame(
        {
            "Message_en": [
                "The product arrived late but support was helpful.",
                "Customer service responded quickly.",
                "The shipment was delayed and the box was damaged.",
                "I love the fast delivery.",
            ]
        }
    )

    teacher = LocalTeacher()
    prism = Prism(df, teacher=teacher)

    embeddings = prism.embed_texts()
    clusters = prism.cluster_texts(distance_threshold=0.8, min_community_size=2)

    print("Embeddings shape:", getattr(embeddings, "shape", None))
    print("Clusters:", clusters)
    print("Flat labels:", prism.get_flat_clusters(clusters))


if __name__ == "__main__":
    main()

