"""Teacher abstractions for PRISM (OpenAI-based and local models)."""

import os
import time
from typing import Any, List, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer


def query_compare(text_1: str, text_2: str) -> str:
    return (
        "Answer in 'Yes' or 'No'. At a high level of detail, are these two pieces "
        "of text saying essentially the same thing: '%s' and '%s'" % (text_1, text_2)
    )


def query_paraphrase(text_1: str) -> str:
    return "Paraphrase the important elements of the following: %s" % (text_1,)


class OpenAITeacher:
    """
    Wrapper around the OpenAI API used as a 'teacher' model.

    Notes
    -----
    - The API key is taken from the `api_key` argument if provided, otherwise
      from the OPENAI_API_KEY environment variable.
    """

    def __init__(
        self,
        model: str = "gpt-5-2025-08-07",
        temperature: float = 0.0,
        query_compare_fn=query_compare,
        query_paraphrase_fn=query_paraphrase,
        embedding_model: str = "text-embedding-3-large",
        max_embedding_length: int = 4000,
        logprobs=None,
        top_logprobs=None,
        name: str | None = None,
        embed_batch_size: int = 200,
        embed_batch_pause: float = 0.0,
        api_key: str | None = None,
    ):
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OpenAITeacher requires an API key. "
                "Set OPENAI_API_KEY or pass api_key= explicitly."
            )

        self.CLIENT = OpenAI(api_key=api_key)
        self.EMBEDDING_MODEL = embedding_model
        self.LOGPROBS = logprobs
        self.MODEL = model
        self.PROJECT_NAME = "PRISM"
        self.QUERY_COMPARE = query_compare_fn
        self.QUERY_PARAPHRASE = query_paraphrase_fn
        self.TEMPERATURE = temperature
        self.TOP_LOGPROBS = top_logprobs
        self.MAX_EMBEDDING_LENGTH = max_embedding_length
        self.EMBED_BATCH_SIZE = embed_batch_size
        self.EMBED_BATCH_PAUSE = embed_batch_pause
        self.NAME = name or f"{model}/{embedding_model}"

    def compare(self, text_1: str, text_2: str) -> Tuple[int | None, str]:
        prompt = self.QUERY_COMPARE(text_1, text_2)
        completion = self.CLIENT.responses.create(
            model=self.MODEL,
            input=prompt,
            reasoning={"effort": "minimal"},
            text={"verbosity": "low"},
        )
        # follow the current responses API layout
        answer = completion.output[1].content[0].text
        similar_binary: int | None = None
        if "YES" in answer.upper()[:7]:
            similar_binary = 1
        elif "NO" in answer.upper()[:7]:
            similar_binary = 0
        return similar_binary, answer

    def paraphrase(self, text_1: str) -> str:
        prompt = self.QUERY_PARAPHRASE(text_1)
        completion = self.CLIENT.chat.completions.create(
            model=self.MODEL,
            temperature=self.TEMPERATURE,
            logprobs=self.LOGPROBS,
            top_logprobs=self.TOP_LOGPROBS,
            messages=[
                {"role": "system", "content": prompt},
            ],
        )
        return completion.choices[0].message.content

    def encode(
        self, texts: Union[pd.Series, str, Sequence[str], np.ndarray]
    ) -> np.ndarray:
        print(f"Texts of length {len(np.array(texts))}")

        if isinstance(texts, pd.Series):
            texts = texts.to_numpy()
        elif isinstance(texts, str):
            texts = np.array([texts])
        elif isinstance(texts, list):
            texts = np.array(texts)
        elif isinstance(texts, np.ndarray):
            texts = texts
        else:
            raise TypeError(
                "texts must be of type Series, String, List, or ndarray; "
                f"not {type(texts)}"
            )

        inputs = np.array(
            [
                (s or "")[: self.MAX_EMBEDDING_LENGTH].replace("\n", " ")
                for s in texts
            ],
            dtype=str,
        )

        embeddings: List[np.ndarray] = []
        for start in range(0, len(texts), self.EMBED_BATCH_SIZE):
            print(f"Embedded {start} items")
            batch = inputs[start : start + self.EMBED_BATCH_SIZE]
            response = self.CLIENT.embeddings.create(
                input=batch,
                model=self.EMBEDDING_MODEL,
            )
            embeddings.extend(data.embedding for data in response.data)
            if self.EMBED_BATCH_PAUSE:
                time.sleep(self.EMBED_BATCH_PAUSE)

        return np.array(embeddings)

    def compare_embeddings(self, text_1: str, text_2: str):
        embedding_1 = self.encode(text_1)
        embedding_2 = self.encode(text_2)
        cosine_sim = cosine_similarity([embedding_1, embedding_2])[0][1]
        return cosine_sim, embedding_1, embedding_2


class LocalTeacher:
    """Local sentence-transformer-based teacher (no external API calls)."""

    def __init__(
        self,
        embedding_model: SentenceTransformer
        | str = "intfloat/multilingual-e5-large-instruct",
        max_embedding_length: int = 4000,
    ):
        if isinstance(embedding_model, str):
            embedding_model = SentenceTransformer(embedding_model)
        self.EMBEDDING_MODEL: SentenceTransformer = embedding_model
        self.MAX_EMBEDDING_LENGTH = max_embedding_length

    def encode(self, text_1: Union[str, Sequence[str]]) -> np.ndarray:
        return self.EMBEDDING_MODEL.encode(text_1)

    def compare_embeddings(self, text_1: str, text_2: str):
        embedding_1 = self.encode(text_1)
        embedding_2 = self.encode(text_2)
        cosine_sim = cosine_similarity([embedding_1, embedding_2])[0][1]
        return cosine_sim, embedding_1, embedding_2

