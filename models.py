"""Embedding model clients: sentence-transformers (local) and Gemini API."""

import time
from typing import List

import numpy as np


def _lazy_genai():
    from google import genai
    return genai


def _lazy_st():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer


class GeminiEmbeddingClient:
    """Client for Google Gemini embedding API."""

    def __init__(
        self,
        api_key: str,
        model_name: str = "gemini-embedding-001",
        batch_size: int = 32,
    ):
        genai = _lazy_genai()
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
        self.batch_size = batch_size
        self.name = model_name

    def embed(self, texts: List[str], task_type: str = "RETRIEVAL_DOCUMENT") -> np.ndarray:
        all_embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            result = self._embed_with_retry(batch, task_type)
            for emb in result.embeddings:
                all_embeddings.append(emb.values)
            if i + self.batch_size < len(texts):
                time.sleep(1)
        return np.array(all_embeddings, dtype=np.float32)

    def _embed_with_retry(
        self,
        batch: List[str],
        task_type: str,
        max_attempts: int = 6,
        base_delay: float = 2.0,
    ):
        from google.genai import errors as genai_errors

        transient = (genai_errors.ServerError, genai_errors.APIError)
        retry_codes = {429, 500, 503, 504}

        for attempt in range(max_attempts):
            try:
                return self.client.models.embed_content(
                    model=self.model_name,
                    contents=batch,
                    config={"task_type": task_type},
                )
            except transient as e:
                code = getattr(e, "code", None)
                is_last = attempt == max_attempts - 1
                if is_last or (code is not None and code not in retry_codes):
                    raise
                delay = base_delay * (2 ** attempt)
                print(
                    f"[GeminiEmbeddingClient] transient error (code={code}), "
                    f"retrying in {delay:.1f}s (attempt {attempt + 1}/{max_attempts})"
                )
                time.sleep(delay)

    def embed_timed(
        self, texts: List[str], task_type: str = "RETRIEVAL_DOCUMENT"
    ) -> tuple[np.ndarray, float]:
        start = time.perf_counter()
        embs = self.embed(texts, task_type=task_type)
        return embs, time.perf_counter() - start


class SentenceTransformerClient:
    """Client for local sentence-transformers models (runs on CUDA/MPS/CPU)."""

    def __init__(
        self,
        model_name: str,
        batch_size: int = 32,
        device: str | None = None,
        trust_remote_code: bool = False,
    ):
        SentenceTransformer = _lazy_st()
        import torch

        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        self.model = SentenceTransformer(
            model_name, device=device, trust_remote_code=trust_remote_code
        )
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device
        self.name = model_name.split("/")[-1]

    def embed(
        self,
        texts: List[str],
        prompt_name: str | None = None,
        task: str | None = None,
    ) -> np.ndarray:
        kwargs = {
            "batch_size": self.batch_size,
            "show_progress_bar": False,
            "normalize_embeddings": True,
        }
        if prompt_name:
            kwargs["prompt_name"] = prompt_name
        if task:
            kwargs["task"] = task
        return self.model.encode(texts, **kwargs).astype(np.float32)

    def embed_timed(
        self,
        texts: List[str],
        prompt_name: str | None = None,
        task: str | None = None,
    ) -> tuple[np.ndarray, float]:
        start = time.perf_counter()
        embs = self.embed(texts, prompt_name=prompt_name, task=task)
        return embs, time.perf_counter() - start
