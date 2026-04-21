import hashlib
import os
import threading

import numpy as np
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import HashingVectorizer


STRUCTURAL_TOKEN = "__STRUCTURAL__"
UNKNOWN_TOKEN = "__UNKNOWN__"


def _safe_label_text(label):
    if label is None:
        return STRUCTURAL_TOKEN
    text = str(label).strip()
    if not text:
        return STRUCTURAL_TOKEN
    if text.lower() in {"unk", "unknown", "none", "null", "nan"}:
        return UNKNOWN_TOKEN
    return text


def _stable_unit_vector(token, dim):
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=16).digest()
    seed = int.from_bytes(digest[:8], byteorder="big", signed=False) % (2**32 - 1)
    rng = np.random.RandomState(seed)
    vec = rng.normal(0.0, 1.0, size=dim).astype(np.float32)
    norm = np.linalg.norm(vec)
    if norm == 0:
        return np.zeros(dim, dtype=np.float32)
    return vec / norm


class LabelProjection(nn.Module):
    """Trainable adapter that compresses frozen text features for the GNN."""

    def __init__(self, input_dim, output_dim, hidden_dim=None, dropout=0.0):
        super(LabelProjection, self).__init__()
        hidden_dim = hidden_dim or max(output_dim, min(256, input_dim))
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, x):
        return self.net(x)


class UniversalLabelEncoder:
    """
    Frozen label encoder
    """

    def __init__(
        self,
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        cache_dir=None,
        backend="auto",
        embedding_dim=384,
        device=None,
    ):
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.requested_backend = backend
        self.embedding_dim = int(embedding_dim)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._backend = None
        self._model = None
        self._memory_cache = {}
        self._lock = threading.Lock()
        self._hash_vectorizer = HashingVectorizer(
            n_features=self.embedding_dim,
            alternate_sign=False,
            analyzer="char_wb",
            ngram_range=(3, 5),
            norm=None,
            lowercase=True,
        )
        self._structural_embedding = _stable_unit_vector(STRUCTURAL_TOKEN, self.embedding_dim)
        self._unknown_embedding = _stable_unit_vector(UNKNOWN_TOKEN, self.embedding_dim)
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)

    @property
    def backend(self):
        if self._backend is None:
            self._backend = self._resolve_backend()
        return self._backend

    def _resolve_backend(self):
        if self.requested_backend == "cache_only":
            return "cache_only"
        if self.requested_backend == "hashing":
            return "hashing"
        if self.requested_backend in {"auto", "sentence_transformers"}:
            try:
                from sentence_transformers import SentenceTransformer  # noqa: F401
                return "sentence_transformers"
            except Exception:
                if self.requested_backend == "sentence_transformers":
                    raise
        return "hashing"

    def _ensure_model(self):
        if self.backend != "sentence_transformers":
            return None
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    def _cache_key(self, label_text):
        return hashlib.blake2b(label_text.encode("utf-8"), digest_size=16).hexdigest()

    def _cache_path(self, label_text):
        if not self.cache_dir:
            return None
        return os.path.join(self.cache_dir, self._cache_key(label_text) + ".npy")

    def _load_cached(self, label_text):
        if label_text in self._memory_cache:
            return self._memory_cache[label_text]
        cache_path = self._cache_path(label_text)
        if cache_path and os.path.exists(cache_path):
            arr = np.load(cache_path).astype(np.float32)
            self._memory_cache[label_text] = arr
            return arr
        return None

    def _store_cached(self, label_text, vec):
        self._memory_cache[label_text] = vec
        cache_path = self._cache_path(label_text)
        if cache_path and not os.path.exists(cache_path):
            np.save(cache_path, vec.astype(np.float32))

    def _encode_special(self, label_text):
        if label_text == STRUCTURAL_TOKEN:
            return self._structural_embedding.copy()
        if label_text == UNKNOWN_TOKEN:
            return self._unknown_embedding.copy()
        return None

    def _encode_hashed(self, texts):
        matrix = self._hash_vectorizer.transform(texts).astype(np.float32).toarray()
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return matrix / norms

    def _encode_sentence_transformers(self, texts):
        model = self._ensure_model()
        vectors = model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return vectors.astype(np.float32)

    def encode_many(self, labels):
        label_texts = [_safe_label_text(label) for label in labels]
        out = [None] * len(label_texts)
        uncached_texts = []
        uncached_positions = []

        with self._lock:
            for i, label_text in enumerate(label_texts):
                special = self._encode_special(label_text)
                if special is not None:
                    out[i] = special
                    continue
                cached = self._load_cached(label_text)
                if cached is not None:
                    out[i] = cached
                else:
                    if self.backend == "cache_only":
                        raise FileNotFoundError(
                            "Missing cached label embedding for '{}' in cache_dir='{}'".format(
                                label_text, self.cache_dir
                            )
                        )
                    uncached_texts.append(label_text)
                    uncached_positions.append(i)

            if uncached_texts:
                if self.backend == "sentence_transformers":
                    encoded = self._encode_sentence_transformers(uncached_texts)
                else:
                    encoded = self._encode_hashed(uncached_texts)
                for pos, label_text, vec in zip(uncached_positions, uncached_texts, encoded):
                    vec = vec.astype(np.float32)
                    self._store_cached(label_text, vec)
                    out[pos] = vec

        return torch.tensor(np.stack(out, axis=0), dtype=torch.float)
