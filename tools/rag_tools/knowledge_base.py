# File: tools/rag_tools/knowledge_base.py
# Description: Retrieves relevant QA guidance from the local text corpus without external services.
# Author Name: Debleena Nandy
# Date: 07-10-2026
"""Small deterministic TF-IDF retrieval over checked-in QA guidance."""

from __future__ import annotations

import logging
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Protocol, Sequence

logger = logging.getLogger(__name__)

TOKEN = re.compile(r"[a-z0-9]+")
STOP = {
    "the", "a", "an", "and", "or", "to", "of", "in", "on", "for", "is", "are", "be", "with", "as", "i", "want",
    "so", "that", "it", "my", "can", "must", "should", "when", "by", "at", "this", "from", "not",
}
Tokens = List[str]
Vectors = List[List[float]]


def tokenize(text: str) -> Tokens:
    tokens = []
    for token in TOKEN.findall(text.lower()):
        if token in STOP:
            continue
        for suffix in ("ing", "ed", "es", "s"):  # light stemming, no extra dependency
            if len(token) > 4 and token.endswith(suffix):
                token = token[: -len(suffix)]
                break
        tokens.append(token)
    return tokens


@dataclass(frozen=True)
class KnowledgeChunk:
    id: str
    kind: str  # checklist | failure_pattern
    title: str
    text: str
    tags: tuple = ()


@dataclass
class RetrievedChunk:
    chunk: KnowledgeChunk
    score: float


Hits = List[RetrievedChunk]
Chunks = List[KnowledgeChunk]


class Embedder(Protocol):
    def embed(self, texts: List[str], model: str) -> Vectors: ...


@dataclass
class KnowledgeBase:
    """Local vector store over QA checklists and known failure patterns.

    Default retriever: TF-IDF cosine (deterministic, offline). With an embedder (Ollama /api/embed) dense
    vectors are used instead; any embedding failure falls back to TF-IDF.
    """

    chunks: List[KnowledgeChunk]
    embedder: Optional[Embedder] = None
    embedding_model: str = "nomic-embed-text"
    _idf: Dict[str, float] = field(default_factory=dict, init=False)
    _vectors: List[Dict[str, float]] = field(default_factory=list, init=False)
    _dense: Optional[Vectors] = field(default=None, init=False)

    def __post_init__(self) -> None:
        documents = [tokenize(f"{c.title} {c.title} {' '.join(c.tags)} {c.text}") for c in self.chunks]
        df: Counter = Counter()
        for doc in documents:
            df.update(set(doc))
        n = max(len(documents), 1)
        self._idf = {term: math.log((1 + n) / (1 + count)) + 1 for term, count in df.items()}
        self._vectors = [self._tfidf(doc) for doc in documents]

    @classmethod
    def from_directory(cls, directory: Path | str, embedder: Optional[Embedder] = None,
                       embedding_model: str = "") -> KnowledgeBase:
        root = Path(directory)
        chunks: List[KnowledgeChunk] = []
        if root.is_dir():
            for file in sorted(root.glob("*.md")):
                chunks.extend(parse_markdown(file.read_text(encoding="utf-8"), file.stem))
        return cls(chunks=chunks, embedder=embedder, embedding_model=embedding_model or "nomic-embed-text")

    def _tfidf(self, tokens: Sequence[str]) -> Dict[str, float]:
        counts = Counter(tokens)
        vector = {t: (c / max(len(tokens), 1)) * self._idf.get(t, 0.0) for t, c in counts.items()}
        norm = math.sqrt(sum(v * v for v in vector.values())) or 1.0
        return {t: v / norm for t, v in vector.items()}

    def search(self, query: str, top_k: int = 3, kind: Optional[str] = None, min_score: float = 0.05) -> Hits:
        candidates = [i for i, c in enumerate(self.chunks) if kind is None or c.kind == kind]
        if not candidates:
            return []
        scores = self._dense_scores(query, candidates) or self._sparse_scores(query, candidates)
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        return [RetrievedChunk(self.chunks[i], round(s, 4)) for i, s in ranked[:top_k] if s >= min_score]

    def _sparse_scores(self, query: str, candidates: List[int]) -> Dict[int, float]:
        q = self._tfidf(tokenize(query))
        return {i: sum(w * self._vectors[i].get(t, 0.0) for t, w in q.items()) for i in candidates}

    def _dense_scores(self, query: str, candidates: List[int]) -> Dict[int, float] | None:
        if self.embedder is None:
            return None
        try:
            if self._dense is None:
                self._dense = self.embedder.embed([f"{c.title}\n{c.text}" for c in self.chunks], self.embedding_model)
            q = self.embedder.embed([query], self.embedding_model)[0]
        except Exception as exc:  # embedding service down or model missing: degrade to TF-IDF
            logger.warning("Embedding retrieval failed, using TF-IDF: %s", exc)
            self.embedder = None
            return None
        dense = self._dense
        return {i: _cosine(q, dense[i]) for i in candidates}


def _cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0


def parse_markdown(text: str, source: str) -> Chunks:
    """Each '## [kind] Title' section becomes a chunk; an optional 'tags:' line adds keywords."""
    chunks = []
    for block in re.split(r"^## ", text, flags=re.MULTILINE)[1:]:
        header, _, body = block.partition("\n")
        match = re.match(r"\[(checklist|failure_pattern)\]\s*(.+)", header.strip())
        if not match:
            continue
        tags: tuple = ()
        lines = []
        for line in body.strip().splitlines():
            if line.lower().startswith("tags:"):
                tags = tuple(t.strip() for t in line.split(":", 1)[1].split(",") if t.strip())
            else:
                lines.append(line)
        slug = re.sub(r"[^a-z0-9]+", "-", match.group(2).lower()).strip("-")
        chunks.append(KnowledgeChunk(f"{source}:{slug}", match.group(1), match.group(2).strip(),
                                     "\n".join(lines).strip(), tags))
    return chunks


def checklist_concerns(chunk: KnowledgeChunk) -> Tokens:
    """Bullet lines of a checklist are concrete concerns, e.g. '- idempotent order creation'."""
    return [line[2:].strip() for line in chunk.text.splitlines() if line.startswith("- ")]