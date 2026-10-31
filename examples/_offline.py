"""
Offline stand-ins used by the ``examples/offline_*.py`` scripts.

Nothing here downloads a model or calls an API, so the offline examples run anywhere
in a second or two. Swap these for real components in your application:

* :class:`HashingEmbeddings` -> ``HuggingFaceEmbeddings`` / ``OpenAIEmbeddings`` / ...
* :class:`ExtractiveStubLLM` -> ``ChatOpenAI`` / ``ChatMistralAI`` / ``ChatOllama`` / ...

The stub LLM is deliberately simple and rule-based. It understands the three prompt
shapes ragfallback sends (query rewriting, answer generation, confidence rating) and
answers them by extracting sentences from the retrieved documents. It is a test double
that lets the retry/fallback machinery run deterministically, not a language model.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

SAMPLE_KB = Path(__file__).resolve().parent / "sample_kb"

_WORD = re.compile(r"[a-z0-9][a-z0-9\-]*")
_STOP = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "can",
        "do",
        "does",
        "for",
        "from",
        "get",
        "have",
        "how",
        "i",
        "in",
        "is",
        "it",
        "may",
        "my",
        "of",
        "on",
        "or",
        "the",
        "their",
        "there",
        "to",
        "was",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "will",
        "with",
        "you",
        "your",
    ]
)

# What a real LLM "knows": everyday phrasing -> the vocabulary the docs actually use.
SYNONYMS: dict[str, list[str]] = {
    "money back": ["refund window", "refund"],
    "charged twice": ["billing dispute", "charge"],
    "too many requests": ["rate limit", "429"],
    "login key": ["api token", "token"],
}


def words(text: str) -> list[str]:
    """Lower-cased content words (stopwords removed)."""
    return [w for w in _WORD.findall(text.lower()) if w not in _STOP]


class HashingEmbeddings(Embeddings):
    """Bag-of-words hashed into a fixed number of buckets (lexical, deterministic)."""

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * self.dim
        for w in words(text):
            v[int(hashlib.md5(w.encode()).hexdigest(), 16) % self.dim] += 1.0
        if not any(v):
            v[0] = 1e-6  # avoid a zero vector for empty / stopword-only text
        return v

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]


class _Reply:
    def __init__(self, content: str) -> None:
        self.content = content
        # Mimic OpenAI-style usage metadata so CostTracker has something to price.
        self.response_metadata = {
            "token_usage": {"prompt_tokens": 400, "completion_tokens": 40},
        }


def _expand(question: str) -> set[str]:
    terms = set(words(question))
    low = question.lower()
    for phrase, alts in SYNONYMS.items():
        if phrase in low:
            for alt in alts:
                terms.update(words(alt))
    return terms


class ExtractiveStubLLM:
    """Deterministic, rule-based stand-in for a chat model (see module docstring)."""

    model_name = "gpt-4o-mini"  # used only for CostTracker pricing

    def __init__(self, min_overlap: int = 2) -> None:
        self.min_overlap = min_overlap
        self.calls = 0

    def invoke(self, messages: Any, **_: Any) -> _Reply:
        self.calls += 1
        prompt = "\n".join(getattr(m, "content", str(m)) for m in messages)
        if "alternative ways to ask" in prompt:
            return _Reply(self._rewrite(prompt))
        if "Rate the confidence" in prompt:
            return _Reply(self._rate(prompt))
        return _Reply(self._answer(prompt))

    def _rewrite(self, prompt: str) -> str:
        m = re.search(r'Original question: "(.*?)"', prompt)
        question = (m.group(1) if m else "").lower()
        rewrites = [alt for phrase, alts in SYNONYMS.items() if phrase in question for alt in alts]
        return json.dumps(rewrites or [question])

    def _answer(self, prompt: str) -> str:
        q = re.search(r"Question: (.*)", prompt)
        question = q.group(1) if q else ""
        docs = prompt.split("Documents:", 1)[-1]
        terms = _expand(question)
        best, best_score, best_src = "", 0, ""
        for block in re.split(r"Document \d+ \(from ", docs)[1:]:
            source, _, body = block.partition("):\n")
            for sentence in re.split(r"(?<=[.!?])\s+", body):
                score = len(terms & set(words(sentence)))
                if score > best_score:
                    best, best_score, best_src = sentence.strip(), score, source
        if best_score < self.min_overlap:
            return json.dumps({"answer": "Not found", "source": ""})
        return json.dumps({"answer": best, "source": best_src})

    def _rate(self, prompt: str) -> str:
        answer = re.search(r"Answer: (.*)", prompt)
        text = (answer.group(1) if answer else "").strip().lower()
        return "0.1" if not text or text.startswith("not found") else "0.9"


def load_paragraphs(kb_dir: Path = SAMPLE_KB) -> list[Document]:
    """One Document per paragraph of the bundled ``sample_kb/*.md`` files."""
    docs: list[Document] = []
    for path in sorted(kb_dir.glob("*.md")):
        paragraphs = [p.strip() for p in path.read_text(encoding="utf-8").split("\n\n")]
        for i, para in enumerate(p for p in paragraphs if p and not p.startswith("#")):
            docs.append(
                Document(page_content=para, metadata={"source": path.name, "chunk_index": i})
            )
    return docs
