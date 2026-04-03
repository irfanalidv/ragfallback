"""Synthetic query transforms for stress-testing retrieval and routing."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence


@dataclass
class SimQuery:
    """One original query mapped to a transformed variant and a scenario label."""

    original: str
    transformed: str
    query_type: str


class QuerySimulator:
    """Produce short, long, ambiguous, and out-of-domain query variants."""

    _DEFAULT_OOD: List[str] = [
        "What is the weather in Tokyo?",
        "How do I bake sourdough bread?",
        "What is the GDP of Norway?",
        "Who won the 1998 FIFA World Cup?",
        "How do I change a car tyre?",
    ]

    _STOPWORDS = frozenset(
        {
            "the",
            "a",
            "an",
            "is",
            "are",
            "was",
            "were",
            "what",
            "how",
            "why",
            "when",
            "where",
            "who",
            "do",
            "does",
        }
    )

    def __init__(self, ood_topics: Optional[List[str]] = None) -> None:
        """Use ``ood_topics`` for out-of-domain replacement or built-in defaults."""
        self._ood_topics: List[str] = (
            list(ood_topics) if ood_topics is not None else list(self._DEFAULT_OOD)
        )

    def _short_keyword(self, query: str) -> str:
        """Keep up to two content tokens, skipping common stopwords when possible."""
        tokens = query.split()
        if not tokens:
            return query
        non_stop = [t for t in tokens if t.lower() not in self._STOPWORDS]
        if len(non_stop) >= 2:
            return " ".join(non_stop[:2])
        return " ".join(tokens[:2])

    def _long_nl(self, query: str) -> str:
        """Wrap the query in a verbose instruction prefix and suffix."""
        return (
            "Please explain in detail: "
            + query
            + " Provide a comprehensive answer with examples if possible."
        )

    def _ambiguous(self, query: str) -> str:
        """Drop probable proper nouns (capitalized tokens); keep original if too short."""
        tokens = query.split()
        if not tokens:
            return query
        kept = [t for t in tokens if not (t[:1].isupper() and len(t) > 1)]
        if len(kept) < 3:
            return query
        return " ".join(kept)

    def _out_of_domain(self, query: str) -> str:
        """Replace the query with a random unrelated topic (ignores ``query``)."""
        if not self._ood_topics:
            return query
        return random.choice(self._ood_topics)

    def simulate(
        self,
        queries: Sequence[str],
        mix: Optional[Dict[str, float]] = None,
    ) -> List[SimQuery]:
        """Assign each query a random transform type weighted by ``mix``."""
        default_mix: Dict[str, float] = {
            "short_keyword": 0.25,
            "long_nl": 0.25,
            "ambiguous": 0.25,
            "out_of_domain": 0.25,
        }
        m = dict(default_mix if mix is None else mix)
        types = list(m.keys())
        weights = [float(m[t]) for t in types]
        total = sum(weights)
        if abs(total - 1.0) > 0.01:
            raise ValueError(
                "mix weights must sum to 1.0 within tolerance 0.01, got {}".format(
                    total
                )
            )
        handlers = {
            "short_keyword": self._short_keyword,
            "long_nl": self._long_nl,
            "ambiguous": self._ambiguous,
            "out_of_domain": self._out_of_domain,
        }
        out: List[SimQuery] = []
        for q in queries:
            kind = random.choices(types, weights=weights, k=1)[0]
            fn = handlers[kind]
            out.append(SimQuery(original=q, transformed=fn(q), query_type=kind))
        return out

    def simulate_unhappy_paths(self, queries: Sequence[str]) -> List[SimQuery]:
        """Emit all four transform types for every input query."""
        order = ("short_keyword", "long_nl", "ambiguous", "out_of_domain")
        handlers = {
            "short_keyword": self._short_keyword,
            "long_nl": self._long_nl,
            "ambiguous": self._ambiguous,
            "out_of_domain": self._out_of_domain,
        }
        out: List[SimQuery] = []
        for q in queries:
            for kind in order:
                out.append(
                    SimQuery(
                        original=q,
                        transformed=handlers[kind](q),
                        query_type=kind,
                    )
                )
        return out
