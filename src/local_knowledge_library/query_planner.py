from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

class QueryPlannerStrategy(ABC):
    @abstractmethod
    def plan(self, query: str) -> str:
        raise NotImplementedError


class RuleBasedQueryPlanner(QueryPlannerStrategy):
    def plan(self, query: str) -> str:
        lower = query.strip().lower()
        if any(token in lower for token in ["summarize", "summary", "summarization"]):
            return "summarization"
        # Superlative/aggregate questions ("which recipe uses the fewest
        # ingredients") need a real computation over structured facts, not
        # similarity search - see recipe_extraction.interpret_aggregate_query,
        # which GroundedQA actually branches on. This label is for
        # transparency (shown in the prompt/response) so it isn't silently
        # misreported as "lookup" like every other query shape here.
        if any(
            token in lower
            for token in [
                "fewest", "least", "simplest", "easiest", "cheapest", "most",
                "highest", "lowest", "quickest", "fastest", "hardest",
            ]
        ):
            return "aggregate_superlative"
        if any(token in lower for token in ["compare", "difference", "versus", "vs"]):
            return "comparison"
        if any(token in lower for token in ["synthesize", "synthesis", "combine", "integration"]):
            return "synthesis"
        if any(token in lower for token in ["evidence", "support", "cited", "proof"]):
            return "evidence_analysis"
        if any(token in lower for token in ["contradict", "contradiction", "disagree"]):
            return "contradiction_analysis"
        if any(token in lower for token in ["who", "what", "when", "where", "why", "how"]):
            return "lookup"
        return "lookup"


class QueryPlanner:
    def __init__(self, strategy: QueryPlannerStrategy | None = None):
        self.strategy = strategy or RuleBasedQueryPlanner()

    def plan(self, query: str) -> str:
        return self.strategy.plan(query)
