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
