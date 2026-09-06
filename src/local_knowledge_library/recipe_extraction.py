"""Recipe segmentation and structured-fact extraction.

Prototype for Phase 6 (see docs/review-repository-propose-enhancements-snug-sloth.md).
Not wired into IngestionPipeline yet. `segment_recipes` is deliberately narrow:
it only recognizes the "one ALL-CAPS heading per recipe" layout common in
older/public-domain cookbooks, not prose documents in general.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

from .abstracts import LLMProvider
from .models import RecipeFact

_HEADING_RE = re.compile(r"^[A-Z][A-Z '\"=.\-]{2,60}$")
_MIN_SEGMENT_WORDS = 15
_EXCERPT_LENGTH = 300


@dataclass
class RecipeSegment:
    name: str
    text: str
    start_line: int


@dataclass
class RecipeFacts:
    recipe_name: str
    ingredients: List[str] = field(default_factory=list)
    step_count: int = 0

    @property
    def ingredient_count(self) -> int:
        return len(self.ingredients)


def segment_recipes(text: str, min_segment_words: int = _MIN_SEGMENT_WORDS) -> List[RecipeSegment]:
    """Split text into recipe-sized units at ALL-CAPS heading lines.

    Drops segments shorter than `min_segment_words` — in practice this is what
    filters out title-page fragments and back-of-book index entries, which are
    heading-shaped but not recipes.
    """
    lines = text.split("\n")
    headings = [
        (i, line.strip())
        for i, line in enumerate(lines)
        if _HEADING_RE.match(line.strip())
    ]

    segments: List[RecipeSegment] = []
    for idx, (start_line, name) in enumerate(headings):
        end_line = headings[idx + 1][0] if idx + 1 < len(headings) else len(lines)
        body = " ".join(lines[start_line + 1 : end_line]).strip()
        if len(body.split()) < min_segment_words:
            continue
        segments.append(RecipeSegment(name=name, text=body, start_line=start_line))

    return segments


_EXTRACTION_PROMPT_TEMPLATE = """Read this recipe and extract structured facts. Respond with ONLY valid JSON, no other text, in this exact shape:
{{"recipe_name": string, "ingredients": [string, ...], "step_count": integer}}

Rules:
- "ingredients" lists each distinct ingredient mentioned, in lowercase, no quantities (e.g. "salt" not "a small quantity of salt").
- "step_count" is the number of distinct preparation actions described.

Recipe:
{recipe_text}
"""


def extract_recipe_facts(segment: RecipeSegment, llm: LLMProvider) -> Optional[RecipeFacts]:
    """Ask the LLM for structured facts about one recipe segment.

    Returns None (rather than raising) on a malformed/non-JSON response, since
    this runs once per recipe across a whole book and one bad response
    shouldn't abort the rest.
    """
    prompt = _EXTRACTION_PROMPT_TEMPLATE.format(recipe_text=f"{segment.name}\n\n{segment.text}")
    response = llm.generate(prompt, max_tokens=400)

    try:
        parsed = json.loads(response)
        return RecipeFacts(
            recipe_name=str(parsed["recipe_name"]),
            ingredients=[str(i) for i in parsed["ingredients"]],
            step_count=int(parsed["step_count"]),
        )
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None


def extract_all(segments: Sequence[RecipeSegment], llm: LLMProvider) -> List[RecipeFacts]:
    """Extract facts for every segment, silently skipping ones that fail."""
    facts = []
    for segment in segments:
        result = extract_recipe_facts(segment, llm)
        if result is not None:
            facts.append(result)
    return facts


def to_recipe_fact(
    segment: RecipeSegment,
    facts: RecipeFacts,
    *,
    library_id: str,
    source_id: str,
    document_id: str,
    page_number: Optional[int] = None,
) -> RecipeFact:
    """Attach ingestion provenance to an extraction result for persistence."""
    excerpt = segment.text[:_EXCERPT_LENGTH].strip()
    if len(segment.text) > _EXCERPT_LENGTH:
        excerpt += "..."
    return RecipeFact(
        library_id=library_id,
        source_id=source_id,
        document_id=document_id,
        recipe_name=facts.recipe_name,
        ingredients=facts.ingredients,
        step_count=facts.step_count,
        source_excerpt=excerpt,
        page_number=page_number,
    )


@dataclass
class AggregateQueryPlan:
    """How to answer a superlative/aggregate recipe question from RecipeFact
    data instead of vector search - see qa.py's GroundedQA._answer_aggregate_query.
    """

    answerable: bool
    metric: Optional[str] = None  # "ingredient_count" | "step_count"
    direction: Optional[str] = None  # "min" | "max"
    reason: Optional[str] = None  # set when answerable is False, e.g. "cost_not_tracked"


_COST_KEYWORDS = ("cheap", "cost", "price", "expensive", "afford", "budget")
_MIN_KEYWORDS = (
    "fewest", "least", "simplest", "simplify", "easiest", "easy",
    "quickest", "quick", "fastest", "smallest", "lowest", "minimum", "shortest",
)
_MAX_KEYWORDS = ("most", "highest", "greatest", "largest", "maximum", "longest", "hardest", "most complex")
_INGREDIENT_KEYWORDS = ("ingredient",)
_STEP_KEYWORDS = ("step", "simplest", "simplify", "easiest", "easy", "quickest", "quick", "fastest", "hardest")


def interpret_aggregate_query(query: str) -> Optional[AggregateQueryPlan]:
    """Return None for queries that aren't a superlative/aggregate recipe
    question at all (plain lookups should keep using vector search).

    Requires an explicit superlative word (a _MIN_KEYWORDS/_MAX_KEYWORDS hit,
    or a cost keyword) to trigger at all - a plain mention of "ingredients" or
    "steps" with no superlative (e.g. "what ingredients are in the risotto?")
    must NOT be misrouted here; those words only pick which metric to use
    once a superlative has already been detected.
    """
    lower = query.strip().lower()

    if any(keyword in lower for keyword in _COST_KEYWORDS):
        return AggregateQueryPlan(answerable=False, reason="cost_not_tracked")

    direction = None
    if any(keyword in lower for keyword in _MIN_KEYWORDS):
        direction = "min"
    elif any(keyword in lower for keyword in _MAX_KEYWORDS):
        direction = "max"

    if direction is None:
        return None

    metric = "ingredient_count" if any(keyword in lower for keyword in _INGREDIENT_KEYWORDS) else None
    if metric is None and any(keyword in lower for keyword in _STEP_KEYWORDS):
        metric = "step_count"

    return AggregateQueryPlan(answerable=True, metric=metric or "ingredient_count", direction=direction)
