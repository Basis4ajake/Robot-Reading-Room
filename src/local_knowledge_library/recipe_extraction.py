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

_HEADING_RE = re.compile(r"^[A-Z][A-Z '\"=.\-]{2,60}$")
_MIN_SEGMENT_WORDS = 15


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
