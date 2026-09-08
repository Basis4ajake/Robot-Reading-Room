"""Recipe segmentation and structured-fact extraction.

Prototype for Phase 6 (see docs/review-repository-propose-enhancements-snug-sloth.md).
Not wired into IngestionPipeline yet. `segment_recipes` is deliberately narrow:
it recognizes "one short title line per recipe" layouts (both ALL-CAPS and
Title Case, confirmed against two real books with different conventions),
not prose documents in general or a title convention with lowercase
connector words ("Soup of the Day") - not yet seen in a real test book.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

from .abstracts import LLMProvider
from .models import RecipeFact

_MIN_SEGMENT_WORDS = 15
_EXCERPT_LENGTH = 300

# A heading candidate line must consist only of these characters - notably no
# digits (measurement/ingredient lines) and no periods (bylines like
# "Mrs. A. D. Savage.", abbreviations, sentence-ending prose).
_ALLOWED_HEADING_CHARS_RE = re.compile(r"^[A-Za-z\s'\"()\-]+$")
_MAX_HEADING_WORDS = 8
# The shortest real single-word title seen in either confirmed test book
# ("PASTE") is 5 characters. Below that, on real data, a single all-caps
# token has only ever been OCR noise ("WIA", "AAT", "DREN", "NNN", ...).
_MIN_SINGLE_WORD_HEADING_LENGTH = 5
_VOWEL_RE = re.compile(r"[aeiouAEIOU]")


def _looks_like_heading(line: str) -> bool:
    """Case-agnostic structural heading check - a title is a short line where
    every word is either ALL-CAPS or Capitalized (never a lowercase-led
    word), which real recipe titles satisfy in both conventions confirmed so
    far and ordinary prose/bylines/ingredient lines do not.
    """
    stripped = line.strip()
    if not stripped or not _ALLOWED_HEADING_CHARS_RE.match(stripped):
        return False

    words = [w.strip("'\"") for w in re.split(r"[\s()\-]+", stripped) if w.strip("'\"")]
    if not words or len(words) > _MAX_HEADING_WORDS:
        return False

    for word in words:
        if not word[0].isupper():
            return False
        rest = word[1:]
        if rest and not (rest.islower() or rest.isupper()):
            return False

    if len(words) == 1:
        word = words[0]
        if len(word) < _MIN_SINGLE_WORD_HEADING_LENGTH or not _VOWEL_RE.search(word):
            return False

    return True


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
    """Split text into recipe-sized units at short title lines (ALL-CAPS or
    Title Case - see `_looks_like_heading`).

    Drops segments shorter than `min_segment_words` — in practice this is what
    filters out title-page fragments and back-of-book index entries, which are
    heading-shaped but not recipes.
    """
    lines = text.split("\n")
    raw_headings = [
        (i, line.strip())
        for i, line in enumerate(lines)
        if _looks_like_heading(line)
    ]

    # Accepting Title Case (needed for real books that don't use ALL-CAPS)
    # also picks up two different real patterns where a heading-shaped line
    # sits immediately under another one, with nothing but blank lines
    # between them:
    #   1. A parenthetical subtitle under a main title (e.g. an
    #      Italian-language alt-name right below its English title) - the
    #      FIRST (real) name should win, and the parenthetical line is pure
    #      separator, never renaming the group.
    #   2. A nesting chain (book title > section header > recipe title,
    #      typically front-matter noise) - the LAST, most specific name
    #      should win, since it's the one actually closest to the real body.
    # Handle both by letting any non-parenthetical heading in a chain
    # supersede an earlier name, while a parenthetical one only extends the
    # chain without renaming it.
    headings: List[tuple] = []
    group_name: Optional[str] = None
    group_body_start: Optional[int] = None
    last_heading_line: Optional[int] = None
    for start_line, name in raw_headings:
        is_subtitle = name.startswith("(") and name.endswith(")")
        if last_heading_line is not None:
            between = lines[last_heading_line + 1 : start_line]
            if not any(line.strip() for line in between):
                last_heading_line = start_line
                group_body_start = start_line
                if not is_subtitle:
                    group_name = name
                continue
        if group_name is not None:
            headings.append((group_body_start, group_name))
        group_name = name
        group_body_start = start_line
        last_heading_line = start_line
    if group_name is not None:
        headings.append((group_body_start, group_name))

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
    """Attach ingestion provenance to an extraction result for persistence.

    Uses `segment.name` (the already-verified heading from `segment_recipes`)
    rather than the LLM's own self-reported `facts.recipe_name` - a real
    30-minute full-book run surfaced the LLM occasionally misreporting the
    name of the recipe it was just given (e.g. "QUEEN'S SOUP" -> a fact
    persisted as "Queen's Soup" with different, wrong ingredient/step
    counts - no such heading exists elsewhere in the source). The heading is
    reliable by construction; the LLM is only needed for the facts about it.
    """
    excerpt = segment.text[:_EXCERPT_LENGTH].strip()
    if len(segment.text) > _EXCERPT_LENGTH:
        excerpt += "..."
    return RecipeFact(
        library_id=library_id,
        source_id=source_id,
        document_id=document_id,
        recipe_name=segment.name,
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

    Also requires the query to say "recipe" explicitly. Without this, a bare
    superlative word anywhere in an ordinary in-book question - e.g. "What's
    the quickest way to knead this dough?" ("quickest" is in _MIN_KEYWORDS)
    or "What's the hardest step in this method?" ("hardest" is in
    _MAX_KEYWORDS) - got misrouted into a cross-recipe aggregate answer even
    though nothing was actually asking to compare recipes. Every real
    aggregate question seen so far, including this project's own motivating
    test case ("which recipe uses the fewest ingredients"), says "recipe"
    explicitly; a query that doesn't just falls through to normal retrieval
    instead, which is the safe default. Known remaining gap, not fixed here:
    a query like "Is this recipe easy to make?" still says "recipe" while
    asking about ONE specific recipe, not comparing across the book - telling
    those apart needs more than a keyword check.
    """
    lower = query.strip().lower()

    if "recipe" not in lower:
        return None

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
