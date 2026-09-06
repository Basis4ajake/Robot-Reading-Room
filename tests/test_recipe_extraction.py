import json

from local_knowledge_library.abstracts import LLMProvider
from local_knowledge_library.recipe_extraction import (
    RecipeFacts,
    RecipeSegment,
    extract_all,
    extract_recipe_facts,
    interpret_aggregate_query,
    segment_recipes,
    to_recipe_fact,
)


class FakeLLM(LLMProvider):
    """Returns a scripted response per call, in order."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.prompts = []

    def generate(self, prompt: str, max_tokens: int = 512) -> str:
        self.prompts.append(prompt)
        return self._responses.pop(0)

    def embed_text(self, texts):
        raise NotImplementedError


def test_segment_recipes_splits_on_all_caps_headings():
    text = (
        "GNOCCHI\n\n"
        "Prepare a certain quantity of boiled potatoes and mix with cheese, "
        "eggs, salt and nutmeg then roll into little sticks and boil them "
        "until they float to the top of the pot.\n\n"
        "VEGETABLE SOUP\n\n"
        "Cut carrots celery and onion into small pieces and simmer them in "
        "broth for one hour until every vegetable is completely tender.\n"
    )

    segments = segment_recipes(text)

    assert [s.name for s in segments] == ["GNOCCHI", "VEGETABLE SOUP"]
    assert "boiled potatoes" in segments[0].text
    assert "carrots celery" in segments[1].text


def test_segment_recipes_drops_short_segments_like_title_pages_and_index_entries():
    text = (
        "THE ITALIAN COOK BOOK\n\n"
        "PASTRIES\n\n"
        "GNOCCHI\n\n"
        "Prepare a certain quantity of boiled potatoes and mix with cheese, "
        "eggs, salt and nutmeg then roll into little sticks and boil them "
        "until they float to the top of the pot of boiling water.\n\n"
        "NUMBERS REFER TO RECIPES\n"
    )

    segments = segment_recipes(text)

    assert [s.name for s in segments] == ["GNOCCHI"]


def test_segment_recipes_respects_custom_min_word_threshold():
    text = "HEADING ONE\n\nshort body\n\nHEADING TWO\n\nalso short\n"

    assert segment_recipes(text, min_segment_words=1) != []
    assert segment_recipes(text, min_segment_words=50) == []


def test_extract_recipe_facts_parses_valid_json_response():
    segment = RecipeSegment(name="GNOCCHI", text="potatoes, cheese, eggs", start_line=0)
    llm = FakeLLM([
        json.dumps({
            "recipe_name": "Gnocchi",
            "ingredients": ["potatoes", "cheese", "eggs"],
            "step_count": 3,
        })
    ])

    facts = extract_recipe_facts(segment, llm)

    assert facts.recipe_name == "Gnocchi"
    assert facts.ingredients == ["potatoes", "cheese", "eggs"]
    assert facts.ingredient_count == 3
    assert facts.step_count == 3


def test_extract_recipe_facts_returns_none_on_malformed_json():
    segment = RecipeSegment(name="GNOCCHI", text="potatoes, cheese, eggs", start_line=0)
    llm = FakeLLM(["not json at all"])

    assert extract_recipe_facts(segment, llm) is None


def test_extract_recipe_facts_returns_none_on_missing_expected_fields():
    segment = RecipeSegment(name="GNOCCHI", text="potatoes, cheese, eggs", start_line=0)
    llm = FakeLLM([json.dumps({"recipe_name": "Gnocchi"})])

    assert extract_recipe_facts(segment, llm) is None


def test_extract_all_skips_failures_and_keeps_successes():
    segments = [
        RecipeSegment(name="A", text="...", start_line=0),
        RecipeSegment(name="B", text="...", start_line=10),
    ]
    llm = FakeLLM([
        "not json",
        json.dumps({"recipe_name": "B", "ingredients": ["x"], "step_count": 1}),
    ])

    facts = extract_all(segments, llm)

    assert len(facts) == 1
    assert facts[0].recipe_name == "B"


def test_to_recipe_fact_attaches_provenance_and_excerpt():
    segment = RecipeSegment(name="GNOCCHI", text="boiled potatoes and cheese " * 5, start_line=3)
    facts = RecipeFacts(recipe_name="Gnocchi", ingredients=["potatoes", "cheese"], step_count=2)

    record = to_recipe_fact(
        segment, facts, library_id="lib-1", source_id="src-1", document_id="doc-1", page_number=7
    )

    assert record.library_id == "lib-1"
    assert record.source_id == "src-1"
    assert record.document_id == "doc-1"
    assert record.recipe_name == "Gnocchi"
    assert record.ingredients == ["potatoes", "cheese"]
    assert record.ingredient_count == 2
    assert record.page_number == 7
    assert "boiled potatoes" in record.source_excerpt


def test_to_recipe_fact_truncates_long_segments_with_ellipsis():
    long_text = "word " * 200
    segment = RecipeSegment(name="LONG", text=long_text, start_line=0)
    facts = RecipeFacts(recipe_name="Long", ingredients=[], step_count=1)

    record = to_recipe_fact(segment, facts, library_id="l", source_id="s", document_id="d")

    assert len(record.source_excerpt) < len(long_text)
    assert record.source_excerpt.endswith("...")
    assert record.page_number is None


def test_interpret_aggregate_query_detects_fewest_ingredients():
    plan = interpret_aggregate_query("Which recipe uses the fewest ingredients?")
    assert plan.answerable is True
    assert plan.metric == "ingredient_count"
    assert plan.direction == "min"


def test_interpret_aggregate_query_detects_most_ingredients():
    plan = interpret_aggregate_query("Which recipe has the most ingredients?")
    assert plan.answerable is True
    assert plan.metric == "ingredient_count"
    assert plan.direction == "max"


def test_interpret_aggregate_query_maps_simplest_to_fewest_steps():
    plan = interpret_aggregate_query("Which recipe is simplest?")
    assert plan.answerable is True
    assert plan.metric == "step_count"
    assert plan.direction == "min"


def test_interpret_aggregate_query_flags_cost_as_unanswerable():
    plan = interpret_aggregate_query("Which recipe is cheapest to make?")
    assert plan.answerable is False
    assert plan.reason == "cost_not_tracked"


def test_interpret_aggregate_query_returns_none_for_plain_lookup():
    assert interpret_aggregate_query("What ingredients are in the risotto recipe?") is None
    assert interpret_aggregate_query("How do I make gnocchi?") is None
