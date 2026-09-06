import json

from local_knowledge_library.abstracts import LLMProvider
from local_knowledge_library.recipe_extraction import (
    RecipeSegment,
    extract_all,
    extract_recipe_facts,
    segment_recipes,
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
