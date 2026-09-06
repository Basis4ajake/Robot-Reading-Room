from local_knowledge_library.query_planner import RuleBasedQueryPlanner


def test_plans_superlative_recipe_questions_as_aggregate():
    planner = RuleBasedQueryPlanner()
    assert planner.plan("Which recipe uses the fewest ingredients?") == "aggregate_superlative"
    assert planner.plan("Which recipe is cheapest to make?") == "aggregate_superlative"


def test_plain_lookup_is_unaffected():
    planner = RuleBasedQueryPlanner()
    assert planner.plan("What ingredients are in the risotto?") == "lookup"
    assert planner.plan("How do I make gnocchi?") == "lookup"
