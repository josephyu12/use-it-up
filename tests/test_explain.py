"""Tests for Stage 3: Explanation engine."""

from __future__ import annotations

from pathlib import Path

import pytest

from useitup.cbr import AdaptedRecipe, AdaptationEntry, CBRAdapter, CBRMatch, CBRRetriever, FeatureBreakdown
from useitup.data_loader import load_recipes
from useitup.explain import (
    Explanation,
    _build_cbr_trace,
    _build_counterfactual,
    _build_goal_trace,
    _build_ingredient_utilization_report,
    _pantry_unused,
    _pantry_used,
    generate_explanation,
    render_explanation,
)
from useitup.matching import (
    DecisionEntry,
    FilterEngine,
    FilterResult,
    IngredientScorer,
    ScoredRecipe,
)
from useitup.pipeline import run_pipeline
from useitup.profile import SoftPreferences, UserProfile, RatingEntry
from useitup.schemas import Ingredient, Recipe

_DATA_DIR = Path(__file__).parent.parent / "data"
_PROFILES_DIR = _DATA_DIR / "profiles"
_FIXTURE_PATH = Path(__file__).parent / "fixtures" / "expected_explanation.md"


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_recipes() -> list[Recipe]:
    return load_recipes(_DATA_DIR / "recipes_sample.json")


@pytest.fixture
def demo_profile() -> UserProfile:
    return UserProfile.model_validate_json(
        (_PROFILES_DIR / "demo_user.json").read_text(encoding="utf-8")
    )


@pytest.fixture
def simple_recipe() -> Recipe:
    """A minimal recipe with known ingredients."""
    return Recipe(
        id="test-r1",
        name="Egg Fried Rice",
        ingredients=[
            Ingredient(name="eggs", quantity=2.0, unit=None, category="protein"),
            Ingredient(name="brown rice", quantity=200.0, unit="g", category="grain"),
            Ingredient(name="garlic", quantity=2.0, unit="cloves", category="vegetable"),
            Ingredient(name="soy sauce", quantity=1.0, unit="tbsp", category="condiment"),
            Ingredient(name="sesame oil", quantity=1.0, unit="tsp", category="fat"),
        ],
        cuisine="Asian",
        dietary_tags=["dairy-free", "nut-free", "gluten-free"],
        prep_time_min=5,
        cook_time_min=10,
        difficulty=1,
        nutrition={"calories": 380.0, "protein_g": 14.0, "carbs_g": 52.0, "fat_g": 10.0},
        flavor_profile=["savory", "umami"],
        instructions=["Cook rice.", "Stir-fry garlic in sesame oil.", "Add rice and eggs; toss with soy sauce."],
    )


@pytest.fixture
def simple_profile() -> UserProfile:
    """A profile with pantry that partially overlaps simple_recipe."""
    return UserProfile(
        user_id="test_user",
        hard_constraints=["gluten-free"],
        soft_preferences=SoftPreferences(
            max_prep_time_min=30,
            preferred_cuisines=["Asian"],
            goals=["high_protein"],
        ),
        pantry=["eggs", "garlic", "brown rice", "olive oil", "spinach"],
    )


@pytest.fixture
def simple_filter_result(simple_recipe: Recipe, simple_profile: UserProfile) -> FilterResult:
    engine = FilterEngine()
    return engine.run([simple_recipe], simple_profile)


@pytest.fixture
def simple_adapted(simple_recipe: Recipe) -> AdaptedRecipe:
    return AdaptedRecipe(recipe=simple_recipe, adaptations=[])


@pytest.fixture
def cold_start_profile() -> UserProfile:
    """A profile with no rating history."""
    return UserProfile(
        user_id="cold_user",
        hard_constraints=[],
        soft_preferences=SoftPreferences(preferred_cuisines=["Asian"], goals=[]),
        pantry=["eggs", "garlic"],
    )


# ---------------------------------------------------------------------------
# Pantry helpers
# ---------------------------------------------------------------------------


def test_pantry_used_exact_match(simple_recipe: Recipe) -> None:
    pantry = ["eggs", "brown rice", "garlic"]
    used = _pantry_used(simple_recipe, pantry)
    assert set(used) == {"eggs", "brown rice", "garlic"}


def test_pantry_unused_correct(simple_recipe: Recipe) -> None:
    pantry = ["eggs", "garlic", "spinach"]
    unused = _pantry_unused(simple_recipe, pantry)
    assert "spinach" in unused
    assert "eggs" not in unused
    assert "garlic" not in unused


def test_pantry_used_plus_unused_covers_full_pantry(simple_recipe: Recipe) -> None:
    pantry = ["eggs", "garlic", "brown rice", "olive oil", "spinach"]
    used = _pantry_used(simple_recipe, pantry)
    unused = _pantry_unused(simple_recipe, pantry)
    assert sorted(used + unused) == sorted(pantry)


# ---------------------------------------------------------------------------
# Goal trace
# ---------------------------------------------------------------------------


def test_goal_trace_non_empty(simple_adapted: AdaptedRecipe, simple_filter_result: FilterResult, simple_profile: UserProfile) -> None:
    trace = _build_goal_trace(simple_adapted, simple_filter_result.decision_log, simple_profile)
    assert len(trace) > 0
    assert "## Goal Trace" in trace


def test_goal_trace_mentions_recipe_name(simple_adapted: AdaptedRecipe, simple_filter_result: FilterResult, simple_profile: UserProfile) -> None:
    trace = _build_goal_trace(simple_adapted, simple_filter_result.decision_log, simple_profile)
    assert simple_adapted.recipe.name in trace


def test_goal_trace_mentions_hard_constraints(simple_adapted: AdaptedRecipe, simple_filter_result: FilterResult, simple_profile: UserProfile) -> None:
    trace = _build_goal_trace(simple_adapted, simple_filter_result.decision_log, simple_profile)
    assert "gluten-free" in trace


def test_goal_trace_mentions_pantry_items(simple_adapted: AdaptedRecipe, simple_filter_result: FilterResult, simple_profile: UserProfile) -> None:
    trace = _build_goal_trace(simple_adapted, simple_filter_result.decision_log, simple_profile)
    # eggs, brown rice, garlic are all in pantry and in the recipe
    assert "eggs" in trace or "garlic" in trace or "brown rice" in trace


def test_goal_trace_shows_adaptations() -> None:
    recipe = Recipe(
        id="adapt-r",
        name="Adapted Dish",
        ingredients=[
            Ingredient(name="tofu", quantity=200.0, unit="g", category="protein"),
            Ingredient(name="garlic", quantity=2.0, unit="cloves", category="vegetable"),
        ],
        cuisine="Asian",
        dietary_tags=["vegan", "vegetarian", "gluten-free", "dairy-free", "nut-free"],
        prep_time_min=10,
        cook_time_min=15,
        difficulty=2,
        nutrition={"calories": 300.0, "protein_g": 18.0, "carbs_g": 10.0, "fat_g": 12.0},
        flavor_profile=["savory"],
        instructions=["Cook tofu.", "Add garlic."],
    )
    adapted = AdaptedRecipe(
        recipe=recipe,
        adaptations=[AdaptationEntry(original="chicken", replacement="tofu", reason="supports vegetarian goal")],
    )
    profile = UserProfile(
        user_id="u",
        soft_preferences=SoftPreferences(goals=["vegetarian"]),
        pantry=["garlic"],
    )
    engine = FilterEngine()
    result = engine.run([recipe], profile)
    trace = _build_goal_trace(adapted, result.decision_log, profile)
    assert "chicken" in trace
    assert "tofu" in trace


# ---------------------------------------------------------------------------
# Counterfactual
# ---------------------------------------------------------------------------


def test_counterfactual_with_hard_rejected(sample_recipes: list[Recipe], demo_profile: UserProfile) -> None:
    engine = FilterEngine()
    filter_result = engine.run(sample_recipes, demo_profile)
    # demo_user has gluten-free constraint; some recipes lack the tag
    adapted_recipe = AdaptedRecipe(recipe=filter_result.survivors[0].recipe, adaptations=[])
    cf = _build_counterfactual(adapted_recipe, filter_result.decision_log, filter_result, sample_recipes, demo_profile)
    assert "## Counterfactual" in cf
    assert len(cf) > 30


def test_counterfactual_graceful_when_none_rejected() -> None:
    """When no recipe is rejected by a hard rule, produce a graceful message."""
    recipe = Recipe(
        id="r1",
        name="Safe Dish",
        ingredients=[Ingredient(name="eggs", quantity=2.0, unit=None, category="protein")],
        cuisine="American",
        dietary_tags=["gluten-free", "nut-free", "dairy-free"],
        prep_time_min=5,
        cook_time_min=5,
        difficulty=1,
        nutrition={"calories": 150.0, "protein_g": 12.0, "carbs_g": 1.0, "fat_g": 10.0},
        flavor_profile=["savory"],
        instructions=["Fry eggs."],
    )
    profile = UserProfile(
        user_id="u",
        hard_constraints=[],
        soft_preferences=SoftPreferences(),
        pantry=["eggs"],
    )
    engine = FilterEngine()
    filter_result = engine.run([recipe], profile)
    adapted = AdaptedRecipe(recipe=recipe, adaptations=[])
    cf = _build_counterfactual(adapted, filter_result.decision_log, filter_result, [recipe], profile)
    assert "## Counterfactual" in cf
    # Should say all recipes passed
    assert "passed" in cf.lower() or "all" in cf.lower()


def test_counterfactual_names_hard_rule_reason(sample_recipes: list[Recipe], demo_profile: UserProfile) -> None:
    engine = FilterEngine()
    filter_result = engine.run(sample_recipes, demo_profile)
    adapted = AdaptedRecipe(recipe=filter_result.survivors[0].recipe, adaptations=[])
    cf = _build_counterfactual(adapted, filter_result.decision_log, filter_result, sample_recipes, demo_profile)
    # The counterfactual should name at least one rejected recipe
    rejected_ids = {
        e.recipe_id for e in filter_result.decision_log
        if not e.passed and e.rule_name in {"AllergyRule", "DietaryRule", "PantryCoverageRule"}
    }
    rejected_names = {r.name for r in sample_recipes if r.id in rejected_ids}
    assert any(name in cf for name in rejected_names)


# ---------------------------------------------------------------------------
# CBR trace
# ---------------------------------------------------------------------------


def test_cbr_trace_with_history(sample_recipes: list[Recipe]) -> None:
    """CBR trace when user has past ratings that match sample recipe IDs."""
    from useitup.matching import PantryCoverageRule, AllergyRule, DietaryRule

    profile = UserProfile(
        user_id="hist_user",
        hard_constraints=[],
        soft_preferences=SoftPreferences(goals=["high_protein"]),
        # Rich pantry to ensure some recipes pass the 50% coverage threshold
        pantry=["eggs", "garlic", "olive oil", "onion", "cumin", "paprika", "black beans", "avocado", "lime"],
        rating_history=[
            RatingEntry(recipe_id="sample-003", rating=5, timestamp="2026-01-10T18:00:00"),
            RatingEntry(recipe_id="sample-009", rating=4, timestamp="2026-01-15T18:00:00"),
        ],
    )
    engine = FilterEngine()
    filter_result = engine.run(sample_recipes, profile)
    assert filter_result.survivors, "Expected at least one survivor with rich pantry"
    survivors = [sr.recipe for sr in filter_result.survivors]
    retriever = CBRRetriever(sample_recipes, profile)
    matches = retriever.retrieve(survivors, k=5)
    adapted = AdaptedRecipe(recipe=matches[0].recipe, adaptations=[])

    trace = _build_cbr_trace(adapted, matches[0], profile)
    assert "## CBR Trace" in trace
    assert len(trace) > 50
    # Should mention past recipe in normal (non-cold-start) mode
    assert "Cold-start" not in trace


def test_cbr_trace_cold_start(sample_recipes: list[Recipe]) -> None:
    # Use a rich pantry so coverage threshold is met
    cold_start_profile = UserProfile(
        user_id="cold_user",
        hard_constraints=[],
        soft_preferences=SoftPreferences(preferred_cuisines=["Asian"], goals=[]),
        pantry=["eggs", "garlic", "olive oil", "onion", "cumin", "paprika", "avocado", "lime"],
    )
    engine = FilterEngine()
    filter_result = engine.run(sample_recipes, cold_start_profile)
    assert filter_result.survivors, "Expected survivors with rich pantry"
    survivors = [sr.recipe for sr in filter_result.survivors]
    retriever = CBRRetriever(sample_recipes, cold_start_profile)
    matches = retriever.retrieve(survivors, k=5)
    adapted = AdaptedRecipe(recipe=matches[0].recipe, adaptations=[])

    trace = _build_cbr_trace(adapted, matches[0], cold_start_profile)
    assert "## CBR Trace" in trace
    assert "Cold-start" in trace or "cold-start" in trace or "No rating history" in trace


def test_cbr_trace_empty_matches() -> None:
    profile = UserProfile(user_id="u", soft_preferences=SoftPreferences(), pantry=[])
    recipe = Recipe(
        id="r", name="X",
        ingredients=[Ingredient(name="egg", quantity=1.0, unit=None, category="protein")],
        cuisine="American", dietary_tags=[], prep_time_min=5, cook_time_min=5,
        difficulty=1, nutrition={}, flavor_profile=[], instructions=["Cook."],
    )
    adapted = AdaptedRecipe(recipe=recipe, adaptations=[])
    trace = _build_cbr_trace(adapted, None, profile)
    assert "## CBR Trace" in trace


# ---------------------------------------------------------------------------
# Ingredient utilization report
# ---------------------------------------------------------------------------


def test_utilization_used_section(simple_adapted: AdaptedRecipe, simple_filter_result: FilterResult, simple_profile: UserProfile) -> None:
    report = _build_ingredient_utilization_report(simple_adapted, simple_filter_result, simple_profile)
    assert "✅" in report
    assert "eggs" in report or "garlic" in report or "brown rice" in report


def test_utilization_unused_section(simple_adapted: AdaptedRecipe, simple_filter_result: FilterResult, simple_profile: UserProfile) -> None:
    report = _build_ingredient_utilization_report(simple_adapted, simple_filter_result, simple_profile)
    # spinach and olive oil are in pantry but not in simple_recipe
    assert "⚠️" in report
    assert "spinach" in report or "olive oil" in report


def test_utilization_missing_section(simple_adapted: AdaptedRecipe, simple_filter_result: FilterResult, simple_profile: UserProfile) -> None:
    report = _build_ingredient_utilization_report(simple_adapted, simple_filter_result, simple_profile)
    # soy sauce and sesame oil are in recipe but not in pantry
    assert "🛒" in report


def test_utilization_classifies_all_pantry_items(simple_recipe: Recipe, simple_filter_result: FilterResult, simple_profile: UserProfile) -> None:
    """Every pantry item must appear in exactly one of used/unused."""
    adapted = AdaptedRecipe(recipe=simple_recipe, adaptations=[])
    report = _build_ingredient_utilization_report(adapted, simple_filter_result, simple_profile)
    used = _pantry_used(simple_recipe, simple_profile.pantry)
    unused = _pantry_unused(simple_recipe, simple_profile.pantry)
    # Confirm no item is missing from both
    for item in simple_profile.pantry:
        assert item in used or item in unused, f"{item!r} not classified"
    # Confirm no overlap
    assert not set(used) & set(unused)


def test_utilization_followup_suggestions(sample_recipes: list[Recipe], demo_profile: UserProfile) -> None:
    engine = FilterEngine()
    filter_result = engine.run(sample_recipes, demo_profile)
    if not filter_result.survivors:
        pytest.skip("No survivors with demo profile")
    adapted = AdaptedRecipe(recipe=filter_result.survivors[0].recipe, adaptations=[])
    report = _build_ingredient_utilization_report(adapted, filter_result, demo_profile)
    assert "💡" in report


# ---------------------------------------------------------------------------
# Full generate_explanation
# ---------------------------------------------------------------------------


def test_generate_explanation_produces_all_fields(
    simple_adapted: AdaptedRecipe,
    simple_filter_result: FilterResult,
    simple_profile: UserProfile,
    simple_recipe: Recipe,
) -> None:
    breakdown = FeatureBreakdown(cuisine=0.0, protein=0.0, cooking_method=0.0, flavor=0.0, difficulty=0.0, prep_time=0.0)
    cbr_match = CBRMatch(
        recipe=simple_recipe,
        similarity_score=0.0,
        nearest_past_recipe=None,
        similarity_breakdown=breakdown,
        fallback_reason="No rating history; ranked by preferred cuisines and prep time",
    )
    expl = generate_explanation(
        adapted=simple_adapted,
        filter_result=simple_filter_result,
        cbr_match=cbr_match,
        profile=simple_profile,
        all_recipes=[simple_recipe],
    )
    assert isinstance(expl, Explanation)
    assert len(expl.goal_trace) > 0
    assert len(expl.counterfactual) > 0
    assert len(expl.cbr_trace) > 0
    assert len(expl.ingredient_utilization_report) > 0


def test_render_explanation_is_valid_markdown(
    simple_adapted: AdaptedRecipe,
    simple_filter_result: FilterResult,
    simple_profile: UserProfile,
    simple_recipe: Recipe,
) -> None:
    breakdown = FeatureBreakdown(cuisine=0.0, protein=0.0, cooking_method=0.0, flavor=0.0, difficulty=0.0, prep_time=0.0)
    cbr_match = CBRMatch(
        recipe=simple_recipe,
        similarity_score=0.0,
        nearest_past_recipe=None,
        similarity_breakdown=breakdown,
        fallback_reason="No rating history",
    )
    expl = generate_explanation(
        adapted=simple_adapted,
        filter_result=simple_filter_result,
        cbr_match=cbr_match,
        profile=simple_profile,
        all_recipes=[simple_recipe],
    )
    rendered = render_explanation(expl)
    assert "---" in rendered  # section separators
    assert "## Goal Trace" in rendered
    assert "## Counterfactual" in rendered
    assert "## CBR Trace" in rendered
    assert "## Ingredient Utilization Report" in rendered


# ---------------------------------------------------------------------------
# Snapshot test: full pipeline on demo profile + sample recipes
# ---------------------------------------------------------------------------


def test_snapshot_full_pipeline(sample_recipes: list[Recipe], demo_profile: UserProfile) -> None:
    """Regression guard: rendered explanation must match the saved fixture."""
    adapted, explanation = run_pipeline(sample_recipes, demo_profile)
    rendered = render_explanation(explanation)

    if not _FIXTURE_PATH.exists():
        _FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _FIXTURE_PATH.write_text(rendered, encoding="utf-8")
        pytest.skip("Snapshot created — re-run tests to verify")

    expected = _FIXTURE_PATH.read_text(encoding="utf-8")
    assert rendered == expected, (
        "Rendered explanation does not match the snapshot. "
        f"If this change is intentional, delete {_FIXTURE_PATH} and re-run to regenerate."
    )
