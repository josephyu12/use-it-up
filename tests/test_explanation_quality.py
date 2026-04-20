"""Rule-based quality checks for generated explanations (no LLM judge required)."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from useitup.data_loader import load_recipes
from useitup.explain import (
    _pantry_unused,
    _pantry_used,
    render_explanation,
)
from useitup.matching import IngredientScorer
from useitup.pipeline import recommend
from useitup.profile import RatingEntry, SoftPreferences, UserProfile
from useitup.schemas import Ingredient, Recipe

_DATA_DIR = Path(__file__).parent.parent / "data"

_TEMPLATE_SLOT_RE = re.compile(r"\{[a-z_]+\}")


@pytest.fixture
def sample_recipes() -> list[Recipe]:
    return load_recipes(_DATA_DIR / "recipes_sample.json")


@pytest.fixture
def base_profile() -> UserProfile:
    return UserProfile(
        user_id="quality_test_user",
        hard_constraints=["gluten-free"],
        soft_preferences=SoftPreferences(
            preferred_cuisines=["Mexican"],
            goals=["low_cost"],
        ),
        pantry=[
            "black beans", "corn tortillas", "avocado", "salsa",
            "cumin", "lime", "cilantro", "garlic",
        ],
        rating_history=[
            RatingEntry(recipe_id="sample-003", rating=5, timestamp="2026-01-10T18:00:00"),
        ],
    )


@pytest.fixture
def rendered(base_profile: UserProfile, sample_recipes: list[Recipe]) -> str:
    results = recommend(base_profile, sample_recipes, top_k=1)
    return render_explanation(results[0].explanation)


# ── Quality check 1: no unresolved template slots ───────────────────────────


def test_no_template_slots_in_explanation(rendered: str) -> None:
    """Explanations must not contain unresolved {placeholder} markers."""
    matches = _TEMPLATE_SLOT_RE.findall(rendered)
    assert not matches, f"Unresolved template slots found: {matches}"


@pytest.mark.parametrize("hard_constraints,goals,pantry", [
    (
        [],
        ["quick"],
        ["eggs", "butter", "chives", "salt", "pepper"],
    ),
    (
        ["gluten-free"],
        [],
        ["cucumber", "tomato", "olive oil", "oregano", "feta cheese",
         "kalamata olives", "lemon juice", "red onion"],
    ),
    (
        ["vegan"],
        ["vegan"],
        ["black beans", "corn tortillas", "avocado", "lime", "cumin", "cilantro", "salsa"],
    ),
])
def test_no_template_slots_across_profiles(
    hard_constraints: list[str],
    goals: list[str],
    pantry: list[str],
    sample_recipes: list[Recipe],
) -> None:
    profile = UserProfile(
        user_id="qual_parameterized",
        hard_constraints=hard_constraints,
        soft_preferences=SoftPreferences(goals=goals),
        pantry=pantry,
    )
    try:
        results = recommend(profile, sample_recipes, top_k=1)
    except ValueError:
        pytest.skip("No survivors for this profile — slot check not applicable")
    rendered = render_explanation(results[0].explanation)
    matches = _TEMPLATE_SLOT_RE.findall(rendered)
    assert not matches, f"Unresolved template slots for profile {hard_constraints}/{goals}: {matches}"


# ── Quality check 2: explanation references a pantry ingredient ──────────────


def test_explanation_mentions_at_least_one_pantry_ingredient(
    base_profile: UserProfile, rendered: str
) -> None:
    rendered_lower = rendered.lower()
    found = any(item.lower() in rendered_lower for item in base_profile.pantry)
    assert found, (
        f"No pantry item found in explanation.\n"
        f"Pantry: {base_profile.pantry}\n"
        f"Explanation snippet:\n{rendered[:400]}"
    )


def test_utilization_report_lists_used_pantry_items(
    base_profile: UserProfile, sample_recipes: list[Recipe]
) -> None:
    results = recommend(base_profile, sample_recipes, top_k=1)
    adapted = results[0].adapted_recipe
    report = results[0].explanation.ingredient_utilization_report
    used = _pantry_used(adapted.recipe, base_profile.pantry)
    if used:
        assert any(item.lower() in report.lower() for item in used), (
            f"Utilization report should name used items {used}.\nReport:\n{report}"
        )


# ── Quality check 3: counterfactual names a specific rejected recipe ─────────


def test_counterfactual_names_rejected_recipe_when_rejections_exist(
    base_profile: UserProfile, sample_recipes: list[Recipe]
) -> None:
    results = recommend(base_profile, sample_recipes, top_k=1)
    decision_log = results[0].decision_log
    counterfactual = results[0].explanation.counterfactual

    hard_rule_names = frozenset({"AllergyRule", "DietaryRule", "PantryCoverageRule"})
    rejected_ids = {
        e.recipe_id for e in decision_log
        if not e.passed and e.rule_name in hard_rule_names
    }
    if not rejected_ids:
        pytest.skip("No hard rejections — counterfactual naming check not applicable")

    rejected_names = {r.name for r in sample_recipes if r.id in rejected_ids}
    assert any(name in counterfactual for name in rejected_names), (
        f"Counterfactual must name at least one of the {len(rejected_names)} rejected recipes.\n"
        f"Rejected: {rejected_names}\nCounterfactual:\n{counterfactual}"
    )


def test_counterfactual_has_substantive_content(rendered: str) -> None:
    """Counterfactual section must contain more than just its header."""
    assert "## Counterfactual" in rendered
    cf_start = rendered.index("## Counterfactual")
    cf_section = rendered[cf_start:]
    # Strip the header itself; remaining content must be non-trivial.
    body = cf_section.replace("## Counterfactual", "").strip()
    assert len(body) > 30, f"Counterfactual body is too short: {body!r}"


# ── Quality check 4: ingredient utilization sums correctly ───────────────────


def test_pantry_used_plus_unused_equals_full_pantry() -> None:
    """used + unused must cover every pantry item exactly once."""
    recipe = Recipe(
        id="q-r1",
        name="Test Dish",
        ingredients=[
            Ingredient(name="eggs", quantity=2.0, unit=None, category="protein"),
            Ingredient(name="garlic", quantity=2.0, unit="cloves", category="vegetable"),
            Ingredient(name="soy sauce", quantity=1.0, unit="tbsp", category="condiment"),
        ],
        cuisine="Asian",
        dietary_tags=["dairy-free", "nut-free"],
        prep_time_min=10,
        cook_time_min=10,
        difficulty=1,
        nutrition={},
        flavor_profile=["savory"],
        instructions=["Cook."],
    )
    pantry = ["eggs", "garlic", "spinach", "olive oil"]
    used = _pantry_used(recipe, pantry)
    unused = _pantry_unused(recipe, pantry)

    assert sorted(used + unused) == sorted(pantry), (
        f"used({used}) + unused({unused}) must equal pantry({pantry})"
    )
    assert not (set(used) & set(unused)), "used and unused must be disjoint"


def test_coverage_score_accounts_for_all_ingredients() -> None:
    """overlap_count + missing_count must equal total ingredient count."""
    recipe = Recipe(
        id="q-r2",
        name="Quality Check Recipe",
        ingredients=[
            Ingredient(name="chicken breast", quantity=200.0, unit="g", category="protein"),
            Ingredient(name="garlic", quantity=2.0, unit="cloves", category="vegetable"),
            Ingredient(name="olive oil", quantity=1.0, unit="tbsp", category="fat"),
            Ingredient(name="lemon juice", quantity=1.0, unit="tbsp", category="condiment"),
        ],
        cuisine="Mediterranean",
        dietary_tags=["gluten-free", "nut-free", "dairy-free"],
        prep_time_min=10,
        cook_time_min=20,
        difficulty=2,
        nutrition={"calories": 300.0},
        flavor_profile=["savory"],
        instructions=["Cook chicken.", "Finish with garlic, olive oil, lemon."],
    )
    pantry = ["garlic", "olive oil"]
    scorer = IngredientScorer()
    score = scorer.score(recipe, pantry)

    assert score.overlap_count + score.missing_count == len(recipe.ingredients)
    assert abs(score.coverage - score.overlap_count / len(recipe.ingredients)) < 1e-9


@pytest.mark.parametrize("pantry,expected_used,expected_missing", [
    (["eggs", "garlic", "soy sauce"], 3, 0),
    (["eggs"], 1, 2),
    (["spinach", "butter"], 0, 3),
])
def test_utilization_counts_match_expected(
    pantry: list[str],
    expected_used: int,
    expected_missing: int,
) -> None:
    recipe = Recipe(
        id="q-r3",
        name="Parametric Dish",
        ingredients=[
            Ingredient(name="eggs", quantity=2.0, unit=None, category="protein"),
            Ingredient(name="garlic", quantity=1.0, unit="cloves", category="vegetable"),
            Ingredient(name="soy sauce", quantity=1.0, unit="tbsp", category="condiment"),
        ],
        cuisine="Asian",
        dietary_tags=[],
        prep_time_min=5,
        cook_time_min=5,
        difficulty=1,
        nutrition={},
        flavor_profile=[],
        instructions=["Cook."],
    )
    scorer = IngredientScorer()
    score = scorer.score(recipe, pantry)
    used = _pantry_used(recipe, pantry)

    assert len(used) == expected_used, f"Expected {expected_used} used items, got {len(used)}: {used}"
    assert score.missing_count == expected_missing
    assert score.overlap_count + score.missing_count == len(recipe.ingredients)
