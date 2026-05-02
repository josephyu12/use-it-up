"""Tests for Stage 1: ingredient matching and rule-based filtering."""

from __future__ import annotations

import pytest

from useitup.matching import (
    COVERAGE_THRESHOLD,
    AllergyRule,
    CuisinePreferenceRule,
    DecisionEntry,
    DietaryRule,
    FilterEngine,
    GoalAlignmentRule,
    IngredientScore,
    IngredientScorer,
    PantryCoverageRule,
    PrepTimeRule,
    ScoredRecipe,
    _ingredient_match,
)
from useitup.profile import SoftPreferences, UserProfile
from useitup.schemas import Ingredient, Recipe


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_recipe(
    recipe_id: str = "r-test",
    name: str = "Test Recipe",
    ingredients: list[Ingredient] | None = None,
    cuisine: str = "American",
    dietary_tags: list[str] | None = None,
    prep_time_min: int = 20,
    cook_time_min: int = 20,
) -> Recipe:
    if ingredients is None:
        ingredients = [
            Ingredient(name="chicken breast", category="protein"),
            Ingredient(name="garlic", category="vegetable"),
        ]
    return Recipe(
        id=recipe_id,
        name=name,
        ingredients=ingredients,
        cuisine=cuisine,
        dietary_tags=dietary_tags or [],
        prep_time_min=prep_time_min,
        cook_time_min=cook_time_min,
        difficulty=1,
        nutrition={"calories": 300.0, "protein_g": 20.0, "carbs_g": 10.0, "fat_g": 8.0},
        flavor_profile=["savory"],
        instructions=["Cook and serve."],
    )


def _make_profile(
    hard_constraints: list[str] | None = None,
    pantry: list[str] | None = None,
    goals: list[str] | None = None,
    preferred_cuisines: list[str] | None = None,
    max_prep_time_min: int | None = None,
) -> UserProfile:
    return UserProfile(
        user_id="test",
        hard_constraints=hard_constraints or [],
        pantry=pantry or [],
        soft_preferences=SoftPreferences(
            goals=goals or [],
            preferred_cuisines=preferred_cuisines or [],
            max_prep_time_min=max_prep_time_min,
        ),
    )


# ---------------------------------------------------------------------------
# IngredientScorer
# ---------------------------------------------------------------------------

class TestIngredientScorer:
    scorer = IngredientScorer()

    def test_exact_match(self) -> None:
        recipe = _make_recipe(ingredients=[
            Ingredient(name="garlic", category="vegetable"),
            Ingredient(name="olive oil", category="fat"),
        ])
        score = self.scorer.score(recipe, pantry=["garlic", "olive oil"])
        assert score.overlap_count == 2
        assert score.missing_count == 0
        assert score.coverage == pytest.approx(1.0)
        assert score.missing_ingredients == []

    def test_fuzzy_match(self) -> None:
        recipe = _make_recipe(ingredients=[
            Ingredient(name="yellow onion", category="vegetable"),
            Ingredient(name="chicken breast", category="protein"),
        ])
        # "onion" fuzzy-matches "yellow onion"; "chicken" fuzzy-matches "chicken breast"
        score = self.scorer.score(recipe, pantry=["onion", "chicken"])
        assert score.overlap_count == 2
        assert score.coverage == pytest.approx(1.0)

    def test_no_overlap(self) -> None:
        recipe = _make_recipe(ingredients=[
            Ingredient(name="truffle", category="vegetable"),
            Ingredient(name="foie gras", category="protein"),
        ])
        score = self.scorer.score(recipe, pantry=["garlic", "onion"])
        assert score.overlap_count == 0
        assert score.coverage == pytest.approx(0.0)
        assert len(score.missing_ingredients) == 2

    def test_full_overlap(self) -> None:
        ings = [
            Ingredient(name="black beans", category="protein"),
            Ingredient(name="brown rice", category="grain"),
            Ingredient(name="cumin", category="spice"),
        ]
        recipe = _make_recipe(ingredients=ings)
        pantry = ["black beans", "brown rice", "cumin", "garlic", "onion"]
        score = self.scorer.score(recipe, pantry=pantry)
        assert score.coverage == pytest.approx(1.0)
        assert score.missing_count == 0

    def test_partial_overlap(self) -> None:
        ings = [
            Ingredient(name="garlic", category="vegetable"),
            Ingredient(name="saffron", category="spice"),
            Ingredient(name="lobster", category="protein"),
        ]
        recipe = _make_recipe(ingredients=ings)
        score = self.scorer.score(recipe, pantry=["garlic"])
        assert score.overlap_count == 1
        assert score.missing_count == 2
        assert score.coverage == pytest.approx(1 / 3)

    def test_empty_pantry(self) -> None:
        recipe = _make_recipe()
        score = self.scorer.score(recipe, pantry=[])
        assert score.coverage == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# AllergyRule
# ---------------------------------------------------------------------------

class TestAllergyRule:
    rule = AllergyRule()

    def test_passes_when_no_allergy_constraint(self) -> None:
        recipe = _make_recipe()  # no nut-free tag
        profile = _make_profile(hard_constraints=[])
        assert self.rule.applies(recipe, profile) is True

    def test_passes_when_recipe_is_nut_free(self) -> None:
        recipe = _make_recipe(dietary_tags=["nut-free"])
        profile = _make_profile(hard_constraints=["nut-free"])
        assert self.rule.applies(recipe, profile) is True

    def test_fails_when_recipe_missing_nut_free_tag(self) -> None:
        recipe = _make_recipe(dietary_tags=[])  # missing nut-free tag
        profile = _make_profile(hard_constraints=["nut-free"])
        assert self.rule.applies(recipe, profile) is False

    def test_fails_when_recipe_contains_nut_ingredient(self) -> None:
        recipe = _make_recipe(
            ingredients=[Ingredient(name="almond flour", category="grain")],
            dietary_tags=["nut-free"],  # tag present but ingredient contradicts it
        )
        profile = _make_profile(hard_constraints=["nut-free"])
        assert self.rule.applies(recipe, profile) is False

    def test_passes_when_recipe_uses_coconut_in_nut_free_recipe(self) -> None:
        recipe = _make_recipe(
            ingredients=[Ingredient(name="coconut milk", category="other")],
            dietary_tags=["nut-free"],
        )
        profile = _make_profile(hard_constraints=["nut-free"])
        assert self.rule.applies(recipe, profile) is True

    def test_fail_reason_mentions_recipe(self) -> None:
        recipe = _make_recipe(name="Walnut Cake", dietary_tags=[])
        profile = _make_profile(hard_constraints=["nut-free"])
        reason = self.rule.fail_reason(recipe, profile)
        assert "Walnut Cake" in reason


# ---------------------------------------------------------------------------
# DietaryRule
# ---------------------------------------------------------------------------

class TestDietaryRule:
    rule = DietaryRule()

    def test_passes_when_no_dietary_constraint(self) -> None:
        recipe = _make_recipe()
        profile = _make_profile(hard_constraints=[])
        assert self.rule.applies(recipe, profile) is True

    def test_passes_when_recipe_has_required_tag(self) -> None:
        recipe = _make_recipe(dietary_tags=["gluten-free", "nut-free"])
        profile = _make_profile(hard_constraints=["gluten-free"])
        assert self.rule.applies(recipe, profile) is True

    def test_fails_when_recipe_missing_gluten_free_tag(self) -> None:
        recipe = _make_recipe(dietary_tags=[])
        profile = _make_profile(hard_constraints=["gluten-free"])
        assert self.rule.applies(recipe, profile) is False

    def test_fails_when_only_one_of_two_constraints_met(self) -> None:
        recipe = _make_recipe(dietary_tags=["vegan"])
        profile = _make_profile(hard_constraints=["vegan", "gluten-free"])
        assert self.rule.applies(recipe, profile) is False

    def test_passes_multiple_constraints(self) -> None:
        recipe = _make_recipe(
            ingredients=[
                Ingredient(name="black beans", category="protein"),
                Ingredient(name="garlic", category="vegetable"),
            ],
            dietary_tags=["vegan", "gluten-free", "dairy-free"],
        )
        profile = _make_profile(hard_constraints=["vegan", "gluten-free"])
        assert self.rule.applies(recipe, profile) is True

    def test_rejects_chicken_recipe_mistagged_vegan(self) -> None:
        recipe = _make_recipe(
            name="Lemon Herb Chicken Orzo",
            ingredients=[
                Ingredient(name="chicken breast", category="protein"),
                Ingredient(name="orzo", category="grain"),
            ],
            dietary_tags=["vegan", "vegetarian", "gluten-free"],
        )
        profile = _make_profile(hard_constraints=["vegan"])
        assert self.rule.applies(recipe, profile) is False

    def test_fail_reason_lists_missing_tags(self) -> None:
        recipe = _make_recipe(dietary_tags=[])
        profile = _make_profile(hard_constraints=["gluten-free"])
        reason = self.rule.fail_reason(recipe, profile)
        assert "gluten-free" in reason


# ---------------------------------------------------------------------------
# PantryCoverageRule
# ---------------------------------------------------------------------------

class TestPantryCoverageRule:
    def test_passes_at_exactly_threshold(self) -> None:
        rule = PantryCoverageRule(threshold=0.5)
        recipe = _make_recipe(ingredients=[
            Ingredient(name="garlic", category="vegetable"),
            Ingredient(name="truffle", category="other"),
        ])
        profile = _make_profile(pantry=["garlic"])
        assert rule.applies(recipe, profile) is True  # coverage = 0.5

    def test_fails_below_threshold(self) -> None:
        rule = PantryCoverageRule(threshold=0.5)
        recipe = _make_recipe(ingredients=[
            Ingredient(name="garlic", category="vegetable"),
            Ingredient(name="truffle", category="other"),
            Ingredient(name="saffron", category="spice"),
        ])
        profile = _make_profile(pantry=["garlic"])
        assert rule.applies(recipe, profile) is False  # coverage ≈ 0.33

    def test_custom_threshold(self) -> None:
        rule = PantryCoverageRule(threshold=0.0)
        recipe = _make_recipe()
        profile = _make_profile(pantry=[])
        assert rule.applies(recipe, profile) is True  # any coverage passes


# ---------------------------------------------------------------------------
# PrepTimeRule
# ---------------------------------------------------------------------------

class TestPrepTimeRule:
    rule = PrepTimeRule()

    def test_passes_when_no_max_time(self) -> None:
        recipe = _make_recipe(prep_time_min=120)
        profile = _make_profile(max_prep_time_min=None)
        assert self.rule.applies(recipe, profile) is True

    def test_passes_within_limit(self) -> None:
        recipe = _make_recipe(prep_time_min=30)
        profile = _make_profile(max_prep_time_min=45)
        assert self.rule.applies(recipe, profile) is True

    def test_fails_over_limit(self) -> None:
        recipe = _make_recipe(prep_time_min=60)
        profile = _make_profile(max_prep_time_min=30)
        assert self.rule.applies(recipe, profile) is False

    def test_passes_at_exact_limit(self) -> None:
        recipe = _make_recipe(prep_time_min=45)
        profile = _make_profile(max_prep_time_min=45)
        assert self.rule.applies(recipe, profile) is True


# ---------------------------------------------------------------------------
# CuisinePreferenceRule
# ---------------------------------------------------------------------------

class TestCuisinePreferenceRule:
    rule = CuisinePreferenceRule()

    def test_passes_when_no_preferences(self) -> None:
        recipe = _make_recipe(cuisine="French")
        profile = _make_profile(preferred_cuisines=[])
        assert self.rule.applies(recipe, profile) is True

    def test_passes_matching_cuisine(self) -> None:
        recipe = _make_recipe(cuisine="Mexican")
        profile = _make_profile(preferred_cuisines=["Mexican", "Asian"])
        assert self.rule.applies(recipe, profile) is True

    def test_fails_non_preferred_cuisine(self) -> None:
        recipe = _make_recipe(cuisine="French")
        profile = _make_profile(preferred_cuisines=["Mexican", "Asian"])
        assert self.rule.applies(recipe, profile) is False


# ---------------------------------------------------------------------------
# GoalAlignmentRule
# ---------------------------------------------------------------------------

class TestGoalAlignmentRule:
    rule = GoalAlignmentRule()

    def test_passes_when_no_goals(self) -> None:
        recipe = _make_recipe(dietary_tags=[])
        profile = _make_profile(goals=[])
        assert self.rule.applies(recipe, profile) is True

    def test_passes_when_recipe_has_aligned_tag(self) -> None:
        recipe = _make_recipe(dietary_tags=["high-protein"])
        profile = _make_profile(goals=["high_protein"])
        assert self.rule.applies(recipe, profile) is True

    def test_fails_when_no_goal_aligned(self) -> None:
        recipe = _make_recipe(dietary_tags=["vegan"])
        profile = _make_profile(goals=["high_protein", "low_cost"])
        assert self.rule.applies(recipe, profile) is False

    def test_passes_any_goal_match(self) -> None:
        recipe = _make_recipe(dietary_tags=["low-cost"])
        profile = _make_profile(goals=["high_protein", "low_cost"])
        assert self.rule.applies(recipe, profile) is True


# ---------------------------------------------------------------------------
# FilterEngine — end-to-end with 50 crafted recipes
# ---------------------------------------------------------------------------

def _nut_allergen_recipe(recipe_id: str) -> Recipe:
    """Recipe that contains nuts and lacks nut-free tag — fails AllergyRule."""
    return _make_recipe(
        recipe_id=recipe_id,
        name=f"Nut Dish {recipe_id}",
        ingredients=[
            Ingredient(name="chicken breast", category="protein"),
            Ingredient(name="garlic", category="vegetable"),
            Ingredient(name="walnut", category="other"),
        ],
        dietary_tags=[],  # no nut-free tag
    )


def _safe_recipe(recipe_id: str, cuisine: str = "American") -> Recipe:
    """Recipe without allergens and with high pantry coverage."""
    return _make_recipe(
        recipe_id=recipe_id,
        name=f"Safe Dish {recipe_id}",
        ingredients=[
            Ingredient(name="chicken breast", category="protein"),
            Ingredient(name="garlic", category="vegetable"),
            Ingredient(name="olive oil", category="fat"),
        ],
        dietary_tags=["nut-free", "gluten-free", "dairy-free"],
        cuisine=cuisine,
    )


def _build_50_recipes() -> list[Recipe]:
    recipes = []
    for i in range(25):
        recipes.append(_nut_allergen_recipe(f"allergen-{i:03d}"))
    for i in range(25):
        recipes.append(_safe_recipe(f"safe-{i:03d}"))
    return recipes


class TestFilterEngineEndToEnd:
    profile = _make_profile(
        hard_constraints=["nut-free"],
        # pantry covers chicken breast, garlic, olive oil exactly
        pantry=["chicken breast", "garlic", "olive oil", "onion", "cumin"],
        goals=["high_protein"],
        preferred_cuisines=["American"],
        max_prep_time_min=60,
    )

    def test_no_allergen_recipes_survive(self) -> None:
        engine = FilterEngine()
        result = engine.run(_build_50_recipes(), self.profile)
        survivor_ids = {sr.recipe.id for sr in result.survivors}
        for i in range(25):
            assert f"allergen-{i:03d}" not in survivor_ids, (
                f"Allergen recipe allergen-{i:03d} should have been eliminated"
            )

    def test_safe_recipes_can_survive(self) -> None:
        engine = FilterEngine()
        result = engine.run(_build_50_recipes(), self.profile)
        assert len(result.survivors) > 0

    def test_decision_log_records_every_recipe(self) -> None:
        engine = FilterEngine()
        result = engine.run(_build_50_recipes(), self.profile)
        logged_ids = {entry.recipe_id for entry in result.decision_log}
        for recipe in _build_50_recipes():
            assert recipe.id in logged_ids, f"{recipe.id} missing from decision log"

    def test_every_elimination_has_reason(self) -> None:
        engine = FilterEngine()
        result = engine.run(_build_50_recipes(), self.profile)
        for entry in result.decision_log:
            if not entry.passed:
                assert entry.reason, f"Empty reason for failed rule on {entry.recipe_id}"

    def test_survivors_have_ingredient_scores(self) -> None:
        engine = FilterEngine()
        result = engine.run(_build_50_recipes(), self.profile)
        for sr in result.survivors:
            assert isinstance(sr.ingredient_score, IngredientScore)
            assert 0.0 <= sr.ingredient_score.coverage <= 1.0

    def test_survivors_have_soft_scores_in_range(self) -> None:
        engine = FilterEngine()
        result = engine.run(_build_50_recipes(), self.profile)
        for sr in result.survivors:
            assert 0.0 <= sr.soft_score <= 1.0

    def test_decision_log_failure_entries_have_rule_name(self) -> None:
        engine = FilterEngine()
        result = engine.run(_build_50_recipes(), self.profile)
        failures = [e for e in result.decision_log if not e.passed]
        assert len(failures) >= 25  # at least 25 allergen recipes eliminated
        for entry in failures:
            assert entry.rule_name  # non-empty rule name


# ---------------------------------------------------------------------------
# FilterEngine — custom rule injection
# ---------------------------------------------------------------------------

class TestIngredientMatchSemantics:
    """Regression tests for the asymmetric ingredient matcher.

    The matcher must distinguish between a pantry item that's more specific
    than the recipe ingredient (safe: user has the variant) and a pantry item
    that's more generic (often unsafe: "cheese" does not mean "feta cheese").
    """

    @pytest.mark.parametrize("pantry,recipe_ing", [
        ("cheese", "feta cheese"),
        ("cheese", "goat cheese"),
        ("cheese", "parmesan cheese"),
        ("cheese", "cheese pizza"),
        ("cheese", "cheese tortellini"),
        ("milk", "almond milk"),
        ("milk", "coconut milk"),
        ("butter", "peanut butter"),
        ("butter", "apple butter"),
        ("flour", "almond flour"),
        ("flour", "rice flour"),
        ("oil", "sesame oil"),
        ("pepper", "bell pepper"),
        ("pepper", "jalapeno pepper"),
        ("beans", "green beans"),
        ("sugar", "sugar snap peas"),
        ("cream", "sour cream"),
        ("cream", "ice cream"),
    ])
    def test_generic_pantry_does_not_match_specific_recipe_variant(
        self, pantry: str, recipe_ing: str
    ) -> None:
        assert _ingredient_match(pantry, recipe_ing) is False

    @pytest.mark.parametrize("pantry,recipe_ing", [
        ("olive oil", "sesame oil"),
        ("olive oil", "vegetable oil"),
        ("cheddar cheese", "feta cheese"),
        ("black beans", "green beans"),
    ])
    def test_cross_variant_ingredients_do_not_match(
        self, pantry: str, recipe_ing: str
    ) -> None:
        """Two specific variants with a shared head noun must not match."""
        assert _ingredient_match(pantry, recipe_ing) is False

    @pytest.mark.parametrize("pantry,recipe_ing", [
        ("peanut butter", "butter"),
        ("almond milk", "milk"),
        ("sour cream", "cream"),
        ("bell pepper", "pepper"),
        ("green beans", "beans"),
        ("almond flour", "flour"),
    ])
    def test_compound_noun_not_head_substitute(
        self, pantry: str, recipe_ing: str
    ) -> None:
        """A compound-noun pantry item is not a substitute for its head noun."""
        assert _ingredient_match(pantry, recipe_ing) is False

    @pytest.mark.parametrize("pantry,recipe_ing", [
        ("chicken", "chicken breast"),
        ("chicken", "chicken thigh"),
        ("onion", "yellow onion"),
        ("onion", "white onion"),
        ("salt", "kosher salt"),
        ("tomato", "roma tomato"),
    ])
    def test_generic_pantry_matches_allowlisted_specific_variant(
        self, pantry: str, recipe_ing: str
    ) -> None:
        """Whitelisted generic→specific pairs still match (e.g. chicken→chicken breast)."""
        assert _ingredient_match(pantry, recipe_ing) is True

    @pytest.mark.parametrize("pantry,recipe_ing", [
        ("parsley", "fresh parsley"),
        ("cilantro", "chopped cilantro"),
        ("ginger", "ground ginger"),
        ("onion", "chopped onion"),
    ])
    def test_prep_modifier_extras_do_not_block_match(
        self, pantry: str, recipe_ing: str
    ) -> None:
        """Recipe-side prep modifiers like 'fresh' or 'ground' don't block a match."""
        assert _ingredient_match(pantry, recipe_ing) is True

    @pytest.mark.parametrize("pantry,recipe_ing", [
        ("yellow onion", "onion"),
        ("chicken breast", "chicken"),
        ("heavy cream", "cream"),
        ("extra virgin olive oil", "olive oil"),
    ])
    def test_specific_pantry_matches_generic_recipe(
        self, pantry: str, recipe_ing: str
    ) -> None:
        """Pantry-more-specific-than-recipe direction is always a valid match."""
        assert _ingredient_match(pantry, recipe_ing) is True


class TestPantryCoverageNotInflated:
    """Integration: coverage must not be inflated by bogus ingredient matches."""

    def test_generic_cheese_does_not_inflate_coverage(self) -> None:
        scorer = IngredientScorer()
        recipe = _make_recipe(ingredients=[
            Ingredient(name="feta cheese", category="dairy"),
            Ingredient(name="olives", category="vegetable"),
            Ingredient(name="cucumber", category="vegetable"),
        ])
        score = scorer.score(recipe, pantry=["cheese"])
        assert score.overlap_count == 0
        assert "feta cheese" not in score.matched_ingredients

    def test_generic_oil_does_not_inflate_coverage(self) -> None:
        scorer = IngredientScorer()
        recipe = _make_recipe(ingredients=[
            Ingredient(name="sesame oil", category="fat"),
            Ingredient(name="soy sauce", category="condiment"),
        ])
        score = scorer.score(recipe, pantry=["olive oil"])
        # olive oil in pantry must NOT match sesame oil in recipe
        assert "sesame oil" not in score.matched_ingredients


class TestFilterEngineCustomRules:
    def test_empty_hard_rules_keeps_all_recipes(self) -> None:
        engine = FilterEngine(hard_rules=[], soft_rules=[])
        recipes = [_safe_recipe("s1"), _nut_allergen_recipe("a1")]
        profile = _make_profile(hard_constraints=["nut-free"])
        result = engine.run(recipes, profile)
        assert len(result.survivors) == 2

    def test_coverage_threshold_zero_accepts_empty_pantry(self) -> None:
        engine = FilterEngine(
            hard_rules=[PantryCoverageRule(threshold=0.0)],
            soft_rules=[],
        )
        recipes = [_safe_recipe("s1")]
        profile = _make_profile(pantry=[])
        result = engine.run(recipes, profile)
        assert len(result.survivors) == 1
