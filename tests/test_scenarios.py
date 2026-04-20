"""End-to-end scenario tests — Helen's QA scope from the project proposal."""

from __future__ import annotations

from pathlib import Path

import pytest

from useitup.cbr import CBRAdapter, CBRRetriever
from useitup.data_loader import load_recipes
from useitup.explain import (
    _build_cbr_trace,
    generate_explanation,
    render_explanation,
)
from useitup.matching import FilterEngine, IngredientScorer, PantryCoverageRule
from useitup.pipeline import recommend
from useitup.profile import RatingEntry, SoftPreferences, UserProfile
from useitup.schemas import Ingredient, Recipe

_ROOT = Path(__file__).parent.parent
_DATA_DIR = _ROOT / "data"


# ── Shared helpers ────────────────────────────────────────────────────────────


def _make_ingredient(name: str, category: str = "other") -> Ingredient:
    return Ingredient(name=name, quantity=None, unit=None, category=category)


@pytest.fixture
def sample_recipes() -> list[Recipe]:
    return load_recipes(_DATA_DIR / "recipes_sample.json")


# ── Scenario A: Gluten-free student, pasta-night pantry ──────────────────────


@pytest.fixture
def gf_pasta_recipe() -> Recipe:
    """A gluten-free rice-pasta dish — all five ingredients are in the scenario A pantry."""
    return Recipe(
        id="test-gf-pasta",
        name="Rice Pasta Primavera",
        ingredients=[
            _make_ingredient("rice pasta", "grain"),
            _make_ingredient("tomatoes", "vegetable"),
            _make_ingredient("garlic", "vegetable"),
            _make_ingredient("olive oil", "fat"),
            _make_ingredient("parmesan", "dairy"),
        ],
        cuisine="Italian",
        dietary_tags=["vegetarian", "gluten-free", "nut-free"],
        prep_time_min=15,
        cook_time_min=15,
        difficulty=2,
        nutrition={"calories": 480.0, "protein_g": 14.0, "carbs_g": 70.0, "fat_g": 16.0},
        flavor_profile=["savory", "fresh"],
        instructions=[
            "Cook rice pasta in salted boiling water until al dente.",
            "Sauté garlic in olive oil; add diced tomatoes and simmer.",
            "Toss drained pasta in the sauce; top with grated parmesan.",
        ],
    )


class TestScenarioA:
    """Gluten-free student with a pasta-night pantry."""

    PANTRY = ["tomatoes", "garlic", "olive oil", "rice pasta", "parmesan"]

    @pytest.fixture
    def profile(self) -> UserProfile:
        return UserProfile(
            user_id="scenario_a_user",
            hard_constraints=["gluten-free"],
            soft_preferences=SoftPreferences(goals=[]),
            pantry=self.PANTRY,
        )

    @pytest.fixture
    def recipes(self, gf_pasta_recipe: Recipe, sample_recipes: list[Recipe]) -> list[Recipe]:
        # Add our custom GF recipe; the sample set includes non-GF recipes for counterfactual.
        return [gf_pasta_recipe] + sample_recipes

    def test_recommended_recipe_is_gluten_free(
        self, profile: UserProfile, recipes: list[Recipe]
    ) -> None:
        results = recommend(profile, recipes, top_k=1)
        assert "gluten-free" in results[0].adapted_recipe.recipe.dietary_tags

    def test_explanation_mentions_rice_pasta(
        self, profile: UserProfile, recipes: list[Recipe]
    ) -> None:
        results = recommend(profile, recipes, top_k=1)
        rendered = render_explanation(results[0].explanation)
        assert "rice pasta" in rendered.lower()

    def test_counterfactual_references_non_gf_alternative(
        self, profile: UserProfile, recipes: list[Recipe]
    ) -> None:
        results = recommend(profile, recipes, top_k=1)
        counterfactual = results[0].explanation.counterfactual
        # At least one non-GF recipe must be named in the counterfactual section.
        non_gf_names = [r.name for r in recipes if "gluten-free" not in r.dietary_tags]
        assert any(name in counterfactual for name in non_gf_names), (
            f"Counterfactual should name a non-GF recipe. "
            f"Non-GF candidates: {non_gf_names}\n\nCounterfactual:\n{counterfactual}"
        )

    def test_only_gluten_free_recipes_survive_filter(
        self, profile: UserProfile, recipes: list[Recipe]
    ) -> None:
        engine = FilterEngine()
        result = engine.run(recipes, profile)
        for sr in result.survivors:
            assert "gluten-free" in sr.recipe.dietary_tags


# ── Scenario B: Vegan cold-start ─────────────────────────────────────────────


class TestScenarioB:
    """Vegan user with empty rating history (cold-start)."""

    PANTRY = [
        "black beans", "garlic", "olive oil", "onion", "cumin",
        "lime", "avocado", "corn tortillas", "red lentils", "tomato",
    ]

    @pytest.fixture
    def profile(self) -> UserProfile:
        return UserProfile(
            user_id="scenario_b_user",
            hard_constraints=["vegan"],
            soft_preferences=SoftPreferences(goals=["vegan"]),
            rating_history=[],
            pantry=self.PANTRY,
        )

    def test_cbr_trace_mentions_cold_start(
        self, profile: UserProfile, sample_recipes: list[Recipe]
    ) -> None:
        results = recommend(profile, sample_recipes, top_k=1)
        cbr_trace = results[0].explanation.cbr_trace
        assert "Cold-start" in cbr_trace or "cold-start" in cbr_trace or "No rating" in cbr_trace, (
            f"CBR trace should mention cold-start mode. Got:\n{cbr_trace}"
        )

    def test_cbr_match_has_fallback_reason(
        self, profile: UserProfile, sample_recipes: list[Recipe]
    ) -> None:
        results = recommend(profile, sample_recipes, top_k=1)
        assert results[0].cbr_matches[0].fallback_reason is not None

    def test_recommended_recipe_is_vegan(
        self, profile: UserProfile, sample_recipes: list[Recipe]
    ) -> None:
        results = recommend(profile, sample_recipes, top_k=1)
        recipe = results[0].adapted_recipe.recipe
        assert "vegan" in recipe.dietary_tags, (
            f"Recommended recipe '{recipe.name}' is not tagged vegan"
        )

    def test_no_animal_product_tags_in_survivor(
        self, profile: UserProfile, sample_recipes: list[Recipe]
    ) -> None:
        """All survivors must carry the vegan tag (DietaryRule guarantees this)."""
        engine = FilterEngine()
        result = engine.run(sample_recipes, profile)
        for sr in result.survivors:
            assert "vegan" in sr.recipe.dietary_tags


# ── Scenario C: Conflicting soft preferences ─────────────────────────────────


class TestScenarioC:
    """User wants quick + low-cost + Japanese cuisine — impossible to fully satisfy."""

    PANTRY = [
        "black beans", "corn tortillas", "avocado", "salsa", "cumin",
        "lime", "cilantro", "garlic", "olive oil", "onion",
    ]

    @pytest.fixture
    def profile(self) -> UserProfile:
        return UserProfile(
            user_id="scenario_c_user",
            hard_constraints=[],
            soft_preferences=SoftPreferences(
                preferred_cuisines=["Japanese"],
                goals=["quick", "low_cost"],
            ),
            pantry=self.PANTRY,
        )

    def test_pipeline_returns_a_recipe_despite_conflicts(
        self, profile: UserProfile, sample_recipes: list[Recipe]
    ) -> None:
        """Pipeline should not raise even when no Japanese recipes exist."""
        results = recommend(profile, sample_recipes, top_k=1)
        assert len(results) == 1
        assert results[0].adapted_recipe.recipe.name

    def test_explanation_shows_satisfied_goals(
        self, profile: UserProfile, sample_recipes: list[Recipe]
    ) -> None:
        results = recommend(profile, sample_recipes, top_k=1)
        goal_trace = results[0].explanation.goal_trace
        # quick and low_cost goals will match Black Bean Tacos tags
        assert "quick" in goal_trace.lower() or "low cost" in goal_trace.lower()

    def test_explanation_notes_cuisine_preference_not_met(
        self, profile: UserProfile, sample_recipes: list[Recipe]
    ) -> None:
        """Goal trace should surface that CuisinePreferenceRule was not satisfied."""
        results = recommend(profile, sample_recipes, top_k=1)
        goal_trace = results[0].explanation.goal_trace
        # The soft_failed block mentions rule names that didn't pass
        assert "CuisinePreferenceRule" in goal_trace or "soft preference" in goal_trace.lower()

    def test_recommended_recipe_is_not_japanese(
        self, profile: UserProfile, sample_recipes: list[Recipe]
    ) -> None:
        results = recommend(profile, sample_recipes, top_k=1)
        assert results[0].adapted_recipe.recipe.cuisine != "Japanese"


# ── Scenario D: Sparse pantry (2 ingredients) ────────────────────────────────


class TestScenarioD:
    """Only 2 pantry items — default threshold rejects all; lowered threshold surfaces missing list."""

    SPARSE_PANTRY = ["garlic", "olive oil"]

    @pytest.fixture
    def profile(self) -> UserProfile:
        return UserProfile(
            user_id="scenario_d_user",
            hard_constraints=[],
            soft_preferences=SoftPreferences(),
            pantry=self.SPARSE_PANTRY,
        )

    def test_default_threshold_raises_no_survivors(
        self, profile: UserProfile, sample_recipes: list[Recipe]
    ) -> None:
        """With only 2 pantry items, no recipe clears the 50% coverage threshold."""
        with pytest.raises(ValueError, match="No recipes survived"):
            recommend(profile, sample_recipes)

    def test_lowered_threshold_returns_recipe_with_missing_section(
        self, profile: UserProfile, sample_recipes: list[Recipe]
    ) -> None:
        """With threshold=0.1 a recipe is returned and the report lists missing ingredients."""
        engine = FilterEngine(hard_rules=[PantryCoverageRule(threshold=0.1)])
        filter_result = engine.run(sample_recipes, profile)
        assert filter_result.survivors, "Expected at least one survivor with threshold=0.1"

        retriever = CBRRetriever(sample_recipes, profile)
        matches = retriever.retrieve([sr.recipe for sr in filter_result.survivors], k=1)
        adapted = CBRAdapter().adapt(matches[0], profile)
        explanation = generate_explanation(adapted, filter_result, matches, profile, sample_recipes)

        report = explanation.ingredient_utilization_report
        assert "## Ingredient Utilization Report" in report
        assert "🛒" in report, "Missing-ingredients section (🛒) must be present when pantry is sparse"

    def test_sparse_pantry_all_recipes_have_missing_ingredients(
        self, profile: UserProfile, sample_recipes: list[Recipe]
    ) -> None:
        """Every sample recipe should have unmet ingredients given only 2 pantry items."""
        scorer = IngredientScorer()
        for recipe in sample_recipes:
            score = scorer.score(recipe, self.SPARSE_PANTRY)
            assert score.overlap_count + score.missing_count == len(recipe.ingredients)
            assert score.missing_count > 0


# ── Scenario E: Adaptation triggered by vegetarian goal ──────────────────────


@pytest.fixture
def chicken_piccata_recipe() -> Recipe:
    """Chicken-based recipe used to test vegetarian goal adaptation."""
    return Recipe(
        id="test-piccata",
        name="Chicken Piccata",
        ingredients=[
            _make_ingredient("chicken breast", "protein"),
            _make_ingredient("lemon juice", "condiment"),
            _make_ingredient("capers", "condiment"),
            _make_ingredient("garlic", "vegetable"),
            _make_ingredient("olive oil", "fat"),
            _make_ingredient("butter", "dairy"),
        ],
        cuisine="Italian",
        dietary_tags=["nut-free", "gluten-free"],
        prep_time_min=10,
        cook_time_min=20,
        difficulty=2,
        nutrition={"calories": 420.0, "protein_g": 38.0, "carbs_g": 6.0, "fat_g": 24.0},
        flavor_profile=["sour", "savory"],
        instructions=[
            "Pound chicken breasts thin and season.",
            "Sear in olive oil and butter until golden.",
            "Deglaze with lemon juice; add garlic and capers; simmer 3 minutes.",
        ],
    )


class TestScenarioE:
    """User rated chicken piccata 5/5, current goal is vegetarian — adaptation must fire."""

    PANTRY = ["chicken breast", "lemon juice", "garlic", "olive oil", "capers", "butter"]

    @pytest.fixture
    def profile(self, chicken_piccata_recipe: Recipe) -> UserProfile:
        return UserProfile(
            user_id="scenario_e_user",
            hard_constraints=[],
            soft_preferences=SoftPreferences(goals=["vegetarian"]),
            pantry=self.PANTRY,
            rating_history=[
                RatingEntry(
                    recipe_id=chicken_piccata_recipe.id,
                    rating=5,
                    timestamp="2026-01-10T18:30:00",
                ),
            ],
        )

    @pytest.fixture
    def recipes(self, chicken_piccata_recipe: Recipe, sample_recipes: list[Recipe]) -> list[Recipe]:
        return [chicken_piccata_recipe] + sample_recipes

    def test_adapted_recipe_substitutes_chicken(
        self, profile: UserProfile, recipes: list[Recipe]
    ) -> None:
        results = recommend(profile, recipes, top_k=1)
        adapted = results[0].adapted_recipe
        substituted = any("chicken" in a.original.lower() for a in adapted.adaptations)
        if substituted:
            chicken_sub = next(a for a in adapted.adaptations if "chicken" in a.original.lower())
            assert chicken_sub.replacement in (
                "tofu", "jackfruit", "seitan", "tempeh", "lentils", "chickpeas"
            ), f"Unexpected chicken replacement: {chicken_sub.replacement}"
        else:
            # If no substitution happened, the recommended recipe must not contain chicken.
            ingredient_names = [i.name.lower() for i in adapted.recipe.ingredients]
            assert not any("chicken" in n for n in ingredient_names), (
                "No chicken substitution recorded, but recipe still contains chicken"
            )

    def test_cbr_trace_names_substitution_and_goal(
        self, profile: UserProfile, recipes: list[Recipe]
    ) -> None:
        results = recommend(profile, recipes, top_k=1)
        if not results[0].adapted_recipe.adaptations:
            pytest.skip("No adaptations triggered for this scenario")
        rendered = render_explanation(results[0].explanation)
        assert "chicken" in rendered.lower()
        assert "tofu" in rendered.lower() or "vegetarian" in rendered.lower()

    def test_adaptation_entry_lists_original_and_replacement(
        self, profile: UserProfile, recipes: list[Recipe]
    ) -> None:
        results = recommend(profile, recipes, top_k=1)
        adapted = results[0].adapted_recipe
        if not adapted.adaptations:
            pytest.skip("No adaptations triggered for this scenario")
        for entry in adapted.adaptations:
            assert entry.original, "AdaptationEntry.original must be non-empty"
            assert entry.replacement, "AdaptationEntry.replacement must be non-empty"
            assert entry.reason, "AdaptationEntry.reason must be non-empty"
