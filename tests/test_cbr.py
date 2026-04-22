"""Tests for cbr.py — CBR Retrieve–Reuse–Revise–Retain."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from useitup.cbr import (
    FEATURE_DIM,
    AdaptedRecipe,
    CBRAdapter,
    CBRMatch,
    CBRRetriever,
    FeatureBreakdown,
    recipe_feature_vector,
    record_success,
)
from useitup.profile import RatingEntry, SoftPreferences, UserProfile
from useitup.schemas import Ingredient, Recipe


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_ingredient(name: str, category: str = "vegetable") -> Ingredient:
    return Ingredient(name=name, category=category)  # type: ignore[arg-type]


def _italian_recipe(recipe_id: str = "r-italian") -> Recipe:
    return Recipe(
        id=recipe_id,
        name="Pasta Primavera",
        ingredients=[
            _make_ingredient("spaghetti", "grain"),
            _make_ingredient("garlic", "vegetable"),
            _make_ingredient("olive oil", "fat"),
        ],
        cuisine="Italian",
        dietary_tags=["vegan", "vegetarian"],
        prep_time_min=10,
        cook_time_min=15,
        difficulty=2,
        nutrition={},
        flavor_profile=["savory", "fresh"],
        instructions=["Boil spaghetti.", "Sauté garlic in olive oil.", "Toss together."],
    )


def _indian_recipe(recipe_id: str = "r-indian") -> Recipe:
    return Recipe(
        id=recipe_id,
        name="Chicken Tikka",
        ingredients=[
            _make_ingredient("chicken breast", "protein"),
            _make_ingredient("yogurt", "dairy"),
            _make_ingredient("garam masala", "spice"),
        ],
        cuisine="Indian",
        dietary_tags=["high-protein", "nut-free"],
        prep_time_min=20,
        cook_time_min=30,
        difficulty=3,
        nutrition={},
        flavor_profile=["spicy", "savory", "rich"],
        instructions=["Marinate chicken in yogurt and spices.", "Grill until cooked."],
    )


def _mexican_recipe(recipe_id: str = "r-mexican") -> Recipe:
    return Recipe(
        id=recipe_id,
        name="Bean Tacos",
        ingredients=[
            _make_ingredient("black beans", "protein"),
            _make_ingredient("corn tortillas", "grain"),
            _make_ingredient("avocado", "fat"),
        ],
        cuisine="Mexican",
        dietary_tags=["vegan", "vegetarian", "gluten-free"],
        prep_time_min=10,
        cook_time_min=10,
        difficulty=1,
        nutrition={},
        flavor_profile=["savory", "fresh", "spicy"],
        instructions=["Warm tortillas.", "Fill with beans and avocado.", "Serve fresh."],
    )


def _base_profile(**kwargs) -> UserProfile:
    defaults: dict = {
        "user_id": "test-user",
        "hard_constraints": [],
        "soft_preferences": SoftPreferences(),
        "rating_history": [],
        "pantry": [],
    }
    defaults.update(kwargs)
    return UserProfile(**defaults)


def _with_ratings(profile: UserProfile, ratings: dict[str, int]) -> UserProfile:
    entries = [
        RatingEntry(recipe_id=rid, rating=r, timestamp="2024-01-01T00:00:00")
        for rid, r in ratings.items()
    ]
    return profile.model_copy(update={"rating_history": entries})


@pytest.fixture
def substitutions_file(tmp_path: Path) -> Path:
    subs = [
        {
            "original": "chicken",
            "replacement": "tofu",
            "reason": "plant-based protein substitute",
            "goals": ["vegetarian", "vegan"],
        },
        {
            "original": "yogurt",
            "replacement": "coconut yogurt",
            "reason": "dairy-free substitute",
            "goals": ["vegan", "dairy_free"],
        },
        {
            "original": "beef",
            "replacement": "lentils",
            "reason": "plant-based substitute",
            "goals": ["vegetarian", "vegan"],
        },
        {
            "original": "butter",
            "replacement": "coconut oil",
            "reason": "dairy-free fat substitute",
            "goals": ["vegan", "dairy_free"],
        },
    ]
    path = tmp_path / "substitutions.json"
    path.write_text(json.dumps(subs))
    return path


# ---------------------------------------------------------------------------
# recipe_feature_vector
# ---------------------------------------------------------------------------

class TestRecipeFeatureVector:
    def test_output_length(self) -> None:
        vec = recipe_feature_vector(_italian_recipe())
        assert vec.shape == (FEATURE_DIM,)

    def test_cuisine_one_hot(self) -> None:
        from useitup.cbr import CUISINES, _CUISINE_SLICE
        vec = recipe_feature_vector(_italian_recipe())
        cuisine_part = vec[_CUISINE_SLICE]
        assert cuisine_part[CUISINES.index("Italian")] == 1.0
        assert cuisine_part.sum() == 1.0

    def test_protein_other_when_no_protein_ingredient(self) -> None:
        from useitup.cbr import PROTEINS, _PROTEIN_SLICE
        vec = recipe_feature_vector(_italian_recipe())  # no protein ingredient
        protein_part = vec[_PROTEIN_SLICE]
        assert protein_part[PROTEINS.index("other")] == 1.0
        assert protein_part.sum() == 1.0

    def test_protein_chicken(self) -> None:
        from useitup.cbr import PROTEINS, _PROTEIN_SLICE
        vec = recipe_feature_vector(_indian_recipe())
        protein_part = vec[_PROTEIN_SLICE]
        assert protein_part[PROTEINS.index("chicken")] == 1.0

    def test_flavor_multi_hot(self) -> None:
        from useitup.cbr import FLAVORS, _FLAVOR_SLICE
        vec = recipe_feature_vector(_italian_recipe())  # savory, fresh
        flavor_part = vec[_FLAVOR_SLICE]
        assert flavor_part[FLAVORS.index("savory")] == 1.0
        assert flavor_part[FLAVORS.index("fresh")] == 1.0
        assert flavor_part[FLAVORS.index("spicy")] == 0.0

    def test_difficulty_normalized(self) -> None:
        from useitup.cbr import _DIFF_SLICE
        vec = recipe_feature_vector(_italian_recipe())  # difficulty=2 → (2-1)/4=0.25
        assert vec[_DIFF_SLICE][0] == pytest.approx(0.25)

    def test_prep_time_normalized(self) -> None:
        from useitup.cbr import MAX_PREP_TIME, _PREP_SLICE
        vec = recipe_feature_vector(_italian_recipe())  # 10 min
        assert vec[_PREP_SLICE][0] == pytest.approx(10.0 / MAX_PREP_TIME)

    def test_prep_time_capped_at_one(self) -> None:
        from useitup.cbr import _PREP_SLICE
        long_recipe = _italian_recipe().model_copy(update={"prep_time_min": 999})
        vec = recipe_feature_vector(long_recipe)
        assert vec[_PREP_SLICE][0] == pytest.approx(1.0)

    def test_cooking_method_inferred_from_instructions(self) -> None:
        from useitup.cbr import COOKING_METHODS, _METHOD_SLICE
        vec = recipe_feature_vector(_italian_recipe())  # "Boil" and "Sauté" in instructions
        method_part = vec[_METHOD_SLICE]
        assert method_part[COOKING_METHODS.index("boil")] == 1.0
        assert method_part[COOKING_METHODS.index("saute")] == 1.0


# ---------------------------------------------------------------------------
# CBRRetriever — centroid computation
# ---------------------------------------------------------------------------

class TestCentroidComputation:
    def test_centroid_equals_single_liked_recipe(self) -> None:
        recipe = _italian_recipe()
        profile = _with_ratings(_base_profile(), {"r-italian": 5})
        retriever = CBRRetriever([recipe], profile)
        expected = recipe_feature_vector(recipe)
        np.testing.assert_allclose(retriever._centroid, expected)

    def test_centroid_weighted_average(self) -> None:
        r1 = _italian_recipe("r1")
        r2 = _mexican_recipe("r2")
        profile = _with_ratings(_base_profile(), {"r1": 4, "r2": 4})
        retriever = CBRRetriever([r1, r2], profile)
        v1 = recipe_feature_vector(r1)
        v2 = recipe_feature_vector(r2)
        expected = np.average([v1, v2], axis=0, weights=[4.0, 4.0])
        np.testing.assert_allclose(retriever._centroid, expected)

    def test_rating_below_4_excluded_from_centroid(self) -> None:
        r1 = _italian_recipe("r1")
        r2 = _indian_recipe("r2")
        # r2 rated 3 — should not affect centroid
        profile = _with_ratings(_base_profile(), {"r1": 5, "r2": 3})
        retriever = CBRRetriever([r1, r2], profile)
        expected = recipe_feature_vector(r1)
        np.testing.assert_allclose(retriever._centroid, expected)

    def test_no_liked_recipes_means_cold_start(self) -> None:
        profile = _base_profile()
        retriever = CBRRetriever([_italian_recipe()], profile)
        assert retriever._cold_start is True


# ---------------------------------------------------------------------------
# CBRRetriever — retrieval ordering
# ---------------------------------------------------------------------------

class TestRetrieval:
    def test_retrieval_orders_by_similarity(self) -> None:
        liked = _italian_recipe("liked")
        similar = _italian_recipe("similar")  # same cuisine, same features
        dissimilar = _indian_recipe("dissimilar")

        profile = _with_ratings(_base_profile(), {"liked": 5})
        retriever = CBRRetriever([liked, similar, dissimilar], profile)
        matches = retriever.retrieve([similar, dissimilar])

        assert matches[0].recipe.id == "similar"
        assert matches[1].recipe.id == "dissimilar"
        assert matches[0].similarity_score >= matches[1].similarity_score

    def test_retrieve_returns_at_most_k(self) -> None:
        recipes = [_italian_recipe("r1"), _mexican_recipe("r2"), _indian_recipe("r3")]
        liked = _italian_recipe("liked")
        profile = _with_ratings(_base_profile(), {"liked": 5})
        retriever = CBRRetriever([liked] + recipes, profile)
        matches = retriever.retrieve(recipes, k=2)
        assert len(matches) == 2

    def test_nearest_past_recipe_populated(self) -> None:
        liked = _italian_recipe("liked")
        candidate = _italian_recipe("candidate")
        profile = _with_ratings(_base_profile(), {"liked": 5})
        retriever = CBRRetriever([liked], profile)
        matches = retriever.retrieve([candidate])
        assert matches[0].nearest_past_recipe is not None
        assert matches[0].nearest_past_recipe.id == "liked"

    def test_similarity_breakdown_fields_present(self) -> None:
        liked = _italian_recipe("liked")
        profile = _with_ratings(_base_profile(), {"liked": 5})
        retriever = CBRRetriever([liked], profile)
        matches = retriever.retrieve([_mexican_recipe()])
        bd = matches[0].similarity_breakdown
        assert isinstance(bd, FeatureBreakdown)
        for val in (bd.cuisine, bd.protein, bd.cooking_method, bd.flavor, bd.difficulty, bd.prep_time):
            assert 0.0 <= val <= 1.0 or val == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# CBRRetriever — cold-start fallback
# ---------------------------------------------------------------------------

class TestColdStart:
    def test_cold_start_no_ratings(self) -> None:
        profile = _base_profile()
        retriever = CBRRetriever([_italian_recipe()], profile)
        matches = retriever.retrieve([_italian_recipe(), _indian_recipe()])
        assert all(m.fallback_reason is not None for m in matches)
        assert all(m.nearest_past_recipe is None for m in matches)

    def test_cold_start_preferred_cuisine_ranked_first(self) -> None:
        prefs = SoftPreferences(preferred_cuisines=["Mexican"])
        profile = _base_profile(soft_preferences=prefs)
        retriever = CBRRetriever([], profile)
        italian = _italian_recipe()
        mexican = _mexican_recipe()
        matches = retriever.retrieve([italian, mexican], k=2)
        assert matches[0].recipe.id == mexican.id

    def test_cold_start_fallback_reason_recorded(self) -> None:
        profile = _base_profile(
            soft_preferences=SoftPreferences(preferred_cuisines=["Italian"])
        )
        retriever = CBRRetriever([], profile)
        matches = retriever.retrieve([_italian_recipe()])
        assert matches[0].fallback_reason is not None
        assert len(matches[0].fallback_reason) > 0


# ---------------------------------------------------------------------------
# CBRAdapter — adaptation
# ---------------------------------------------------------------------------

class TestAdaptation:
    def test_no_adaptation_when_no_conflict(self, substitutions_file: Path) -> None:
        adapter = CBRAdapter(substitutions_path=substitutions_file)
        profile = _base_profile(
            soft_preferences=SoftPreferences(goals=["vegetarian"])
        )
        match = CBRMatch(
            recipe=_italian_recipe(),  # no meat
            similarity_score=0.9,
            nearest_past_recipe=None,
            similarity_breakdown=FeatureBreakdown(0, 0, 0, 0, 0, 0),
        )
        result = adapter.adapt(match, profile)
        assert result.adaptations == []
        assert result.recipe is match.recipe

    def test_adaptation_triggers_on_vegetarian_goal(self, substitutions_file: Path) -> None:
        adapter = CBRAdapter(substitutions_path=substitutions_file)
        profile = _base_profile(
            soft_preferences=SoftPreferences(goals=["vegetarian"])
        )
        match = CBRMatch(
            recipe=_indian_recipe(),  # has chicken
            similarity_score=0.8,
            nearest_past_recipe=None,
            similarity_breakdown=FeatureBreakdown(0, 0, 0, 0, 0, 0),
        )
        result = adapter.adapt(match, profile)
        assert len(result.adaptations) >= 1
        orig_names = [a.original for a in result.adaptations]
        assert any("chicken" in o.lower() for o in orig_names)
        replaced_names = [a.replacement for a in result.adaptations]
        assert any("tofu" in r.lower() for r in replaced_names)

    def test_adaptation_replaces_ingredient_in_recipe(self, substitutions_file: Path) -> None:
        adapter = CBRAdapter(substitutions_path=substitutions_file)
        profile = _base_profile(
            soft_preferences=SoftPreferences(goals=["vegetarian"])
        )
        match = CBRMatch(
            recipe=_indian_recipe(),
            similarity_score=0.8,
            nearest_past_recipe=None,
            similarity_breakdown=FeatureBreakdown(0, 0, 0, 0, 0, 0),
        )
        result = adapter.adapt(match, profile)
        adapted_names = [ing.name.lower() for ing in result.recipe.ingredients]
        assert "chicken breast" not in adapted_names
        assert any("tofu" in n for n in adapted_names)

    def test_adaptation_logs_reason(self, substitutions_file: Path) -> None:
        adapter = CBRAdapter(substitutions_path=substitutions_file)
        profile = _base_profile(
            soft_preferences=SoftPreferences(goals=["vegetarian"])
        )
        match = CBRMatch(
            recipe=_indian_recipe(),
            similarity_score=0.8,
            nearest_past_recipe=None,
            similarity_breakdown=FeatureBreakdown(0, 0, 0, 0, 0, 0),
        )
        result = adapter.adapt(match, profile)
        assert all(a.reason for a in result.adaptations)

    def test_adaptation_does_not_trigger_on_hard_constraint(
        self, substitutions_file: Path
    ) -> None:
        """Recipes violating hard constraints should have been filtered by Stage 1.
        The adapter must NOT substitute for hard constraints — it only handles goals."""
        adapter = CBRAdapter(substitutions_path=substitutions_file)
        # hard constraint is dairy-free, but goals list is empty
        profile = _base_profile(
            hard_constraints=["dairy-free"],
            soft_preferences=SoftPreferences(goals=[]),
        )
        match = CBRMatch(
            recipe=_indian_recipe(),  # has yogurt/dairy — but no goal conflict
            similarity_score=0.7,
            nearest_past_recipe=None,
            similarity_breakdown=FeatureBreakdown(0, 0, 0, 0, 0, 0),
        )
        result = adapter.adapt(match, profile)
        # No adaptation because dairy_free is a constraint, not a goal
        assert result.adaptations == []

    def test_dairy_free_goal_triggers_adaptation(self, substitutions_file: Path) -> None:
        adapter = CBRAdapter(substitutions_path=substitutions_file)
        profile = _base_profile(
            soft_preferences=SoftPreferences(goals=["dairy_free"])
        )
        match = CBRMatch(
            recipe=_indian_recipe(),  # has yogurt
            similarity_score=0.7,
            nearest_past_recipe=None,
            similarity_breakdown=FeatureBreakdown(0, 0, 0, 0, 0, 0),
        )
        result = adapter.adapt(match, profile)
        replaced = [a.replacement for a in result.adaptations]
        assert any("coconut yogurt" in r for r in replaced)

    def test_adaptation_updates_category_and_dietary_tags(self, substitutions_file: Path) -> None:
        adapter = CBRAdapter(substitutions_path=substitutions_file)
        profile = _base_profile(
            soft_preferences=SoftPreferences(goals=["dairy_free"])
        )
        match = CBRMatch(
            recipe=_indian_recipe(),
            similarity_score=0.7,
            nearest_past_recipe=None,
            similarity_breakdown=FeatureBreakdown(0, 0, 0, 0, 0, 0),
        )
        result = adapter.adapt(match, profile)
        ingredients = {ing.name.lower(): ing.category for ing in result.recipe.ingredients}
        assert "coconut yogurt" in ingredients
        assert ingredients["coconut yogurt"] != "dairy"
        assert "dairy-free" in result.recipe.dietary_tags


# ---------------------------------------------------------------------------
# record_success (Retain step)
# ---------------------------------------------------------------------------

class TestRecordSuccess:
    def test_record_success_appends_rating(self) -> None:
        profile = _base_profile()
        updated = record_success(profile, "r-new", rating=5)
        assert len(updated.rating_history) == 1
        assert updated.rating_history[-1].recipe_id == "r-new"
        assert updated.rating_history[-1].rating == 5

    def test_record_success_does_not_mutate_original(self) -> None:
        profile = _base_profile()
        _ = record_success(profile, "r-new")
        assert len(profile.rating_history) == 0

    def test_record_success_default_rating_is_five(self) -> None:
        profile = _base_profile()
        updated = record_success(profile, "r-new")
        assert updated.rating_history[-1].rating == 5

    def test_record_success_benefits_future_retrieval(self) -> None:
        italian = _italian_recipe("r-italian")
        profile = _base_profile()
        updated = record_success(profile, "r-italian")

        retriever = CBRRetriever([italian], updated)
        assert retriever._cold_start is False
        matches = retriever.retrieve([_italian_recipe("candidate")])
        assert matches[0].similarity_score > 0.0
