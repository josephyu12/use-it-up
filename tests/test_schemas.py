"""Tests for useitup.schemas."""

import pytest
from pydantic import ValidationError

from useitup.schemas import Ingredient, Recipe


def make_ingredient(**kwargs) -> dict:
    base = {"name": "chicken", "quantity": 100.0, "unit": "g", "category": "protein"}
    return {**base, **kwargs}


def make_recipe(**kwargs) -> dict:
    base = {
        "id": "r1",
        "name": "Test Recipe",
        "ingredients": [make_ingredient()],
        "cuisine": "American",
        "dietary_tags": ["vegan"],
        "prep_time_min": 10,
        "cook_time_min": 20,
        "difficulty": 3,
        "nutrition": {"calories": 400.0},
        "flavor_profile": ["savory"],
        "instructions": ["Cook it."],
    }
    return {**base, **kwargs}


# ---------------------------------------------------------------------------
# Ingredient tests
# ---------------------------------------------------------------------------

class TestIngredient:
    def test_full_ingredient(self):
        ing = Ingredient(name="chicken", quantity=200.0, unit="g", category="protein")
        assert ing.name == "chicken"
        assert ing.quantity == 200.0
        assert ing.unit == "g"
        assert ing.category == "protein"

    def test_optional_quantity_and_unit(self):
        ing = Ingredient(name="salt", quantity=None, unit=None, category="spice")
        assert ing.quantity is None
        assert ing.unit is None

    def test_invalid_category(self):
        with pytest.raises(ValidationError):
            Ingredient(name="salt", quantity=None, unit=None, category="mineral")

    def test_empty_name_raises(self):
        with pytest.raises(ValidationError):
            Ingredient(name="   ", quantity=None, unit=None, category="spice")

    def test_name_stripped(self):
        ing = Ingredient(name="  basil  ", quantity=None, unit=None, category="vegetable")
        assert ing.name == "basil"

    def test_all_categories(self):
        categories = [
            "protein", "vegetable", "grain", "dairy",
            "spice", "fat", "condiment", "other",
        ]
        for cat in categories:
            ing = Ingredient(name="item", quantity=None, unit=None, category=cat)
            assert ing.category == cat


# ---------------------------------------------------------------------------
# Recipe tests
# ---------------------------------------------------------------------------

class TestRecipe:
    def test_valid_recipe(self):
        r = Recipe.model_validate(make_recipe())
        assert r.id == "r1"
        assert r.name == "Test Recipe"
        assert len(r.ingredients) == 1

    def test_invalid_difficulty_out_of_range(self):
        with pytest.raises(ValidationError):
            Recipe.model_validate(make_recipe(difficulty=6))

    def test_invalid_difficulty_zero(self):
        with pytest.raises(ValidationError):
            Recipe.model_validate(make_recipe(difficulty=0))

    def test_all_valid_difficulties(self):
        for d in range(1, 6):
            r = Recipe.model_validate(make_recipe(difficulty=d))
            assert r.difficulty == d

    def test_invalid_dietary_tag(self):
        with pytest.raises(ValidationError):
            Recipe.model_validate(make_recipe(dietary_tags=["pescatarian"]))

    def test_invalid_flavor_tag(self):
        with pytest.raises(ValidationError):
            Recipe.model_validate(make_recipe(flavor_profile=["tangy"]))

    def test_all_dietary_tags_valid(self):
        tags = [
            "vegan", "vegetarian", "gluten-free", "dairy-free", "nut-free",
            "low-carb", "high-protein", "low-cost", "quick",
        ]
        r = Recipe.model_validate(make_recipe(dietary_tags=tags))
        assert set(r.dietary_tags) == set(tags)

    def test_all_flavor_tags_valid(self):
        tags = ["spicy", "savory", "sweet", "sour", "umami", "smoky", "fresh", "rich"]
        r = Recipe.model_validate(make_recipe(flavor_profile=tags))
        assert set(r.flavor_profile) == set(tags)

    def test_empty_ingredients_raises(self):
        with pytest.raises(ValidationError):
            Recipe.model_validate(make_recipe(ingredients=[]))

    def test_duplicate_tags_deduplicated(self):
        r = Recipe.model_validate(make_recipe(dietary_tags=["vegan", "vegan"]))
        assert r.dietary_tags.count("vegan") == 1

    def test_duplicate_flavor_deduplicated(self):
        r = Recipe.model_validate(make_recipe(flavor_profile=["savory", "savory"]))
        assert r.flavor_profile.count("savory") == 1

    def test_empty_dietary_tags_allowed(self):
        r = Recipe.model_validate(make_recipe(dietary_tags=[]))
        assert r.dietary_tags == []

    def test_nutrition_arbitrary_keys(self):
        r = Recipe.model_validate(make_recipe(nutrition={"calories": 300.0, "fiber_g": 5.0}))
        assert r.nutrition["fiber_g"] == 5.0
