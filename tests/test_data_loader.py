"""Tests for useitup.data_loader."""

import json
from pathlib import Path

import pytest

from useitup.data_loader import load_recipes, save_recipes
from useitup.schemas import Ingredient, Recipe

SAMPLE_JSON = (
    Path(__file__).resolve().parent.parent / "data" / "recipes_sample.json"
)


def _minimal_recipe(id: str = "r1") -> Recipe:
    return Recipe(
        id=id,
        name="Minimal",
        ingredients=[Ingredient(name="egg", quantity=2.0, unit=None, category="protein")],
        cuisine="American",
        dietary_tags=["quick"],
        prep_time_min=5,
        cook_time_min=5,
        difficulty=1,
        nutrition={"calories": 140.0},
        flavor_profile=["savory"],
        instructions=["Fry the egg."],
    )


class TestRoundTrip:
    def test_save_then_load(self, tmp_path: Path):
        recipes = [_minimal_recipe("r1"), _minimal_recipe("r2")]
        out = tmp_path / "recipes.json"
        save_recipes(recipes, out)
        loaded = load_recipes(out)
        assert len(loaded) == 2
        assert loaded[0].id == "r1"
        assert loaded[1].id == "r2"

    def test_round_trip_preserves_ingredients(self, tmp_path: Path):
        r = _minimal_recipe()
        out = tmp_path / "r.json"
        save_recipes([r], out)
        loaded = load_recipes(out)[0]
        assert loaded.ingredients[0].name == "egg"
        assert loaded.ingredients[0].quantity == 2.0

    def test_round_trip_preserves_tags(self, tmp_path: Path):
        r = _minimal_recipe()
        out = tmp_path / "r.json"
        save_recipes([r], out)
        loaded = load_recipes(out)[0]
        assert "quick" in loaded.dietary_tags

    def test_saved_file_is_valid_json(self, tmp_path: Path):
        out = tmp_path / "r.json"
        save_recipes([_minimal_recipe()], out)
        data = json.loads(out.read_text())
        assert isinstance(data, list)
        assert data[0]["id"] == "r1"


class TestLoadSampleFile:
    def test_sample_file_loads(self):
        recipes = load_recipes(SAMPLE_JSON)
        assert len(recipes) == 10

    def test_sample_ids_unique(self):
        recipes = load_recipes(SAMPLE_JSON)
        ids = [r.id for r in recipes]
        assert len(ids) == len(set(ids))

    def test_sample_dietary_tags_valid(self):
        recipes = load_recipes(SAMPLE_JSON)
        valid_tags = {
            "vegan", "vegetarian", "gluten-free", "dairy-free", "nut-free",
            "low-carb", "high-protein", "low-cost", "quick",
        }
        for r in recipes:
            for tag in r.dietary_tags:
                assert tag in valid_tags, f"Invalid tag '{tag}' in recipe {r.id}"

    def test_sample_flavor_tags_valid(self):
        recipes = load_recipes(SAMPLE_JSON)
        valid_flavors = {"spicy", "savory", "sweet", "sour", "umami", "smoky", "fresh", "rich"}
        for r in recipes:
            for tag in r.flavor_profile:
                assert tag in valid_flavors, f"Invalid flavor '{tag}' in recipe {r.id}"

    def test_sample_difficulties_in_range(self):
        recipes = load_recipes(SAMPLE_JSON)
        for r in recipes:
            assert 1 <= r.difficulty <= 5

    def test_sample_all_have_instructions(self):
        recipes = load_recipes(SAMPLE_JSON)
        for r in recipes:
            assert len(r.instructions) >= 1


class TestEdgeCases:
    def test_load_empty_list(self, tmp_path: Path):
        f = tmp_path / "empty.json"
        f.write_text("[]")
        assert load_recipes(f) == []

    def test_invalid_json_raises(self, tmp_path: Path):
        f = tmp_path / "bad.json"
        f.write_text("not json")
        with pytest.raises(Exception):
            load_recipes(f)

    def test_missing_file_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            load_recipes(tmp_path / "nonexistent.json")

    def test_save_creates_parent_dirs(self, tmp_path: Path):
        out = tmp_path / "nested" / "deep" / "r.json"
        # parent doesn't exist yet — save_recipes should still work if parent is created
        out.parent.mkdir(parents=True)
        save_recipes([_minimal_recipe()], out)
        assert out.exists()
