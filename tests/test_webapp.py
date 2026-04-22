from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from useitup.webapp import create_app

_CURATED_RECIPES = Path(__file__).parent.parent / "data" / "recipes_curated.json"
client = TestClient(create_app(recipes_path=_CURATED_RECIPES))


class TestRecommendApi:
    def test_dairy_free_request_does_not_recommend_grilled_cheese(self) -> None:
        resp = client.post("/api/recommend", json={
            "pantry": ["bread", "olive oil", "butter"],
            "hard_constraints": ["dairy-free"],
            "goals": ["dairy_free"],
            "preferred_cuisines": [],
            "max_prep_time_min": 45,
            "top_k": 3,
            "rating_history": [],
        })

        assert resp.status_code == 200
        data = resp.json()
        assert data["recommendations"] == []
        assert "No recipes survived" in data["error"]

    def test_match_fields_are_scored_from_adapted_recipe(self) -> None:
        resp = client.post("/api/recommend", json={
            "pantry": ["bread", "butter", "cheddar cheese"],
            "hard_constraints": [],
            "goals": [],
            "preferred_cuisines": ["American"],
            "max_prep_time_min": 20,
            "top_k": 1,
            "rating_history": [],
        })

        assert resp.status_code == 200
        data = resp.json()
        assert data["recommendations"]
        rec = data["recommendations"][0]
        ingredient_names = {ingredient["name"].lower() for ingredient in rec["recipe"]["ingredients"]}
        matched = set(rec["match"]["matched_ingredients"])
        missing = set(rec["match"]["missing_ingredients"])

        assert matched.issubset(ingredient_names)
        assert missing.issubset(ingredient_names)
