"""Tests for pipeline.py — recommend() API and notebook smoke test."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from useitup.data_loader import load_recipes
from useitup.pipeline import Recommendation, recommend, run_pipeline
from useitup.profile import SoftPreferences, UserProfile

_ROOT = Path(__file__).parent.parent
_DATA_DIR = _ROOT / "data"
_PROFILES_DIR = _DATA_DIR / "profiles"


@pytest.fixture
def sample_recipes():
    return load_recipes(_DATA_DIR / "recipes_sample.json")


@pytest.fixture
def demo_profile() -> UserProfile:
    from useitup.profile import load_profile
    return load_profile("demo_user", base_dir=_PROFILES_DIR)


@pytest.fixture
def minimal_profile() -> UserProfile:
    return UserProfile(
        user_id="test",
        pantry=["chicken", "garlic", "olive oil", "onion", "rice", "cumin", "paprika", "lime"],
        hard_constraints=[],
        soft_preferences=SoftPreferences(max_prep_time_min=60, goals=[]),
    )


# ── recommend() ──────────────────────────────────────────────────────────────

class TestRecommend:
    def test_returns_list_of_recommendations(self, demo_profile, sample_recipes):
        results = recommend(demo_profile, sample_recipes, top_k=1)
        assert isinstance(results, list)
        assert len(results) == 1
        assert isinstance(results[0], Recommendation)

    def test_top_k_respected(self, demo_profile, sample_recipes):
        results = recommend(demo_profile, sample_recipes, top_k=3)
        assert len(results) <= 3

    def test_recommendation_fields_populated(self, demo_profile, sample_recipes):
        r = recommend(demo_profile, sample_recipes, top_k=1)[0]
        assert r.adapted_recipe.recipe.name
        assert r.explanation.goal_trace
        assert r.explanation.counterfactual
        assert r.explanation.cbr_trace
        assert r.explanation.ingredient_utilization_report
        assert r.decision_log
        assert r.cbr_match is not None
        assert r.filter_result.survivors

    def test_decision_log_non_empty(self, demo_profile, sample_recipes):
        r = recommend(demo_profile, sample_recipes)[0]
        assert len(r.decision_log) > 0

    def test_raises_when_no_survivors(self, sample_recipes):
        restrictive = UserProfile(
            user_id="test",
            pantry=[],
            hard_constraints=["vegan", "gluten-free", "nut-free", "dairy-free"],
            soft_preferences=SoftPreferences(max_prep_time_min=5),
        )
        with pytest.raises(ValueError, match="No recipes survived"):
            recommend(restrictive, sample_recipes)

    def test_top_k_default_is_1(self, demo_profile, sample_recipes):
        results = recommend(demo_profile, sample_recipes)
        assert len(results) == 1

    def test_served_recipe_still_respects_hard_constraints_after_adaptation(self):
        from useitup.data_loader import load_recipes

        recipes = load_recipes(_DATA_DIR / "recipes_curated.json")
        restrictive = UserProfile(
            user_id="test",
            pantry=["bread", "olive oil", "butter"],
            hard_constraints=["dairy-free"],
            soft_preferences=SoftPreferences(goals=["dairy_free"]),
        )
        with pytest.raises(ValueError, match="No recipes survived"):
            recommend(restrictive, recipes, top_k=1)

    def test_all_results_share_same_filter_result(self, demo_profile, sample_recipes):
        results = recommend(demo_profile, sample_recipes, top_k=3)
        if len(results) > 1:
            assert results[0].filter_result is results[1].filter_result

    def test_each_result_has_its_own_cbr_match(self, demo_profile, sample_recipes):
        """Each Recommendation carries its own CBRMatch — not a shared list."""
        results = recommend(demo_profile, sample_recipes, top_k=3)
        if len(results) > 1:
            # Distinct recommendations should carry distinct CBRMatch objects
            # (one per retrieved case), though they share the upstream FilterResult.
            ids = {id(r.cbr_match) for r in results}
            assert len(ids) == len(results)

    def test_minimal_profile_cold_start(self, minimal_profile, sample_recipes):
        # With a generous pantry and no hard constraints, the relaxed
        # essential-coverage check should still surface recommendations.
        results = recommend(minimal_profile, sample_recipes, top_k=1)
        assert len(results) == 1
        assert results[0].adapted_recipe.recipe.name

    def test_explanation_markdown_contains_recipe_name(self, demo_profile, sample_recipes):
        from useitup.explain import render_explanation
        r = recommend(demo_profile, sample_recipes)[0]
        md = render_explanation(r.explanation)
        assert r.adapted_recipe.recipe.name in md


# ── run_pipeline() backward compat ───────────────────────────────────────────

class TestRunPipelineCompat:
    def test_returns_tuple(self, demo_profile, sample_recipes):
        adapted, explanation = run_pipeline(sample_recipes, demo_profile)
        assert adapted.recipe.name
        assert explanation.goal_trace

    def test_same_result_as_recommend(self, demo_profile, sample_recipes):
        adapted_new, expl_new = run_pipeline(sample_recipes, demo_profile)
        r = recommend(demo_profile, sample_recipes)[0]
        assert adapted_new.recipe.id == r.adapted_recipe.recipe.id


# ── Notebook smoke test ───────────────────────────────────────────────────────

def test_notebook_executes(tmp_path):
    """Ensure the notebook runs top-to-bottom without errors via nbconvert."""
    out = tmp_path / "executed.ipynb"
    nb_path = _ROOT / "notebooks" / "UseItUp.ipynb"
    result = subprocess.run(
        [
            sys.executable, "-m", "jupyter", "nbconvert",
            "--to", "notebook",
            "--execute",
            "--ExecutePreprocessor.timeout=120",
            "--output", str(out),
            str(nb_path),
        ],
        capture_output=True,
        text=True,
        cwd=str(_ROOT),
        timeout=150,
    )
    assert result.returncode == 0, (
        f"Notebook execution failed.\n"
        f"STDOUT:\n{result.stdout}\n"
        f"STDERR:\n{result.stderr}"
    )
    assert out.exists(), "nbconvert did not produce an output file"
