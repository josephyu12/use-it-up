"""UseItUp end-to-end demo.

Zero-argument CLI that showcases the full three-stage pipeline on three
contrasting scenarios. Designed to run on the Yale Zoo with only the
packages listed in requirements.txt (pydantic, numpy).

Run from the project root:
    python scripts/demo.py

or with the editable install active:
    python -m scripts.demo
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SRC = _PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from useitup.data_loader import load_recipes
from useitup.explain import render_explanation
from useitup.matching import IngredientScorer
from useitup.pipeline import recommend
from useitup.profile import RatingEntry, SoftPreferences, UserProfile
from useitup.schemas import Recipe

_CURATED = _PROJECT_ROOT / "data" / "recipes_curated.json"
_SAMPLE = _PROJECT_ROOT / "data" / "recipes_sample.json"


@dataclass
class Scenario:
    title: str
    blurb: str
    profile: UserProfile


def _rule(ch: str = "=") -> str:
    return ch * 78


def _section(title: str) -> None:
    print()
    print(_rule("="))
    print(f"  {title}")
    print(_rule("="))


def _subsection(title: str) -> None:
    print()
    print(_rule("-"))
    print(f"  {title}")
    print(_rule("-"))


def _build_scenarios() -> list[Scenario]:
    gluten_free_student = UserProfile(
        user_id="demo_glutenfree",
        hard_constraints=["gluten-free", "nut-free"],
        soft_preferences=SoftPreferences(
            max_prep_time_min=40,
            preferred_cuisines=["Mexican", "Mediterranean"],
            goals=["high_protein", "low_cost"],
        ),
        pantry=[
            "black beans", "brown rice", "eggs", "garlic", "onion",
            "olive oil", "lime", "cilantro", "avocado", "cumin",
            "paprika", "canned tomatoes", "feta cheese",
        ],
        rating_history=[],
    )
    weekday_cook = UserProfile(
        user_id="demo_weekday",
        hard_constraints=[],
        soft_preferences=SoftPreferences(
            max_prep_time_min=30,
            preferred_cuisines=["Italian", "American"],
            goals=["quick"],
        ),
        pantry=[
            "chicken breast", "pasta", "garlic", "olive oil", "onion",
            "cherry tomatoes", "parmesan cheese", "basil", "salt", "pepper",
        ],
        rating_history=[
            RatingEntry(recipe_id="sample-001", rating=5, timestamp="2026-03-01T18:00:00"),
            RatingEntry(recipe_id="sample-007", rating=4, timestamp="2026-03-04T19:00:00"),
        ],
    )
    vegan_adapter = UserProfile(
        user_id="demo_vegan",
        hard_constraints=[],
        soft_preferences=SoftPreferences(
            max_prep_time_min=45,
            preferred_cuisines=["Indian", "Asian"],
            goals=["vegan", "high_protein"],
        ),
        pantry=[
            "tofu", "chickpeas", "coconut milk", "rice", "onion", "garlic",
            "ginger", "curry powder", "soy sauce", "sesame oil", "spinach",
            "cilantro",
        ],
        rating_history=[],
    )
    return [
        Scenario(
            title="Scenario A — Gluten-free student, pantry-first",
            blurb=(
                "Hard constraints: gluten-free, nut-free. Goals: high_protein, "
                "low_cost. Demonstrates hard-rule rejection, coverage scoring, "
                "and pantry utilization reporting."
            ),
            profile=gluten_free_student,
        ),
        Scenario(
            title="Scenario B — Weekday cook with rating history (CBR warm start)",
            blurb=(
                "Quick goal; short prep window. Has two 4/5-star past ratings, "
                "so CBR retrieval uses a weighted centroid instead of cold "
                "start. Demonstrates the Retrieve step of case-based reasoning."
            ),
            profile=weekday_cook,
        ),
        Scenario(
            title="Scenario C — Vegan goal forcing ingredient adaptation",
            blurb=(
                "Demonstrates the Revise step: if the retrieved case contains "
                "chicken or dairy, CBRAdapter substitutes via substitutions.json "
                "and the explanation records every swap."
            ),
            profile=vegan_adapter,
        ),
    ]


def _print_summary_line(recipe: Recipe, score) -> None:
    tags = ", ".join(recipe.dietary_tags) if recipe.dietary_tags else "—"
    print(
        f"  {recipe.name}  "
        f"[{recipe.cuisine}, {recipe.prep_time_min}+{recipe.cook_time_min} min, "
        f"tags: {tags}]"
    )
    print(
        f"    coverage={score.coverage:.0%}  weighted={score.weighted_coverage:.0%}  "
        f"essentials={score.essential_overlap_count}/{score.essential_total}"
    )
    if score.matched_ingredients:
        print(f"    matched:  {', '.join(score.matched_ingredients)}")
    if score.missing_ingredients:
        print(f"    missing:  {', '.join(score.missing_ingredients)}")


def _run_scenario(scenario: Scenario, recipes: list[Recipe]) -> None:
    _section(scenario.title)
    print(scenario.blurb)
    print()
    print(f"Pantry ({len(scenario.profile.pantry)} items): "
          f"{', '.join(scenario.profile.pantry)}")
    if scenario.profile.hard_constraints:
        print(f"Hard constraints: {', '.join(scenario.profile.hard_constraints)}")
    prefs = scenario.profile.soft_preferences
    soft_bits: list[str] = []
    if prefs.goals:
        soft_bits.append(f"goals={prefs.goals}")
    if prefs.preferred_cuisines:
        soft_bits.append(f"preferred={prefs.preferred_cuisines}")
    if prefs.max_prep_time_min is not None:
        soft_bits.append(f"max_prep={prefs.max_prep_time_min}min")
    if soft_bits:
        print("Soft preferences: " + "; ".join(soft_bits))
    if scenario.profile.rating_history:
        print(f"Rating history: {len(scenario.profile.rating_history)} past ratings")

    try:
        results = recommend(scenario.profile, recipes, top_k=3)
    except ValueError as e:
        print(f"\nNo viable recommendation: {e}")
        return

    scorer = IngredientScorer()

    _subsection("Top 3 recommendations")
    for i, rec in enumerate(results, 1):
        score = scorer.score(rec.adapted_recipe.recipe, scenario.profile.pantry)
        print(f"\n#{i}")
        _print_summary_line(rec.adapted_recipe.recipe, score)
        if rec.adapted_recipe.adaptations:
            for adapt in rec.adapted_recipe.adaptations:
                print(f"    adapted: {adapt.original} → {adapt.replacement}"
                      f"  ({adapt.reason})")

    _subsection("Full explanation for top pick")
    print(render_explanation(results[0].explanation))

    passed = sum(1 for e in results[0].decision_log if e.passed)
    failed = sum(1 for e in results[0].decision_log if not e.passed)
    _subsection("Decision log summary")
    print(f"  Rules fired: {len(results[0].decision_log)}  "
          f"(passed: {passed}, failed: {failed})")
    failed_by_rule: dict[str, int] = {}
    for entry in results[0].decision_log:
        if not entry.passed:
            failed_by_rule[entry.rule_name] = failed_by_rule.get(entry.rule_name, 0) + 1
    for rule_name, count in sorted(failed_by_rule.items(), key=lambda kv: -kv[1]):
        print(f"    {rule_name}: {count} recipes eliminated")


def _choose_catalog() -> tuple[list[Recipe], Path]:
    path = _CURATED if _CURATED.exists() else _SAMPLE
    return load_recipes(path), path


def main() -> int:
    print(_rule("#"))
    print("  UseItUp — Explainable Recipe Recommendation Demo")
    print("  CS 4580/5580 Final Project")
    print(_rule("#"))

    recipes, catalog_path = _choose_catalog()
    print(f"\nLoaded {len(recipes)} recipes from {catalog_path.name}.")
    print("Pipeline: Stage 1 (matching + rules) → Stage 2 (CBR) → Stage 3 (explanation).")

    scenarios = _build_scenarios()
    for scenario in scenarios:
        _run_scenario(scenario, recipes)

    print()
    print(_rule("#"))
    print("  Demo complete.")
    print(f"  Ran {len(scenarios)} scenarios against {len(recipes)} recipes.")
    print("  For the interactive version, open notebooks/UseItUp.ipynb.")
    print(_rule("#"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
