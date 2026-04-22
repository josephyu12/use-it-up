"""Top-level orchestrator: wires Stage 1 → Stage 2 → Stage 3."""

from __future__ import annotations

from dataclasses import dataclass

from useitup.cbr import AdaptedRecipe, CBRAdapter, CBRMatch, CBRRetriever
from useitup.explain import Explanation, generate_explanation
from useitup.matching import (
    AllergyRule,
    DecisionEntry,
    DietaryRule,
    FilterEngine,
    FilterResult,
    PantryCoverageRule,
)
from useitup.profile import UserProfile
from useitup.schemas import Recipe


@dataclass
class Recommendation:
    """A single recommendation bundling the adapted recipe, explanation, and debug data."""

    adapted_recipe: AdaptedRecipe
    explanation: Explanation
    decision_log: list[DecisionEntry]
    cbr_match: CBRMatch
    filter_result: FilterResult


def recommend(
    profile: UserProfile,
    recipes: list[Recipe],
    top_k: int = 1,
) -> list[Recommendation]:
    """Run the full 3-stage pipeline; return top_k Recommendation objects."""
    filter_engine = FilterEngine()
    filter_result: FilterResult = filter_engine.run(recipes, profile)

    if not filter_result.survivors and profile.pantry:
        relaxed_engine = FilterEngine(hard_rules=[
            AllergyRule(),
            DietaryRule(),
            PantryCoverageRule(threshold=0.35),
        ])
        filter_result = relaxed_engine.run(recipes, profile)

    if not filter_result.survivors:
        raise ValueError(
            "No recipes survived the filtering stage. "
            "Try relaxing hard constraints or expanding the pantry."
        )

    survivor_recipes = [sr.recipe for sr in filter_result.survivors]
    retriever = CBRRetriever(recipes, profile)
    cbr_matches: list[CBRMatch] = retriever.retrieve(
        survivor_recipes,
        k=min(len(survivor_recipes), max(top_k * 2, 12)),
    )

    adapter = CBRAdapter()
    results: list[Recommendation] = []
    # Post-adaptation check: require_all_essentials=False because
    # substituted ingredients (e.g. chicken → tofu) won't be in the user's
    # pantry and would otherwise fail the essential coverage test.
    hard_rules = [AllergyRule(), DietaryRule(), PantryCoverageRule(require_all_essentials=False)]
    for match in cbr_matches:
        adapted = adapter.adapt(match, profile)
        if not _passes_hard_rules(adapted.recipe, profile, hard_rules):
            continue
        explanation = generate_explanation(
            adapted=adapted,
            filter_result=filter_result,
            cbr_match=match,
            profile=profile,
            all_recipes=recipes,
        )
        results.append(Recommendation(
            adapted_recipe=adapted,
            explanation=explanation,
            decision_log=filter_result.decision_log,
            cbr_match=match,
            filter_result=filter_result,
        ))
        if len(results) >= top_k:
            break

    if not results:
        raise ValueError(
            "No recipes survived the filtering stage. "
            "Try relaxing hard constraints or expanding the pantry."
        )
    return results


def run_pipeline(
    recipes: list[Recipe],
    profile: UserProfile,
    cbr_k: int = 5,
) -> tuple[AdaptedRecipe, Explanation]:
    """Legacy entry point kept for backward compatibility."""
    results = recommend(profile, recipes, top_k=1)
    return results[0].adapted_recipe, results[0].explanation


def _passes_hard_rules(recipe: Recipe, profile: UserProfile, rules) -> bool:
    return all(rule.applies(recipe, profile) for rule in rules)
