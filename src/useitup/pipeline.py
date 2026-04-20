"""Top-level orchestrator: wires Stage 1 → Stage 2 → Stage 3."""

from __future__ import annotations

from dataclasses import dataclass

from useitup.cbr import AdaptedRecipe, CBRAdapter, CBRMatch, CBRRetriever
from useitup.explain import Explanation, generate_explanation
from useitup.matching import DecisionEntry, FilterEngine, FilterResult
from useitup.profile import UserProfile
from useitup.schemas import Recipe


@dataclass
class Recommendation:
    """A single recommendation bundling the adapted recipe, explanation, and debug data."""

    adapted_recipe: AdaptedRecipe
    explanation: Explanation
    decision_log: list[DecisionEntry]
    cbr_matches: list[CBRMatch]
    filter_result: FilterResult


def recommend(
    profile: UserProfile,
    recipes: list[Recipe],
    top_k: int = 1,
) -> list[Recommendation]:
    """Run the full 3-stage pipeline; return top_k Recommendation objects."""
    filter_engine = FilterEngine()
    filter_result: FilterResult = filter_engine.run(recipes, profile)

    if not filter_result.survivors:
        raise ValueError(
            "No recipes survived the filtering stage. "
            "Try relaxing hard constraints or expanding the pantry."
        )

    survivor_recipes = [sr.recipe for sr in filter_result.survivors]
    retriever = CBRRetriever(recipes, profile)
    cbr_matches: list[CBRMatch] = retriever.retrieve(survivor_recipes, k=max(top_k, 5))

    adapter = CBRAdapter()
    results: list[Recommendation] = []
    for match in cbr_matches[:top_k]:
        adapted = adapter.adapt(match, profile)
        explanation = generate_explanation(
            adapted=adapted,
            filter_result=filter_result,
            cbr_matches=cbr_matches,
            profile=profile,
            all_recipes=recipes,
        )
        results.append(Recommendation(
            adapted_recipe=adapted,
            explanation=explanation,
            decision_log=filter_result.decision_log,
            cbr_matches=cbr_matches,
            filter_result=filter_result,
        ))
    return results


def run_pipeline(
    recipes: list[Recipe],
    profile: UserProfile,
    cbr_k: int = 5,
) -> tuple[AdaptedRecipe, Explanation]:
    """Legacy entry point kept for backward compatibility."""
    results = recommend(profile, recipes, top_k=1)
    return results[0].adapted_recipe, results[0].explanation
