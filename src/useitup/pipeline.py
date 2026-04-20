"""Top-level orchestrator: wires Stage 1 → Stage 2 → Stage 3."""

from __future__ import annotations

from useitup.cbr import AdaptedRecipe, CBRAdapter, CBRMatch, CBRRetriever
from useitup.explain import Explanation, generate_explanation
from useitup.matching import FilterEngine, FilterResult
from useitup.profile import UserProfile
from useitup.schemas import Recipe


def run_pipeline(
    recipes: list[Recipe],
    profile: UserProfile,
    cbr_k: int = 5,
) -> tuple[AdaptedRecipe, Explanation]:
    """Run the full 3-stage recommendation pipeline and return the top adapted recipe + explanation."""
    # Stage 1: Rule-based filtering
    filter_engine = FilterEngine()
    filter_result: FilterResult = filter_engine.run(recipes, profile)

    if not filter_result.survivors:
        raise ValueError(
            "No recipes survived the filtering stage. "
            "Try relaxing hard constraints or expanding the pantry."
        )

    # Stage 2: CBR retrieve + adapt
    survivor_recipes = [sr.recipe for sr in filter_result.survivors]
    retriever = CBRRetriever(recipes, profile)
    cbr_matches: list[CBRMatch] = retriever.retrieve(survivor_recipes, k=cbr_k)

    adapter = CBRAdapter()
    adapted: AdaptedRecipe = adapter.adapt(cbr_matches[0], profile)

    # Stage 3: Explanation generation
    explanation: Explanation = generate_explanation(
        adapted=adapted,
        filter_result=filter_result,
        cbr_matches=cbr_matches,
        profile=profile,
        all_recipes=recipes,
    )

    return adapted, explanation
