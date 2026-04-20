"""Stage 3: Explanation generation (goal trace, counterfactual, CBR trace, utilization report)."""

from __future__ import annotations

from dataclasses import dataclass

from rapidfuzz import fuzz

from useitup.cbr import AdaptedRecipe, CBRMatch
from useitup.matching import (
    CuisinePreferenceRule,
    DecisionEntry,
    FilterResult,
    GOAL_TO_TAG,
    GoalAlignmentRule,
    IngredientScorer,
    PrepTimeRule,
    ScoredRecipe,
)
from useitup.profile import UserProfile
from useitup.schemas import Recipe

_HARD_RULE_NAMES = frozenset({"AllergyRule", "DietaryRule", "PantryCoverageRule"})
_FUZZY_THRESHOLD: int = 75


def _fuzz(a: str, b: str) -> bool:
    return fuzz.partial_ratio(a.lower(), b.lower()) >= _FUZZY_THRESHOLD


def _pantry_used(recipe: Recipe, pantry: list[str]) -> list[str]:
    """Pantry items that fuzzy-match at least one recipe ingredient."""
    ing_names = [i.name.lower() for i in recipe.ingredients]
    return [item for item in pantry if any(_fuzz(item, n) for n in ing_names)]


def _pantry_unused(recipe: Recipe, pantry: list[str]) -> list[str]:
    """Pantry items that match NO recipe ingredient."""
    ing_names = [i.name.lower() for i in recipe.ingredients]
    return [item for item in pantry if not any(_fuzz(item, n) for n in ing_names)]


def _soft_score(recipe: Recipe, profile: UserProfile) -> float:
    """Compute hypothetical soft score for a recipe (no hard-rule filtering)."""
    rules = [PrepTimeRule(), CuisinePreferenceRule(), GoalAlignmentRule()]
    total = sum(r.weight for r in rules)
    earned = sum(r.weight for r in rules if r.applies(recipe, profile))
    return earned / total if total > 0.0 else 1.0


@dataclass
class Explanation:
    goal_trace: str
    counterfactual: str
    cbr_trace: str
    ingredient_utilization_report: str


def _build_goal_trace(
    adapted: AdaptedRecipe,
    decision_log: list[DecisionEntry],
    profile: UserProfile,
) -> str:
    recipe = adapted.recipe
    scorer = IngredientScorer()
    ing_score = scorer.score(recipe, profile.pantry)
    total = ing_score.overlap_count + ing_score.missing_count
    pct = int(ing_score.coverage * 100)
    used = _pantry_used(recipe, profile.pantry)

    recipe_tags = set(recipe.dietary_tags)
    matched_goals = [g for g in profile.soft_preferences.goals if GOAL_TO_TAG.get(g, "") in recipe_tags]

    # Pull soft rule outcomes for this recipe from the decision log
    recipe_log = [e for e in decision_log if e.recipe_id == recipe.id]
    soft_failed = [e for e in recipe_log if not e.passed and e.rule_name not in _HARD_RULE_NAMES]

    lines = [f"## Goal Trace: Why **{recipe.name}**?", ""]
    lines.append(
        f"I recommend **{recipe.name}** because you have "
        f"**{ing_score.overlap_count} of {total} ingredients** ({pct}%) on hand."
    )
    if used:
        lines.append(f"\n**Pantry items used:** {', '.join(used)}.")
    if matched_goals:
        goal_labels = [g.replace("_", " ") for g in matched_goals]
        lines.append(f"\n**Goals satisfied:** {', '.join(goal_labels)}.")
    else:
        lines.append("\n*No explicit goal tags matched — hard constraints are still respected.*")
    if profile.hard_constraints:
        lines.append(f"\n**Hard constraints respected:** {', '.join(profile.hard_constraints)}.")
    if adapted.adaptations:
        lines.append("\n**Adaptations made for your goals:**")
        for a in adapted.adaptations:
            lines.append(f"- Replaced **{a.original}** → **{a.replacement}**: {a.reason}")
    if soft_failed:
        names = ", ".join(e.rule_name for e in soft_failed)
        lines.append(
            f"\n*Note: {len(soft_failed)} soft preference(s) not fully met ({names}), "
            "but this was the best available match.*"
        )
    return "\n".join(lines)


def _build_counterfactual(
    adapted: AdaptedRecipe,
    decision_log: list[DecisionEntry],
    filter_result: FilterResult,
    all_recipes: list[Recipe],
    profile: UserProfile,
) -> str:
    # Collect recipe_ids that failed at least one hard rule
    recipe_index: dict[str, Recipe] = {r.id: r for r in all_recipes}
    survivor_ids = {sr.recipe.id for sr in filter_result.survivors}

    hard_fail_reasons: dict[str, str] = {}
    for entry in decision_log:
        if not entry.passed and entry.rule_name in _HARD_RULE_NAMES:
            if entry.recipe_id not in hard_fail_reasons:
                hard_fail_reasons[entry.recipe_id] = entry.reason

    truly_rejected = {rid: reason for rid, reason in hard_fail_reasons.items() if rid not in survivor_ids}

    if not truly_rejected:
        return (
            "## Counterfactual\n\n"
            "All candidate recipes passed hard constraints. "
            "The top recommendation was chosen based on the highest combined "
            "ingredient coverage and soft preference score."
        )

    # Rank rejected recipes by ingredient coverage, then by hypothetical soft score
    scorer = IngredientScorer()
    best_recipe: Recipe | None = None
    best_coverage = -1.0
    best_reason = ""

    for rid, reason in truly_rejected.items():
        recipe = recipe_index.get(rid)
        if recipe is None:
            continue
        score = scorer.score(recipe, profile.pantry)
        if score.coverage > best_coverage:
            best_coverage = score.coverage
            best_recipe = recipe
            best_reason = reason

    if best_recipe is None:
        return (
            "## Counterfactual\n\n"
            "Could not resolve any hard-rejected recipe from the candidate pool."
        )

    # Determine which constraint(s) blocked it
    violated = [c for c in profile.hard_constraints if c not in set(best_recipe.dietary_tags)]

    # Hypothetical position: where would it rank among survivors by soft score?
    hyp_score = _soft_score(best_recipe, profile)
    survivors_sorted = sorted(filter_result.survivors, key=lambda s: s.soft_score, reverse=True)
    position = sum(1 for s in survivors_sorted if s.soft_score >= hyp_score) + 1

    # Identify its strongest soft dimension
    recipe_tags = set(best_recipe.dietary_tags)
    strong_tags = [t for t in ("high-protein", "low-cost", "quick") if t in recipe_tags]
    top_soft_label = strong_tags[0] if strong_tags else "ingredient coverage"

    lines = [
        "## Counterfactual",
        "",
        f"I did not recommend **{best_recipe.name}** because: _{best_reason}_.",
        "",
    ]
    if violated:
        lines.append(
            f"If you removed your **{', '.join(violated)}** constraint(s), "
            f"it would have ranked **#{position}** in the candidate pool "
            f"due to its **{top_soft_label}** profile."
        )
    return "\n".join(lines)


def _build_cbr_trace(
    adapted: AdaptedRecipe,
    cbr_matches: list[CBRMatch],
    profile: UserProfile,
) -> str:
    if not cbr_matches:
        return "## CBR Trace\n\nNo candidates were available for case-based reasoning."

    best = cbr_matches[0]
    recipe = adapted.recipe

    if best.fallback_reason is not None:
        lines = [
            "## CBR Trace",
            "",
            f"**Cold-start mode:** {best.fallback_reason}.",
            "",
            f"**{recipe.name}** was selected as the top candidate using cuisine preference "
            "and prep-time ranking, because no past recipe ratings are available "
            "to guide similarity matching.",
        ]
        if adapted.adaptations:
            lines.append("\n**Adaptations applied:**")
            for a in adapted.adaptations:
                lines.append(f"- {a.original} → {a.replacement}: {a.reason}")
        return "\n".join(lines)

    past = best.nearest_past_recipe
    if past is None:
        return (
            "## CBR Trace\n\n"
            f"**{recipe.name}** was retrieved via CBR "
            f"(similarity: {best.similarity_score:.2f}) but no nearest past recipe was identified."
        )

    # Look up rating and date from profile history
    rating_entry = next(
        (e for e in reversed(profile.rating_history) if e.recipe_id == past.id), None
    )
    rating_str = f"{rating_entry.rating}/5" if rating_entry else "unknown rating"
    date_str = rating_entry.timestamp[:10] if rating_entry else "unknown date"

    # Identify matched feature dimensions (similarity ≥ 0.7)
    bd = best.similarity_breakdown
    feature_scores = {
        "cuisine": bd.cuisine,
        "protein": bd.protein,
        "cooking method": bd.cooking_method,
        "flavor profile": bd.flavor,
        "difficulty": bd.difficulty,
        "prep time": bd.prep_time,
    }
    matching_features = [k for k, v in feature_scores.items() if v >= 0.7]

    lines = [
        "## CBR Trace",
        "",
        f"**{recipe.name}** is based on **{past.name}**, "
        f"which you rated **{rating_str}** on {date_str}.",
        f"\n**CBR similarity score:** {best.similarity_score:.2f} "
        f"(cuisine: {bd.cuisine:.2f}, flavor: {bd.flavor:.2f}, protein: {bd.protein:.2f})",
        "",
    ]
    if matching_features:
        lines.append(f"**Features kept from past recipe:** {', '.join(matching_features)}.")
    if adapted.adaptations:
        lines.append("\n**Adaptations made:**")
        for a in adapted.adaptations:
            lines.append(f"- Replaced **{a.original}** with **{a.replacement}**: {a.reason}")
    else:
        lines.append("\n*No ingredient adaptations were needed.*")
    return "\n".join(lines)


def _build_ingredient_utilization_report(
    adapted: AdaptedRecipe,
    filter_result: FilterResult,
    profile: UserProfile,
) -> str:
    recipe = adapted.recipe
    used = _pantry_used(recipe, profile.pantry)
    unused = _pantry_unused(recipe, profile.pantry)

    scorer = IngredientScorer()
    ing_score = scorer.score(recipe, profile.pantry)
    missing = ing_score.missing_ingredients

    # Suggest follow-up recipes from survivors that use still-unused pantry items
    follow_ups: list[tuple[Recipe, list[str]]] = []
    for scored in filter_result.survivors:
        if scored.recipe.id == recipe.id:
            continue
        consumed = _pantry_used(scored.recipe, unused)
        if consumed:
            follow_ups.append((scored.recipe, consumed))
    follow_ups.sort(key=lambda x: len(x[1]), reverse=True)
    top_follow_ups = follow_ups[:2]

    lines = ["## Ingredient Utilization Report", ""]
    if used:
        lines.append("### ✅ Pantry Ingredients Used")
        for item in used:
            lines.append(f"- {item}")
        lines.append("")
    if unused:
        lines.append("### ⚠️ Pantry Ingredients NOT Used")
        for item in unused:
            lines.append(f"- {item}")
        lines.append("")
    if missing:
        lines.append("### 🛒 Missing Ingredients to Buy")
        for item in missing:
            lines.append(f"- {item}")
        lines.append("")
    lines.append("### 💡 Follow-Up Recipe Suggestions")
    if top_follow_ups:
        for r, items in top_follow_ups:
            lines.append(f"- **{r.name}** — would also use: {', '.join(items)}")
    else:
        lines.append("No other candidates found that would further reduce pantry waste.")
    return "\n".join(lines)


def generate_explanation(
    adapted: AdaptedRecipe,
    filter_result: FilterResult,
    cbr_matches: list[CBRMatch],
    profile: UserProfile,
    all_recipes: list[Recipe],
) -> Explanation:
    return Explanation(
        goal_trace=_build_goal_trace(adapted, filter_result.decision_log, profile),
        counterfactual=_build_counterfactual(
            adapted, filter_result.decision_log, filter_result, all_recipes, profile
        ),
        cbr_trace=_build_cbr_trace(adapted, cbr_matches, profile),
        ingredient_utilization_report=_build_ingredient_utilization_report(
            adapted, filter_result, profile
        ),
    )


def render_explanation(explanation: Explanation) -> str:
    """Concatenate all four explanation sections into a single markdown document."""
    parts = [
        explanation.goal_trace,
        explanation.counterfactual,
        explanation.cbr_trace,
        explanation.ingredient_utilization_report,
    ]
    return "\n\n---\n\n".join(parts)
