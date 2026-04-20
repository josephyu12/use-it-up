"""Stage 1: Ingredient matching and rule-based filtering."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from rapidfuzz import fuzz

from useitup.profile import UserProfile
from useitup.schemas import Recipe

FUZZY_THRESHOLD: int = 75
COVERAGE_THRESHOLD: float = 0.5

_NUT_KEYWORDS = frozenset({
    "almond", "cashew", "walnut", "peanut", "pecan", "pistachio",
    "hazelnut", "pine nut", "macadamia", "chestnut", "nut",
})
_GLUTEN_KEYWORDS = frozenset({
    "wheat", "flour", "barley", "rye", "spaghetti", "pasta", "bread",
    "noodle", "breadcrumb", "baguette", "pita", "couscous", "semolina",
    "farro", "bulgur", "spelt",
})
_DAIRY_KEYWORDS = frozenset({
    "milk", "cheese", "butter", "cream", "yogurt", "feta", "ghee",
    "mozzarella", "parmesan", "cheddar", "brie", "ricotta",
})

_CONSTRAINT_KEYWORDS: dict[str, frozenset[str]] = {
    "nut-free": _NUT_KEYWORDS,
    "gluten-free": _GLUTEN_KEYWORDS,
    "dairy-free": _DAIRY_KEYWORDS,
}

# Maps profile goals → recipe dietary_tags
GOAL_TO_TAG: dict[str, str] = {
    "high_protein": "high-protein",
    "low_cost": "low-cost",
    "vegetarian": "vegetarian",
    "vegan": "vegan",
    "low_carb": "low-carb",
    "quick": "quick",
    "dairy_free": "dairy-free",
}


@dataclass
class IngredientScore:
    overlap_count: int
    missing_count: int
    coverage: float
    missing_ingredients: list[str]


@dataclass
class DecisionEntry:
    rule_name: str
    recipe_id: str
    passed: bool
    reason: str


@dataclass
class ScoredRecipe:
    recipe: Recipe
    ingredient_score: IngredientScore
    soft_score: float  # weighted fraction of soft rules satisfied, in [0, 1]


@dataclass
class FilterResult:
    survivors: list[ScoredRecipe]
    decision_log: list[DecisionEntry]


@runtime_checkable
class Rule(Protocol):
    reason: str
    is_hard: bool
    weight: float

    def applies(self, recipe: Recipe, profile: UserProfile) -> bool: ...

    def fail_reason(self, recipe: Recipe, profile: UserProfile) -> str: ...


def _fuzzy_match(a: str, b: str) -> bool:
    return fuzz.partial_ratio(a.lower(), b.lower()) >= FUZZY_THRESHOLD


def _ingredient_contains_keyword(recipe: Recipe, keywords: frozenset[str]) -> bool:
    for ing in recipe.ingredients:
        name = ing.name.lower()
        if any(kw in name for kw in keywords):
            return True
    return False


class IngredientScorer:
    def score(self, recipe: Recipe, pantry: list[str]) -> IngredientScore:
        recipe_ings = [ing.name.lower() for ing in recipe.ingredients]
        matched: set[str] = set()

        for pantry_item in pantry:
            for recipe_ing in recipe_ings:
                if _fuzzy_match(pantry_item, recipe_ing):
                    matched.add(recipe_ing)

        total = len(recipe_ings)
        overlap = len(matched)
        missing = [ing for ing in recipe_ings if ing not in matched]
        coverage = overlap / total if total > 0 else 0.0

        return IngredientScore(
            overlap_count=overlap,
            missing_count=len(missing),
            coverage=coverage,
            missing_ingredients=missing,
        )


class AllergyRule:
    """Hard rule: reject if recipe contains allergen ingredients (nut-free)."""

    reason = "Recipe contains allergens conflicting with allergy constraints"
    is_hard = True
    weight = 1.0

    _ALLERGY_TAGS = frozenset({"nut-free"})

    def applies(self, recipe: Recipe, profile: UserProfile) -> bool:
        active = [c for c in profile.hard_constraints if c in self._ALLERGY_TAGS]
        if not active:
            return True
        recipe_tags = set(recipe.dietary_tags)
        for constraint in active:
            if constraint not in recipe_tags:
                return False
            keywords = _CONSTRAINT_KEYWORDS.get(constraint, frozenset())
            if keywords and _ingredient_contains_keyword(recipe, keywords):
                return False
        return True

    def fail_reason(self, recipe: Recipe, profile: UserProfile) -> str:
        violated = [
            c for c in profile.hard_constraints
            if c in self._ALLERGY_TAGS and c not in set(recipe.dietary_tags)
        ]
        return (
            f"Recipe '{recipe.name}' eliminated: missing allergy-safe tags {violated}"
        )


class DietaryRule:
    """Hard rule: reject if recipe dietary_tags don't cover user's dietary constraints."""

    reason = "Recipe does not satisfy dietary restriction"
    is_hard = True
    weight = 1.0

    _DIETARY_TAGS = frozenset({"vegan", "vegetarian", "gluten-free", "dairy-free"})

    def applies(self, recipe: Recipe, profile: UserProfile) -> bool:
        active = [c for c in profile.hard_constraints if c in self._DIETARY_TAGS]
        if not active:
            return True
        recipe_tags = set(recipe.dietary_tags)
        return all(c in recipe_tags for c in active)

    def fail_reason(self, recipe: Recipe, profile: UserProfile) -> str:
        active = [c for c in profile.hard_constraints if c in self._DIETARY_TAGS]
        missing_tags = [c for c in active if c not in set(recipe.dietary_tags)]
        return (
            f"Recipe '{recipe.name}' eliminated: required tags {missing_tags} absent"
        )


class PantryCoverageRule:
    """Hard rule: reject if ingredient coverage is below the threshold."""

    is_hard = True
    weight = 1.0

    def __init__(self, threshold: float = COVERAGE_THRESHOLD) -> None:
        self.threshold = threshold
        self.reason = f"Ingredient coverage below threshold ({threshold:.0%})"
        self._scorer = IngredientScorer()

    def applies(self, recipe: Recipe, profile: UserProfile) -> bool:
        score = self._scorer.score(recipe, profile.pantry)
        return score.coverage >= self.threshold

    def fail_reason(self, recipe: Recipe, profile: UserProfile) -> str:
        score = self._scorer.score(recipe, profile.pantry)
        return (
            f"Recipe '{recipe.name}' eliminated: coverage {score.coverage:.0%} "
            f"< {self.threshold:.0%}"
        )


class PrepTimeRule:
    """Soft rule: penalize recipes exceeding the user's max prep time."""

    reason = "Recipe prep time exceeds preference"
    is_hard = False
    weight = 0.4

    def applies(self, recipe: Recipe, profile: UserProfile) -> bool:
        max_time = profile.soft_preferences.max_prep_time_min
        if max_time is None:
            return True
        return recipe.prep_time_min <= max_time

    def fail_reason(self, recipe: Recipe, profile: UserProfile) -> str:
        return (
            f"Recipe '{recipe.name}': prep_time {recipe.prep_time_min} min "
            f"> max {profile.soft_preferences.max_prep_time_min} min"
        )


class CuisinePreferenceRule:
    """Soft rule: boost recipes matching the user's preferred cuisines."""

    reason = "Recipe cuisine not in preferred cuisines"
    is_hard = False
    weight = 0.3

    def applies(self, recipe: Recipe, profile: UserProfile) -> bool:
        prefs = profile.soft_preferences.preferred_cuisines
        if not prefs:
            return True
        return recipe.cuisine in prefs

    def fail_reason(self, recipe: Recipe, profile: UserProfile) -> str:
        return (
            f"Recipe '{recipe.name}': cuisine '{recipe.cuisine}' not in "
            f"preferred {profile.soft_preferences.preferred_cuisines}"
        )


class GoalAlignmentRule:
    """Soft rule: boost recipes that satisfy at least one user goal."""

    reason = "Recipe does not align with user goals"
    is_hard = False
    weight = 0.3

    def applies(self, recipe: Recipe, profile: UserProfile) -> bool:
        goals = profile.soft_preferences.goals
        if not goals:
            return True
        recipe_tags = set(recipe.dietary_tags)
        return any(GOAL_TO_TAG.get(g, "") in recipe_tags for g in goals)

    def fail_reason(self, recipe: Recipe, profile: UserProfile) -> str:
        goals = profile.soft_preferences.goals
        mapped = [GOAL_TO_TAG.get(g, g) for g in goals]
        return (
            f"Recipe '{recipe.name}': no goal tags {mapped} found in "
            f"recipe tags {list(recipe.dietary_tags)}"
        )


_DEFAULT_HARD_RULES: list[Rule] = [
    AllergyRule(),
    DietaryRule(),
    PantryCoverageRule(),
]

_DEFAULT_SOFT_RULES: list[Rule] = [
    PrepTimeRule(),
    CuisinePreferenceRule(),
    GoalAlignmentRule(),
]


def _build_reason(rule: Rule, recipe: Recipe, profile: UserProfile, passed: bool) -> str:
    if passed:
        return f"Passed {type(rule).__name__}: {rule.reason}"
    return rule.fail_reason(recipe, profile)


class FilterEngine:
    def __init__(
        self,
        hard_rules: list[Rule] | None = None,
        soft_rules: list[Rule] | None = None,
    ) -> None:
        self._hard = hard_rules if hard_rules is not None else list(_DEFAULT_HARD_RULES)
        self._soft = soft_rules if soft_rules is not None else list(_DEFAULT_SOFT_RULES)
        self._scorer = IngredientScorer()

    def run(self, recipes: list[Recipe], profile: UserProfile) -> FilterResult:
        log: list[DecisionEntry] = []
        survivors: list[Recipe] = []

        for recipe in recipes:
            eliminated = False
            for rule in self._hard:
                passed = rule.applies(recipe, profile)
                log.append(DecisionEntry(
                    rule_name=type(rule).__name__,
                    recipe_id=recipe.id,
                    passed=passed,
                    reason=_build_reason(rule, recipe, profile, passed),
                ))
                if not passed:
                    eliminated = True
                    break
            if not eliminated:
                survivors.append(recipe)

        scored: list[ScoredRecipe] = []
        for recipe in survivors:
            ing_score = self._scorer.score(recipe, profile.pantry)
            soft_score = self._score_soft(recipe, profile, log)
            scored.append(ScoredRecipe(
                recipe=recipe,
                ingredient_score=ing_score,
                soft_score=soft_score,
            ))

        return FilterResult(survivors=scored, decision_log=log)

    def _score_soft(
        self, recipe: Recipe, profile: UserProfile, log: list[DecisionEntry]
    ) -> float:
        total_weight = sum(r.weight for r in self._soft)
        if total_weight == 0.0:
            return 1.0

        weighted_sum = 0.0
        for rule in self._soft:
            passed = rule.applies(recipe, profile)
            log.append(DecisionEntry(
                rule_name=type(rule).__name__,
                recipe_id=recipe.id,
                passed=passed,
                reason=_build_reason(rule, recipe, profile, passed),
            ))
            if passed:
                weighted_sum += rule.weight

        return weighted_sum / total_weight
