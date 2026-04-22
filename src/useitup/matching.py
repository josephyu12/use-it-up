"""Stage 1: Ingredient matching and rule-based filtering."""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Protocol, runtime_checkable

from rapidfuzz import fuzz

from useitup.profile import UserProfile
from useitup.schemas import Ingredient, Recipe

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

_PLANT_BASED_DAIRY_PREFIXES = ("coconut ", "oat ", "vegan ", "cashew ", "almond ", "soy ")

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
    matched_ingredients: list[str]
    essential_overlap_count: int = 0
    essential_total: int = 0
    essential_coverage: float = 0.0
    essential_missing_ingredients: list[str] | None = None
    weighted_coverage: float = 0.0


_STOPWORDS = frozenset({
    "fresh", "dried", "ground", "crushed", "chopped", "minced", "large", "small",
    "extra", "virgin", "boneless", "skinless", "whole", "plain", "low", "fat",
    "optional", "for", "serving",
})
_GARNISH_KEYWORDS = frozenset({
    "salt", "pepper", "parsley", "cilantro", "chives", "scallion", "green onion",
    "sesame seeds", "red pepper flakes", "lemon wedges", "lime wedges",
})
_TOKEN_ALIASES: dict[str, str] = {
    "scallions": "green onion",
    "scallion": "green onion",
    "spring onion": "green onion",
    "garbanzo": "chickpea",
    "garbanzos": "chickpea",
    "chickpeas": "chickpea",
    "tomatoes": "tomato",
    "eggs": "egg",
    "tortillas": "tortilla",
    "noodles": "noodle",
}


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
    return _ingredient_match(a, b)


@lru_cache(maxsize=None)
def _normalize_token(token: str) -> str:
    token = token.strip().lower()
    token = re.sub(r"[^a-z0-9]+", "", token)
    if not token:
        return ""
    if token in _TOKEN_ALIASES:
        return _TOKEN_ALIASES[token]
    if len(token) > 4 and token.endswith("ies"):
        token = token[:-3] + "y"
    elif len(token) > 3 and token.endswith("es"):
        token = token[:-2]
    elif len(token) > 3 and token.endswith("s"):
        token = token[:-1]
    return _TOKEN_ALIASES.get(token, token)


@lru_cache(maxsize=None)
def _ingredient_tokens(text: str) -> frozenset[str]:
    tokens = {
        _normalize_token(token)
        for token in re.split(r"[^a-z0-9]+", text.lower())
    }
    return frozenset(token for token in tokens if token and token not in _STOPWORDS)


def _ingredient_match(a: str, b: str) -> bool:
    a_tokens = _ingredient_tokens(a)
    b_tokens = _ingredient_tokens(b)
    if not a_tokens or not b_tokens:
        return False
    if a_tokens == b_tokens or a_tokens.issubset(b_tokens) or b_tokens.issubset(a_tokens):
        return True
    if a_tokens & b_tokens:
        return True

    normalized_a = " ".join(sorted(a_tokens))
    normalized_b = " ".join(sorted(b_tokens))
    return (
        fuzz.token_set_ratio(normalized_a, normalized_b) >= 92
        or fuzz.ratio(normalized_a, normalized_b) >= 88
    )


def _ingredient_contains_keyword(recipe: Recipe, keywords: frozenset[str]) -> bool:
    for ing in recipe.ingredients:
        name = ing.name.lower()
        if _name_matches_constraint_keywords(name, keywords):
            return True
    return False


def _name_matches_constraint_keywords(name: str, keywords: frozenset[str]) -> bool:
    if keywords is _NUT_KEYWORDS and "coconut" in name:
        return False
    if keywords is _DAIRY_KEYWORDS and any(name.startswith(prefix) for prefix in _PLANT_BASED_DAIRY_PREFIXES):
        return False
    return any(kw in name for kw in keywords)


def _is_garnish(name: str) -> bool:
    low = name.lower()
    return any(keyword in low for keyword in _GARNISH_KEYWORDS)


class IngredientScorer:
    def __init__(self) -> None:
        # Scorer is created fresh per request; cache memoizes repeat calls
        # within one request (e.g. Rule.applies → Rule.fail_reason re-scoring
        # the same recipe under the same pantry).
        self._cache: dict[tuple[str, tuple[str, ...], tuple[str, ...]], IngredientScore] = {}

    def score(self, recipe: Recipe, pantry: list[str]) -> IngredientScore:
        ingredient_signature = tuple(ingredient.name.lower() for ingredient in recipe.ingredients)
        cache_key = (recipe.id, ingredient_signature, tuple(pantry))
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached
        recipe_name_tokens = _ingredient_tokens(recipe.name)
        recipe_ings = recipe.ingredients
        matched: list[str] = []
        essential: list[str] = []
        essential_missing: list[str] = []
        total_weight = 0.0
        matched_weight = 0.0

        for idx, ingredient in enumerate(recipe_ings):
            ing_name = ingredient.name.lower()
            weight = self._ingredient_weight(ingredient, idx, recipe_name_tokens)
            total_weight += weight
            is_match = any(_fuzzy_match(pantry_item, ing_name) for pantry_item in pantry)
            if is_match:
                matched.append(ing_name)
                matched_weight += weight
            if self._is_essential(ingredient, idx, recipe_name_tokens):
                essential.append(ing_name)
                if not is_match:
                    essential_missing.append(ing_name)

        total = len(recipe_ings)
        overlap = len(matched)
        missing = [ing.name.lower() for ing in recipe_ings if ing.name.lower() not in matched]
        essential_overlap = len(essential) - len(essential_missing)
        essential_total = len(essential)
        coverage = overlap / total if total > 0 else 0.0
        essential_coverage = essential_overlap / essential_total if essential_total > 0 else coverage
        weighted_coverage = matched_weight / total_weight if total_weight > 0 else 0.0

        result = IngredientScore(
            overlap_count=overlap,
            missing_count=len(missing),
            coverage=coverage,
            missing_ingredients=missing,
            matched_ingredients=matched,
            essential_overlap_count=essential_overlap,
            essential_total=essential_total,
            essential_coverage=essential_coverage,
            essential_missing_ingredients=essential_missing,
            weighted_coverage=weighted_coverage,
        )
        self._cache[cache_key] = result
        return result

    def _ingredient_weight(
        self,
        ingredient: Ingredient,
        idx: int,
        recipe_name_tokens: set[str],
    ) -> float:
        category_weights = {
            "protein": 1.0,
            "grain": 1.0,
            "vegetable": 0.75,
            "dairy": 0.7,
            "fat": 0.65,
            "condiment": 0.45,
            "spice": 0.25,
            "other": 0.55,
        }
        weight = category_weights.get(ingredient.category, 0.5)
        ing_tokens = _ingredient_tokens(ingredient.name)
        if idx < 2 and ingredient.category in {"protein", "grain", "vegetable", "dairy", "fat"}:
            weight += 0.2
        if idx < 3 and ingredient.category in {"fat", "dairy"}:
            weight += 0.15
        if ing_tokens & recipe_name_tokens:
            weight += 0.35
        if _is_garnish(ingredient.name):
            weight = min(weight, 0.2)
        return weight

    def _is_essential(
        self,
        ingredient: Ingredient,
        idx: int,
        recipe_name_tokens: set[str],
    ) -> bool:
        if ingredient.is_core is not None:
            return ingredient.is_core
        ing_name = ingredient.name.lower()
        ing_tokens = _ingredient_tokens(ing_name)
        if _is_garnish(ing_name):
            return False
        if ing_tokens & recipe_name_tokens:
            return True
        if ingredient.category in {"protein", "grain"}:
            return True
        if idx < 2 and ingredient.category in {"protein", "grain", "vegetable", "dairy", "fat"}:
            return True
        return idx < 3 and ingredient.category in {"fat", "dairy", "vegetable"}


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
    """Hard rule: reject if coverage is too low or any core ingredient is missing."""

    is_hard = True
    weight = 1.0

    def __init__(
        self,
        threshold: float = COVERAGE_THRESHOLD,
        require_all_essentials: bool | None = None,
    ) -> None:
        self.threshold = threshold
        # Auto-relax essential check at low thresholds (sparse pantry / adaptation).
        if require_all_essentials is None:
            self.require_all_essentials = threshold > 0.35
        else:
            self.require_all_essentials = require_all_essentials
        self.reason = f"Ingredient coverage below threshold ({threshold:.0%})"
        self._scorer = IngredientScorer()

    def applies(self, recipe: Recipe, profile: UserProfile) -> bool:
        if self.threshold <= 0.0:
            return True
        score = self._scorer.score(recipe, profile.pantry)
        passes_raw = score.coverage >= self.threshold
        if not passes_raw:
            return False
        if self.require_all_essentials:
            return not (score.essential_missing_ingredients or [])
        return True

    def fail_reason(self, recipe: Recipe, profile: UserProfile) -> str:
        if self.threshold <= 0.0:
            return f"Passed PantryCoverageRule: {self.reason}"
        score = self._scorer.score(recipe, profile.pantry)
        missing_essential = score.essential_missing_ingredients or []
        return (
            f"Recipe '{recipe.name}' eliminated: raw coverage {score.coverage:.0%}, "
            f"weighted coverage {score.weighted_coverage:.0%}, "
            f"essential coverage {score.essential_coverage:.0%} "
            f"(need {self.threshold:.0%} raw"
            + (" and all core ingredients present" if self.require_all_essentials else "")
            + f"); missing essentials: {missing_essential or ['none']}"
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
