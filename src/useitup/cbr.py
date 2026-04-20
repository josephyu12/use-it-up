"""Stage 2: Case-based reasoning (Retrieve–Reuse–Revise–Retain)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from useitup.profile import UserProfile, add_rating
from useitup.schemas import Recipe

_DATA_DIR = Path(__file__).parent.parent.parent / "data"
_SUBSTITUTIONS_PATH = _DATA_DIR / "substitutions.json"

MAX_PREP_TIME: float = 120.0

CUISINES: list[str] = [
    "Italian", "Mexican", "Asian", "Indian",
    "American", "Mediterranean", "French", "Middle Eastern",
]
PROTEINS: list[str] = [
    "chicken", "beef", "pork", "fish", "shrimp",
    "tofu", "lentils", "beans", "eggs", "other",
]
COOKING_METHODS: list[str] = [
    "bake", "fry", "boil", "grill", "steam", "roast", "saute", "raw",
]
FLAVORS: list[str] = [
    "spicy", "savory", "sweet", "sour", "umami", "smoky", "fresh", "rich",
]

_METHOD_KEYWORDS: dict[str, list[str]] = {
    "bake": ["bake", "baked", "baking", "oven"],
    "fry": ["fry", "fried", "frying", "pan-fry", "deep fry", "deep-fry"],
    "boil": ["boil", "boiled", "boiling", "simmer", "simmered"],
    "grill": ["grill", "grilled", "grilling", "barbecue", "bbq", "char"],
    "steam": ["steam", "steamed", "steaming"],
    "roast": ["roast", "roasted", "roasting"],
    "saute": ["sauté", "saute", "sautéed", "stir-fry", "stir fry", "stir-fried"],
    "raw": ["raw", "no-cook", "no cook", "ceviche", "uncooked"],
}

# Goals → ingredient name keywords that conflict
_GOAL_CONFLICTS: dict[str, list[str]] = {
    "vegetarian": [
        "chicken", "beef", "pork", "fish", "shrimp", "lamb",
        "turkey", "bacon", "ham", "meat", "anchovy",
    ],
    "vegan": [
        "chicken", "beef", "pork", "fish", "shrimp", "lamb", "turkey",
        "bacon", "ham", "meat", "anchovy",
        "milk", "cheese", "butter", "cream", "yogurt", "egg", "honey",
        "feta", "ghee", "parmesan", "mozzarella", "cheddar",
    ],
    "dairy_free": [
        "milk", "cheese", "butter", "cream", "yogurt", "feta", "ghee",
        "mozzarella", "parmesan", "cheddar", "brie", "ricotta", "sour cream",
    ],
}

# Feature vector slice boundaries
_CUISINE_SLICE = slice(0, len(CUISINES))
_PROTEIN_SLICE = slice(len(CUISINES), len(CUISINES) + len(PROTEINS))
_METHOD_SLICE = slice(
    len(CUISINES) + len(PROTEINS),
    len(CUISINES) + len(PROTEINS) + len(COOKING_METHODS),
)
_FLAVOR_SLICE = slice(
    len(CUISINES) + len(PROTEINS) + len(COOKING_METHODS),
    len(CUISINES) + len(PROTEINS) + len(COOKING_METHODS) + len(FLAVORS),
)
_DIFF_SLICE = slice(
    len(CUISINES) + len(PROTEINS) + len(COOKING_METHODS) + len(FLAVORS),
    len(CUISINES) + len(PROTEINS) + len(COOKING_METHODS) + len(FLAVORS) + 1,
)
_PREP_SLICE = slice(
    len(CUISINES) + len(PROTEINS) + len(COOKING_METHODS) + len(FLAVORS) + 1,
    len(CUISINES) + len(PROTEINS) + len(COOKING_METHODS) + len(FLAVORS) + 2,
)

FEATURE_DIM: int = len(CUISINES) + len(PROTEINS) + len(COOKING_METHODS) + len(FLAVORS) + 2


@dataclass
class FeatureBreakdown:
    cuisine: float
    protein: float
    cooking_method: float
    flavor: float
    difficulty: float
    prep_time: float


@dataclass
class CBRMatch:
    recipe: Recipe
    similarity_score: float
    nearest_past_recipe: Recipe | None
    similarity_breakdown: FeatureBreakdown
    fallback_reason: str | None = None


@dataclass
class AdaptationEntry:
    original: str
    replacement: str
    reason: str


@dataclass
class AdaptedRecipe:
    recipe: Recipe
    adaptations: list[AdaptationEntry] = field(default_factory=list)


def _group_cosine(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def recipe_feature_vector(recipe: Recipe) -> np.ndarray:
    """Encode a recipe as a fixed-length feature vector."""
    parts: list[np.ndarray] = []

    # Cuisine one-hot
    cuisine_vec = np.zeros(len(CUISINES))
    cuisine_lower = recipe.cuisine.lower()
    for i, c in enumerate(CUISINES):
        if c.lower() in cuisine_lower or cuisine_lower in c.lower():
            cuisine_vec[i] = 1.0
            break
    parts.append(cuisine_vec)

    # Primary protein one-hot (first protein-category ingredient wins)
    protein_vec = np.zeros(len(PROTEINS))
    protein_ings = [ing for ing in recipe.ingredients if ing.category == "protein"]
    if protein_ings:
        ing_name = protein_ings[0].name.lower()
        matched = False
        for i, p in enumerate(PROTEINS[:-1]):  # last entry is "other"
            if p in ing_name:
                protein_vec[i] = 1.0
                matched = True
                break
        if not matched:
            protein_vec[-1] = 1.0
    else:
        protein_vec[-1] = 1.0
    parts.append(protein_vec)

    # Cooking method multi-hot from instructions
    method_vec = np.zeros(len(COOKING_METHODS))
    instructions_text = " ".join(recipe.instructions).lower()
    for i, method in enumerate(COOKING_METHODS):
        keywords = _METHOD_KEYWORDS.get(method, [method])
        if any(kw in instructions_text for kw in keywords):
            method_vec[i] = 1.0
    if method_vec.sum() == 0.0:
        method_vec[COOKING_METHODS.index("raw")] = 1.0
    parts.append(method_vec)

    # Flavor multi-hot
    flavor_vec = np.zeros(len(FLAVORS))
    recipe_flavors = set(recipe.flavor_profile)
    for i, f in enumerate(FLAVORS):
        if f in recipe_flavors:
            flavor_vec[i] = 1.0
    parts.append(flavor_vec)

    # Difficulty normalized to [0, 1]
    parts.append(np.array([(recipe.difficulty - 1) / 4.0]))

    # Prep time normalized to [0, 1]
    parts.append(np.array([min(recipe.prep_time_min / MAX_PREP_TIME, 1.0)]))

    return np.concatenate(parts)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


class CBRRetriever:
    def __init__(self, recipes: list[Recipe], profile: UserProfile) -> None:
        self._profile = profile
        self._recipe_index: dict[str, Recipe] = {r.id: r for r in recipes}

        liked_entries = [e for e in profile.rating_history if e.rating >= 4]
        liked_pairs: list[tuple[Recipe, int]] = [
            (self._recipe_index[e.recipe_id], e.rating)
            for e in liked_entries
            if e.recipe_id in self._recipe_index
        ]

        if liked_pairs:
            vectors = np.array([recipe_feature_vector(r) for r, _ in liked_pairs])
            weights = np.array([float(rating) for _, rating in liked_pairs])
            self._centroid: np.ndarray = np.average(vectors, axis=0, weights=weights)
            self._liked: list[tuple[Recipe, np.ndarray]] = [
                (r, recipe_feature_vector(r)) for r, _ in liked_pairs
            ]
            self._cold_start = False
        else:
            self._centroid = np.zeros(FEATURE_DIM)
            self._liked = []
            self._cold_start = True

    def retrieve(self, candidates: list[Recipe], k: int = 5) -> list[CBRMatch]:
        if self._cold_start:
            return self._cold_start_retrieve(candidates, k)

        results: list[CBRMatch] = []
        for recipe in candidates:
            vec = recipe_feature_vector(recipe)
            sim = _cosine_similarity(vec, self._centroid)

            nearest: Recipe | None = None
            nearest_sim = -1.0
            for past_recipe, past_vec in self._liked:
                s = _cosine_similarity(vec, past_vec)
                if s > nearest_sim:
                    nearest_sim = s
                    nearest = past_recipe

            breakdown = FeatureBreakdown(
                cuisine=_group_cosine(vec[_CUISINE_SLICE], self._centroid[_CUISINE_SLICE]),
                protein=_group_cosine(vec[_PROTEIN_SLICE], self._centroid[_PROTEIN_SLICE]),
                cooking_method=_group_cosine(vec[_METHOD_SLICE], self._centroid[_METHOD_SLICE]),
                flavor=_group_cosine(vec[_FLAVOR_SLICE], self._centroid[_FLAVOR_SLICE]),
                difficulty=_group_cosine(vec[_DIFF_SLICE], self._centroid[_DIFF_SLICE]),
                prep_time=_group_cosine(vec[_PREP_SLICE], self._centroid[_PREP_SLICE]),
            )

            results.append(CBRMatch(
                recipe=recipe,
                similarity_score=sim,
                nearest_past_recipe=nearest,
                similarity_breakdown=breakdown,
            ))

        results.sort(key=lambda m: m.similarity_score, reverse=True)
        return results[:k]

    def _cold_start_retrieve(self, candidates: list[Recipe], k: int) -> list[CBRMatch]:
        preferred = set(self._profile.soft_preferences.preferred_cuisines)

        def _rank(r: Recipe) -> tuple[float, float]:
            cuisine_score = 1.0 if r.cuisine in preferred else 0.0
            time_score = 1.0 - min(r.prep_time_min / MAX_PREP_TIME, 1.0)
            return (cuisine_score, time_score)

        sorted_candidates = sorted(candidates, key=_rank, reverse=True)
        fallback_msg = (
            "No rating history; ranked by preferred cuisines and prep time"
            if preferred
            else "No rating history or cuisine preferences; ranked by prep time"
        )

        zero_breakdown = FeatureBreakdown(
            cuisine=0.0, protein=0.0, cooking_method=0.0,
            flavor=0.0, difficulty=0.0, prep_time=0.0,
        )
        return [
            CBRMatch(
                recipe=recipe,
                similarity_score=0.0,
                nearest_past_recipe=None,
                similarity_breakdown=zero_breakdown,
                fallback_reason=fallback_msg,
            )
            for recipe in sorted_candidates[:k]
        ]


class CBRAdapter:
    def __init__(self, substitutions_path: Path | None = None) -> None:
        path = substitutions_path if substitutions_path is not None else _SUBSTITUTIONS_PATH
        self._substitutions: list[dict] = json.loads(path.read_text(encoding="utf-8"))

    def adapt(self, match: CBRMatch, profile: UserProfile) -> AdaptedRecipe:
        goals = set(profile.soft_preferences.goals)
        new_ingredients = list(match.recipe.ingredients)
        adaptations: list[AdaptationEntry] = []

        for i, ingredient in enumerate(match.recipe.ingredients):
            ing_lower = ingredient.name.lower()
            for goal in goals:
                conflicts = _GOAL_CONFLICTS.get(goal, [])
                if not any(kw in ing_lower for kw in conflicts):
                    continue
                sub = self._find_substitution(ing_lower, goal)
                if sub is None:
                    continue
                new_ingredients[i] = ingredient.model_copy(
                    update={"name": sub["replacement"]}
                )
                adaptations.append(AdaptationEntry(
                    original=ingredient.name,
                    replacement=sub["replacement"],
                    reason=sub["reason"],
                ))
                break  # one substitution per ingredient

        adapted_recipe = (
            match.recipe.model_copy(update={"ingredients": new_ingredients})
            if adaptations
            else match.recipe
        )
        return AdaptedRecipe(recipe=adapted_recipe, adaptations=adaptations)

    def _find_substitution(self, ing_name: str, goal: str) -> dict | None:
        for sub in self._substitutions:
            original_kw = sub.get("original", "").lower()
            if original_kw and original_kw in ing_name:
                sub_goals: list[str] = sub.get("goals", [])
                if not sub_goals or goal in sub_goals:
                    return sub
        return None


def record_success(profile: UserProfile, recipe_id: str, rating: int = 5) -> UserProfile:
    return add_rating(profile, recipe_id, rating)
