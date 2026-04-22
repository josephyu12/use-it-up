"""FastAPI web UI for UseItUp. Wraps the 3-stage pipeline behind a REST API."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict
import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from useitup.data_loader import load_recipes
from useitup.explain import render_explanation
from useitup.matching import GOAL_TO_TAG, IngredientScorer, _ingredient_tokens
from useitup.pipeline import recommend
from useitup.profile import SoftPreferences, UserProfile, load_profile

_PKG_DIR = Path(__file__).parent
_STATIC_DIR = _PKG_DIR / "static"
_DATA_DIR = _PKG_DIR.parent.parent / "data"
_RECIPES_FULL = _DATA_DIR / "recipes.json"
_RECIPES_SAMPLE = _DATA_DIR / "recipes_sample.json"
_RECIPES_CURATED = _DATA_DIR / "recipes_curated.json"


def _default_recipes_path() -> Path:
    override = os.getenv("USEITUP_RECIPES_PATH")
    if override:
        return Path(override)
    if os.getenv("VERCEL") and _RECIPES_CURATED.exists():
        return _RECIPES_CURATED
    if _RECIPES_FULL.exists():
        return _RECIPES_FULL
    if _RECIPES_CURATED.exists():
        return _RECIPES_CURATED
    return _RECIPES_SAMPLE


_RECIPES_PATH = _default_recipes_path()

_CUISINE_EMOJI: dict[str, str] = {
    "Italian": "\U0001F1EE\U0001F1F9",
    "Mexican": "\U0001F1F2\U0001F1FD",
    "Asian": "\U0001F9C2",
    "Indian": "\U0001F1EE\U0001F1F3",
    "American": "\U0001F1FA\U0001F1F8",
    "Mediterranean": "\U0001F1EC\U0001F1F7",
    "French": "\U0001F1EB\U0001F1F7",
    "Middle Eastern": "\U0001F9C6",
}

# Keyword → Unsplash photo. Matched against lowercased recipe name (first hit wins).
# Ordered list so more-specific keywords beat generic ones.
_KEYWORD_IMAGES: list[tuple[str, str]] = [
    ("carbonara", "https://images.unsplash.com/photo-1612874742237-6526221588e3?w=900&q=80"),
    ("alfredo", "https://images.unsplash.com/photo-1473093295043-cdd812d0e601?w=900&q=80"),
    ("arrabbiata", "https://images.unsplash.com/photo-1551892374-ecf8754cf8b0?w=900&q=80"),
    ("cacio", "https://images.unsplash.com/photo-1627042633145-b780d842ba0a?w=900&q=80"),
    ("lasagna", "https://images.unsplash.com/photo-1619895092538-128341789043?w=900&q=80"),
    ("pizza", "https://images.unsplash.com/photo-1604068549290-dea0e4a305ca?w=900&q=80"),
    ("caprese", "https://images.unsplash.com/photo-1608897013039-887f21d8c804?w=900&q=80"),
    ("risotto", "https://images.unsplash.com/photo-1476124369491-e7addf5db371?w=900&q=80"),
    ("minestrone", "https://images.unsplash.com/photo-1547592180-85f173990554?w=900&q=80"),
    ("gnocchi", "https://images.unsplash.com/photo-1587740908075-9e245070dfaa?w=900&q=80"),
    ("parmesan", "https://images.unsplash.com/photo-1625944525533-473f1a3d54e7?w=900&q=80"),
    ("aglio", "https://images.unsplash.com/photo-1556761223-4c4282c73f77?w=900&q=80"),
    ("spaghetti", "https://images.unsplash.com/photo-1556761223-4c4282c73f77?w=900&q=80"),
    ("pasta", "https://images.unsplash.com/photo-1473093295043-cdd812d0e601?w=900&q=80"),

    ("taco", "https://images.unsplash.com/photo-1565299585323-38d6b0865b47?w=900&q=80"),
    ("burrito", "https://images.unsplash.com/photo-1626700051175-6818013e1d4f?w=900&q=80"),
    ("quesadilla", "https://images.unsplash.com/photo-1618040996337-11b2b2b5ccef?w=900&q=80"),
    ("enchilada", "https://images.unsplash.com/photo-1534352956036-cd81e27dd615?w=900&q=80"),
    ("fajita", "https://images.unsplash.com/photo-1534352956036-cd81e27dd615?w=900&q=80"),
    ("guacamole", "https://images.unsplash.com/photo-1600335895229-6e75511892c8?w=900&q=80"),
    ("huevos", "https://images.unsplash.com/photo-1590412200988-a436970781fa?w=900&q=80"),
    ("pozole", "https://images.unsplash.com/photo-1584278860047-22db9ff82bed?w=900&q=80"),
    ("tortilla", "https://images.unsplash.com/photo-1604467794349-0b74285de7e7?w=900&q=80"),
    ("carne asada", "https://images.unsplash.com/photo-1551504734-5ee1c4a1479b?w=900&q=80"),

    ("ramen", "https://images.unsplash.com/photo-1617421753170-46511a8d73fc?w=900&q=80"),
    ("pad thai", "https://images.unsplash.com/photo-1559314809-0d155014e29e?w=900&q=80"),
    ("fried rice", "https://images.unsplash.com/photo-1603133872878-684f208fb84b?w=900&q=80"),
    ("stir fry", "https://images.unsplash.com/photo-1603133872878-684f208fb84b?w=900&q=80"),
    ("teriyaki", "https://images.unsplash.com/photo-1546069901-ba9599a7e63c?w=900&q=80"),
    ("pho", "https://images.unsplash.com/photo-1576577445504-6af96477db52?w=900&q=80"),
    ("kung pao", "https://images.unsplash.com/photo-1525755662778-989d0524087e?w=900&q=80"),
    ("sesame noodle", "https://images.unsplash.com/photo-1526318896980-cf78c088247c?w=900&q=80"),
    ("kimchi", "https://images.unsplash.com/photo-1590301157890-4810ed352733?w=900&q=80"),
    ("thai green curry", "https://images.unsplash.com/photo-1455619452474-d2be8b1e70cd?w=900&q=80"),
    ("bulgogi", "https://images.unsplash.com/photo-1498654896293-37aacf113fd9?w=900&q=80"),
    ("egg drop", "https://images.unsplash.com/photo-1547592180-85f173990554?w=900&q=80"),

    ("tikka", "https://images.unsplash.com/photo-1588166524941-3bf61a9c41db?w=900&q=80"),
    ("dal", "https://images.unsplash.com/photo-1585937421612-70a008356fbe?w=900&q=80"),
    ("palak", "https://images.unsplash.com/photo-1601050690597-df0568f70950?w=900&q=80"),
    ("chana", "https://images.unsplash.com/photo-1631292784640-2b24be784d5d?w=900&q=80"),
    ("biryani", "https://images.unsplash.com/photo-1563379091339-03b21ab4a4f8?w=900&q=80"),
    ("butter chicken", "https://images.unsplash.com/photo-1603894584373-5ac82b2ae398?w=900&q=80"),
    ("aloo gobi", "https://images.unsplash.com/photo-1589301760014-d929f3979dbc?w=900&q=80"),
    ("tandoori", "https://images.unsplash.com/photo-1599487488170-d11ec9c172f0?w=900&q=80"),
    ("saag", "https://images.unsplash.com/photo-1601050690597-df0568f70950?w=900&q=80"),
    ("curry", "https://images.unsplash.com/photo-1455619452474-d2be8b1e70cd?w=900&q=80"),

    ("pulled pork", "https://images.unsplash.com/photo-1606755962773-d324e0a13086?w=900&q=80"),
    ("cheeseburger", "https://images.unsplash.com/photo-1568901346375-23c9450c58cd?w=900&q=80"),
    ("fried chicken", "https://images.unsplash.com/photo-1626645738196-c2a7c87a8f58?w=900&q=80"),
    ("mac and cheese", "https://images.unsplash.com/photo-1543352634-a1c51d9f1fa7?w=900&q=80"),
    ("chili", "https://images.unsplash.com/photo-1455619452474-d2be8b1e70cd?w=900&q=80"),
    ("cobb", "https://images.unsplash.com/photo-1546069901-ba9599a7e63c?w=900&q=80"),
    ("grilled cheese", "https://images.unsplash.com/photo-1528735602780-2552fd46c7af?w=900&q=80"),
    ("buffalo wing", "https://images.unsplash.com/photo-1527477396000-e27163b481c2?w=900&q=80"),
    ("meatloaf", "https://images.unsplash.com/photo-1544025162-d76694265947?w=900&q=80"),
    ("pancake", "https://images.unsplash.com/photo-1528207776546-365bb710ee93?w=900&q=80"),

    ("greek salad", "https://images.unsplash.com/photo-1540189549336-e6e99c3679fe?w=900&q=80"),
    ("hummus", "https://images.unsplash.com/photo-1571197119282-7c4edabd7aba?w=900&q=80"),
    ("falafel", "https://images.unsplash.com/photo-1593560704563-f176a2eb61db?w=900&q=80"),
    ("tabbouleh", "https://images.unsplash.com/photo-1505253716362-afaea1d3d1af?w=900&q=80"),
    ("chickpea salad", "https://images.unsplash.com/photo-1505253213348-cd54c92b37fe?w=900&q=80"),
    ("spanakopita", "https://images.unsplash.com/photo-1625944525533-473f1a3d54e7?w=900&q=80"),
    ("souvlaki", "https://images.unsplash.com/photo-1544025162-d76694265947?w=900&q=80"),
    ("baba", "https://images.unsplash.com/photo-1571197119282-7c4edabd7aba?w=900&q=80"),

    ("omelette", "https://images.unsplash.com/photo-1510693206972-df098062cb71?w=900&q=80"),
    ("quiche", "https://images.unsplash.com/photo-1565788969829-948b9a4d1031?w=900&q=80"),
    ("onion soup", "https://images.unsplash.com/photo-1547592180-85f173990554?w=900&q=80"),
    ("ratatouille", "https://images.unsplash.com/photo-1572453800999-e8d2d1589b7c?w=900&q=80"),
    ("croque", "https://images.unsplash.com/photo-1528735602780-2552fd46c7af?w=900&q=80"),
    ("niçoise", "https://images.unsplash.com/photo-1551248429-40975aa4de74?w=900&q=80"),
    ("nicoise", "https://images.unsplash.com/photo-1551248429-40975aa4de74?w=900&q=80"),
    ("coq au vin", "https://images.unsplash.com/photo-1547592180-85f173990554?w=900&q=80"),
    ("crêpe", "https://images.unsplash.com/photo-1519676867240-f03562e64548?w=900&q=80"),
    ("crepe", "https://images.unsplash.com/photo-1519676867240-f03562e64548?w=900&q=80"),

    ("shakshuka", "https://images.unsplash.com/photo-1590412200988-a436970781fa?w=900&q=80"),
    ("shawarma", "https://images.unsplash.com/photo-1599487488170-d11ec9c172f0?w=900&q=80"),
    ("kebab", "https://images.unsplash.com/photo-1529042410759-befb1204b468?w=900&q=80"),
    ("tagine", "https://images.unsplash.com/photo-1585937421612-70a008356fbe?w=900&q=80"),
    ("couscous", "https://images.unsplash.com/photo-1505253213348-cd54c92b37fe?w=900&q=80"),
    ("kofta", "https://images.unsplash.com/photo-1529042410759-befb1204b468?w=900&q=80"),
    ("fattoush", "https://images.unsplash.com/photo-1540189549336-e6e99c3679fe?w=900&q=80"),
    ("mujaddara", "https://images.unsplash.com/photo-1585937421612-70a008356fbe?w=900&q=80"),

    ("salad", "https://images.unsplash.com/photo-1505253716362-afaea1d3d1af?w=900&q=80"),
    ("soup", "https://images.unsplash.com/photo-1547592180-85f173990554?w=900&q=80"),
    ("sandwich", "https://images.unsplash.com/photo-1528735602780-2552fd46c7af?w=900&q=80"),
    ("wrap", "https://images.unsplash.com/photo-1593560704563-f176a2eb61db?w=900&q=80"),
    ("rice", "https://images.unsplash.com/photo-1603133872878-684f208fb84b?w=900&q=80"),
]

_CUISINE_FALLBACK_IMAGE: dict[str, str] = {
    "Italian": "https://images.unsplash.com/photo-1498579150354-977475b7ea0b?w=900&q=80",
    "Mexican": "https://images.unsplash.com/photo-1552332386-f8dd00bc2f85?w=900&q=80",
    "Asian": "https://images.unsplash.com/photo-1546069901-ba9599a7e63c?w=900&q=80",
    "Indian": "https://images.unsplash.com/photo-1585937421612-70a008356fbe?w=900&q=80",
    "American": "https://images.unsplash.com/photo-1568901346375-23c9450c58cd?w=900&q=80",
    "Mediterranean": "https://images.unsplash.com/photo-1540189549336-e6e99c3679fe?w=900&q=80",
    "French": "https://images.unsplash.com/photo-1571877227200-a0d98ea607e9?w=900&q=80",
    "Middle Eastern": "https://images.unsplash.com/photo-1544025162-d76694265947?w=900&q=80",
}

_GENERIC_FALLBACK_IMAGE = (
    "https://images.unsplash.com/photo-1504674900247-0877df9cc836?w=900&q=80"
)
_STAPLE_SUGGESTIONS = [
    "olive oil", "garlic", "onion", "eggs", "rice", "pasta", "canned tomatoes",
    "black beans", "chicken breast", "spinach", "bread", "cheddar cheese",
]
_SHORTLIST_LIMIT = 1200
_SHORTLIST_FLOOR = 300


def _image_for(recipe_id: str, cuisine: str, name: str = "") -> str:
    low = name.lower()
    for kw, url in _KEYWORD_IMAGES:
        if kw in low:
            return url
    return _CUISINE_FALLBACK_IMAGE.get(cuisine, _GENERIC_FALLBACK_IMAGE)


def _recipe_to_dict(recipe: Any) -> dict[str, Any]:
    d = recipe.model_dump()
    d["image_url"] = _image_for(recipe.id, recipe.cuisine, recipe.name)
    d["cuisine_emoji"] = _CUISINE_EMOJI.get(recipe.cuisine, "\U0001F37D️")
    d["total_time_min"] = recipe.prep_time_min + recipe.cook_time_min
    return d


def _precompute_search_tokens(recipe: Any) -> frozenset[str]:
    tokens = set(_ingredient_tokens(recipe.name))
    tokens.update(_ingredient_tokens(recipe.cuisine))
    for ingredient in recipe.ingredients:
        tokens.update(_ingredient_tokens(ingredient.name))
    return frozenset(tokens)


def _shortlist_recipes(
    recipes: list[Any],
    search_tokens: list[frozenset[str]],
    profile: UserProfile,
    limit: int = _SHORTLIST_LIMIT,
) -> list[Any]:
    pantry_tokens: set[str] = set()
    for item in profile.pantry:
        pantry_tokens.update(_ingredient_tokens(item))

    preferred = {c.lower() for c in profile.soft_preferences.preferred_cuisines}
    wanted_tags = {GOAL_TO_TAG.get(goal, "") for goal in profile.soft_preferences.goals}
    wanted_tags.discard("")
    hard_constraints = set(profile.hard_constraints)
    max_prep = profile.soft_preferences.max_prep_time_min

    if not pantry_tokens and not preferred and not wanted_tags and not hard_constraints and max_prep is None:
        return recipes[:limit]

    scored: list[tuple[float, Any]] = []
    for recipe, recipe_tokens in zip(recipes, search_tokens):
        overlap = len(recipe_tokens & pantry_tokens)
        cuisine_match = 1 if recipe.cuisine.lower() in preferred else 0
        goal_match = len(set(recipe.dietary_tags) & wanted_tags)
        hard_match = len(set(recipe.dietary_tags) & hard_constraints)
        prep_match = 1 if max_prep is not None and recipe.prep_time_min <= max_prep else 0

        if overlap == 0 and cuisine_match == 0 and goal_match == 0 and hard_match == 0 and prep_match == 0:
            continue

        score = (
            cuisine_match * 1000
            + goal_match * 250
            + hard_match * 150
            + overlap * 25
            + prep_match * 10
            - recipe.prep_time_min * 0.05
        )
        scored.append((score, recipe))

    if len(scored) < _SHORTLIST_FLOOR:
        return recipes[:limit]

    scored.sort(key=lambda item: item[0], reverse=True)
    return [recipe for _, recipe in scored[:limit]]


def _shopping_advice(
    profile: UserProfile,
    recipes: list[Any],
    results: list[Any],
) -> dict[str, Any]:
    pantry = {item.lower() for item in profile.pantry}
    pantry_count = len(profile.pantry)
    suggestions: list[str] = []
    preferred_cuisines = list(profile.soft_preferences.preferred_cuisines or [])
    scorer = IngredientScorer()

    top_recipe = results[0].adapted_recipe.recipe if results else None
    top_score = scorer.score(top_recipe, profile.pantry) if top_recipe is not None else None
    survivor_count = len(results[0].filter_result.survivors) if results and results[0].filter_result.survivors else 0
    preferred_survivor_count = (
        sum(1 for survivor in results[0].filter_result.survivors if survivor.recipe.cuisine in preferred_cuisines)
        if results and results[0].filter_result.survivors and preferred_cuisines
        else 0
    )

    sparse_pantry = pantry_count < 7
    weak_match = (
        top_score is not None
        and (top_score.coverage < 0.55 or top_score.essential_coverage < 0.67)
    )
    limited_variety = survivor_count > 0 and survivor_count <= 3
    limited_preferred_options = bool(preferred_cuisines) and preferred_survivor_count <= 1
    should_alert = sparse_pantry or weak_match or limited_variety or limited_preferred_options

    if results:
        core_missing_counter: Counter[str] = Counter()
        missing_counter: Counter[str] = Counter()
        for rec in results[:8]:
            rec_score = scorer.score(rec.adapted_recipe.recipe, profile.pantry)
            for item in rec_score.essential_missing_ingredients or []:
                if item not in pantry:
                    core_missing_counter[item] += 2
            for item in rec_score.missing_ingredients:
                if item not in pantry:
                    missing_counter[item] += 1

        suggestions.extend(
            item for item, _ in core_missing_counter.most_common(5)
            if item not in suggestions
        )
        suggestions.extend(
            item for item, _ in missing_counter.most_common(8)
            if item not in suggestions
        )

    for staple in _STAPLE_SUGGESTIONS:
        if staple not in pantry and staple not in suggestions:
            suggestions.append(staple)

    reason_bits: list[str] = []
    if sparse_pantry:
        reason_bits.append(f"you only have {pantry_count} pantry item{'s' if pantry_count != 1 else ''} selected")
    if limited_preferred_options:
        cuisine_label = " / ".join(preferred_cuisines)
        recipe_word = "recipe" if preferred_survivor_count == 1 else "recipes"
        verb = "fits" if preferred_survivor_count == 1 else "fit"
        reason_bits.append(
            f"you only have {preferred_survivor_count} {cuisine_label} {recipe_word} that really {verb} this pantry"
        )
    elif limited_variety:
        recipe_word = "recipe" if survivor_count == 1 else "recipes"
        reason_bits.append(f"there are only {survivor_count} workable {recipe_word} from what you have on hand")
    if weak_match and top_score is not None:
        missing_count = len(top_score.missing_ingredients)
        if missing_count > 0:
            ingredient_word = "ingredient" if missing_count == 1 else "ingredients"
            reason_bits.append(f"even the best match still needs about {missing_count} more {ingredient_word}")

    if should_alert:
        if reason_bits:
            message = (
                "A quick grocery run would help. "
                + "Right now " + " and ".join(reason_bits) + "."
            )
        else:
            message = "A quick grocery run would unlock stronger recipe matches."
    else:
        message = ""

    return {
        "show_alert": should_alert,
        "message": message,
        "suggestions": suggestions[:8],
    }


class RecommendRequest(BaseModel):
    pantry: list[str]
    hard_constraints: list[str] = []
    goals: list[str] = []
    preferred_cuisines: list[str] = []
    max_prep_time_min: int | None = None
    top_k: int = 3
    rating_history: list[dict[str, Any]] = []


_MAX_TOP_K = 12


def create_app(recipes_path: Path | None = None) -> FastAPI:
    app = FastAPI(title="UseItUp", version="0.1.0")

    recipes = load_recipes(recipes_path or _RECIPES_PATH)
    recipe_search_tokens = [_precompute_search_tokens(recipe) for recipe in recipes]
    scorer = IngredientScorer()

    @app.get("/api/recipes")
    def list_recipes() -> list[dict[str, Any]]:
        return [_recipe_to_dict(r) for r in recipes]

    @app.get("/api/profile/demo")
    def get_demo_profile() -> dict[str, Any]:
        profile = load_profile("demo_user")
        return profile.model_dump()

    @app.get("/api/vocabulary")
    def vocabulary() -> dict[str, list[str]]:
        return {
            "dietary_tags": [
                "vegan", "vegetarian", "gluten-free", "dairy-free",
                "nut-free", "low-carb", "high-protein", "low-cost", "quick",
            ],
            "goals": [
                "high_protein", "low_cost", "vegetarian", "vegan",
                "low_carb", "quick", "dairy_free",
            ],
            "cuisines": [
                "Italian", "Mexican", "Asian", "Indian", "American",
                "Mediterranean", "French", "Middle Eastern",
            ],
        }

    @app.post("/api/recommend")
    def run_recommend(req: RecommendRequest) -> dict[str, Any]:
        try:
            profile = UserProfile(
                user_id="web_session",
                hard_constraints=req.hard_constraints,
                soft_preferences=SoftPreferences(
                    max_prep_time_min=req.max_prep_time_min,
                    preferred_cuisines=req.preferred_cuisines,
                    goals=req.goals,
                ),
                rating_history=req.rating_history,  # type: ignore[arg-type]
                pantry=[p.strip() for p in req.pantry if p.strip()],
            )
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid profile: {e}")

        candidate_recipes = _shortlist_recipes(recipes, recipe_search_tokens, profile)

        try:
            results = recommend(profile, candidate_recipes, top_k=max(1, min(req.top_k, _MAX_TOP_K)))
        except ValueError as e:
            return {
                "recommendations": [],
                "error": str(e),
                "decision_log": [],
                "summary": {
                    "candidates_considered": len(candidate_recipes),
                    "survivors": 0,
                    "coverage_top": 0.0,
                    "weighted_coverage_top": 0.0,
                    "essential_coverage_top": 0.0,
                    "shopping_advice": _shopping_advice(profile, candidate_recipes, []),
                },
            }

        payload_recs = []
        for rec in results:
            score = scorer.score(rec.adapted_recipe.recipe, profile.pantry)
            payload_recs.append({
                "recipe": _recipe_to_dict(rec.adapted_recipe.recipe),
                "adaptations": [asdict(a) for a in rec.adapted_recipe.adaptations],
                "match": {
                    "coverage": score.coverage,
                    "weighted_coverage": score.weighted_coverage,
                    "essential_coverage": score.essential_coverage,
                    "matched_ingredients": score.matched_ingredients,
                    "missing_ingredients": score.missing_ingredients,
                    "essential_missing_ingredients": score.essential_missing_ingredients or [],
                    "essential_overlap_count": score.essential_overlap_count,
                    "essential_total": score.essential_total,
                },
                "explanation": {
                    "goal_trace": rec.explanation.goal_trace,
                    "counterfactual": rec.explanation.counterfactual,
                    "cbr_trace": rec.explanation.cbr_trace,
                    "ingredient_utilization_report": rec.explanation.ingredient_utilization_report,
                    "rendered": render_explanation(rec.explanation),
                },
                "cbr": {
                    "similarity_score": rec.cbr_match.similarity_score,
                    "fallback_reason": rec.cbr_match.fallback_reason,
                    "nearest_past_recipe": (
                        _recipe_to_dict(rec.cbr_match.nearest_past_recipe)
                        if rec.cbr_match.nearest_past_recipe is not None
                        else None
                    ),
                    "breakdown": asdict(rec.cbr_match.similarity_breakdown),
                },
            })

        top_score = scorer.score(results[0].adapted_recipe.recipe, profile.pantry) if results else None
        summary = {
            "candidates_considered": len(candidate_recipes),
            "survivors": len(results[0].filter_result.survivors),
            "coverage_top": (
                top_score.coverage if top_score else 0.0
            ),
            "weighted_coverage_top": (
                top_score.weighted_coverage if top_score else 0.0
            ),
            "essential_coverage_top": (
                top_score.essential_coverage if top_score else 0.0
            ),
            "shopping_advice": _shopping_advice(profile, candidate_recipes, results),
        }

        # Top 8 rules fired, in log order
        log = [
            {
                "rule_name": e.rule_name,
                "recipe_id": e.recipe_id,
                "passed": e.passed,
                "reason": e.reason,
            }
            for e in results[0].decision_log[:40]
        ]

        return {
            "recommendations": payload_recs,
            "decision_log": log,
            "summary": summary,
        }

    # Serve static assets (CSS/JS) and the SPA entry point.
    app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")

    @app.get("/")
    def root() -> FileResponse:
        return FileResponse(_STATIC_DIR / "index.html")

    return app


app = create_app()
