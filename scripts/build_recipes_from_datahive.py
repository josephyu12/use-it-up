"""Convert the datahiveai/recipes-with-nutrition parquet into data/recipes.json.

Source dataset (39k recipes, CC BY-NC 4.0):
    https://huggingface.co/datasets/datahiveai/recipes-with-nutrition

This script maps the Edamam-flavored source fields onto UseItUp's Recipe
schema, inferring the fields that the source doesn't carry (prep/cook times,
difficulty, flavor profile, is_core flags, some dietary tags).

Run:
    python scripts/build_recipes_from_datahive.py \\
        --parquet data/raw/datahive_train.parquet \\
        --out data/recipes.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from useitup.enrichment import (  # noqa: E402
    classify_category,
    compute_weight_shares,
    infer_flavor,
    infer_is_core,
)
from useitup.schemas import Recipe  # noqa: E402

# --------------------------------------------------------------------------
# Label mapping: Edamam health/diet labels → UseItUp DietaryTag values
# --------------------------------------------------------------------------

_HEALTH_LABEL_MAP: dict[str, str] = {
    "vegan": "vegan",
    "vegetarian": "vegetarian",
    "gluten-free": "gluten-free",
    "dairy-free": "dairy-free",
    "low-carb": "low-carb",
    "high-protein": "high-protein",
}
_NUT_FREE_REQUIRED: set[str] = {"tree-nut-free", "peanut-free"}

_ALLOWED_DIETARY: set[str] = {
    "vegan", "vegetarian", "gluten-free", "dairy-free", "nut-free",
    "low-carb", "high-protein", "low-cost", "quick",
}

# --------------------------------------------------------------------------
# Nutrition: pull macros out of the `digest` tree
# --------------------------------------------------------------------------

_NUTRIENT_TAGS: dict[str, str] = {
    "FAT": "fat_g",
    "CHOCDF": "carbs_g",
    "PROCNT": "protein_g",
}


def _parse_json(raw: object) -> object:
    """Datahive stringifies all list/dict fields — decode safely."""
    if raw is None or raw == "":
        return None
    if isinstance(raw, (list, dict)):
        return raw
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        return None


def _nutrition_per_serving(
    calories_total: float | None,
    digest: list[dict] | None,
    servings: float,
) -> dict[str, float]:
    servings = max(servings or 1.0, 1.0)
    out: dict[str, float] = {}
    if calories_total is not None:
        out["calories"] = round(calories_total / servings, 1)
    if digest:
        for entry in digest:
            tag = entry.get("tag")
            if tag in _NUTRIENT_TAGS:
                total = entry.get("total") or 0.0
                out[_NUTRIENT_TAGS[tag]] = round(total / servings, 2)
    for key in ("calories", "protein_g", "carbs_g", "fat_g"):
        out.setdefault(key, 0.0)
    return out


# --------------------------------------------------------------------------
# Time & difficulty inference
# --------------------------------------------------------------------------

def _estimate_times(num_ingredients: int, dish_types: list[str]) -> tuple[int, int]:
    """Rough prep/cook heuristic, informed by dish type.

    Source has no explicit times, so we estimate from recipe complexity.
    Ranges stay plausible (5-75 min total) and feed the `quick` dietary tag.
    """
    dish = " ".join(dish_types).lower()
    if "drink" in dish or "salad" in dish or "sandwich" in dish:
        prep = 10
        cook = 0
    elif "desserts" in dish or "dessert" in dish or "bread" in dish:
        prep = 15
        cook = 30
    elif "soup" in dish or "stew" in dish:
        prep = 15
        cook = 40
    elif "pizza" in dish:
        prep = 20
        cook = 15
    else:
        prep = 10 + max(0, min(num_ingredients - 3, 10))
        cook = 15 + max(0, min(num_ingredients - 3, 15))
    return prep, cook


def _estimate_difficulty(num_ingredients: int) -> int:
    if num_ingredients <= 4:
        return 1
    if num_ingredients <= 6:
        return 2
    if num_ingredients <= 9:
        return 3
    if num_ingredients <= 12:
        return 4
    return 5


# --------------------------------------------------------------------------
# Ingredient normalization
# --------------------------------------------------------------------------

_MEASURE_STOP = {"", "unit", "<unit>", "serving", "servings"}


def _clean_ingredient_name(food: str) -> str:
    name = (food or "").strip().lower()
    name = re.sub(r"\s+", " ", name)
    # Strip trailing parentheticals like "parsley (for garnish)"
    name = re.sub(r"\s*\([^)]*\)\s*$", "", name).strip()
    return name


def _clean_unit(measure: str | None) -> str | None:
    if not measure:
        return None
    m = measure.strip().lower()
    if m in _MEASURE_STOP:
        return None
    return m


def _build_ingredient(
    raw: dict,
    idx: int,
    recipe_name: str,
    weight_share: float | None,
) -> dict | None:
    food = _clean_ingredient_name(raw.get("food") or "")
    if not food:
        return None
    category = classify_category(food)
    is_core = infer_is_core(
        food, category, idx, recipe_name,
        weight_share=weight_share,
    )
    qty = raw.get("quantity")
    try:
        quantity = float(qty) if qty is not None else None
    except (TypeError, ValueError):
        quantity = None
    return {
        "name": food,
        "quantity": quantity,
        "unit": _clean_unit(raw.get("measure")),
        "category": category,
        "is_core": is_core,
    }


# --------------------------------------------------------------------------
# Label normalization
# --------------------------------------------------------------------------

def _normalize_dietary_tags(
    health_labels: list[str],
    diet_labels: list[str],
    ingredient_names: list[str],
    total_minutes: int,
) -> list[str]:
    lowered = {label.strip().lower() for label in (health_labels or [])}
    tags: list[str] = []

    for label, mapped in _HEALTH_LABEL_MAP.items():
        if label in lowered:
            tags.append(mapped)

    # Datahive uses two separate tree-nut-free / peanut-free labels; our
    # schema has a single `nut-free`. Require both to avoid false positives.
    if _NUT_FREE_REQUIRED.issubset(lowered):
        tags.append("nut-free")

    diet_lowered = {d.strip().lower() for d in (diet_labels or [])}
    if "low-carb" in diet_lowered:
        tags.append("low-carb")
    if "high-protein" in diet_lowered:
        tags.append("high-protein")

    if total_minutes <= 30:
        tags.append("quick")

    seen: set[str] = set()
    deduped: list[str] = []
    for tag in tags:
        if tag in _ALLOWED_DIETARY and tag not in seen:
            seen.add(tag)
            deduped.append(tag)
    return deduped


def _normalize_cuisine(raw: list[str] | None) -> str:
    if not raw:
        return "Other"
    first = raw[0].strip()
    if not first:
        return "Other"
    # "south east asian" → "South East Asian"
    return " ".join(word.capitalize() for word in first.split())


def _slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return slug[:60] or "recipe"


# --------------------------------------------------------------------------
# Per-row conversion
# --------------------------------------------------------------------------

def convert_row(row: dict, idx: int) -> dict | None:
    name = (row.get("recipe_name") or "").strip()
    if not name:
        return None

    raw_ings = _parse_json(row.get("ingredients"))
    if not isinstance(raw_ings, list) or not raw_ings:
        return None

    weight_shares = compute_weight_shares(raw_ings)
    ingredients: list[dict] = []
    for i_idx, raw in enumerate(raw_ings):
        if not isinstance(raw, dict):
            continue
        share = weight_shares[i_idx] if i_idx < len(weight_shares) else None
        built = _build_ingredient(raw, i_idx, name, share)
        if built is not None:
            ingredients.append(built)
    if not ingredients:
        return None

    cuisine = _normalize_cuisine(_parse_json(row.get("cuisine_type")) or [])
    dish_types = _parse_json(row.get("dish_type")) or []
    health_labels = _parse_json(row.get("health_labels")) or []
    diet_labels = _parse_json(row.get("diet_labels")) or []

    prep, cook = _estimate_times(len(ingredients), dish_types)
    total = prep + cook
    difficulty = _estimate_difficulty(len(ingredients))

    ing_names = [ing["name"] for ing in ingredients]
    dietary_tags = _normalize_dietary_tags(health_labels, diet_labels, ing_names, total)
    flavor = infer_flavor(ing_names)

    servings = row.get("servings")
    servings_f = float(servings) if servings is not None else 1.0
    digest = _parse_json(row.get("digest"))
    calories = row.get("calories")
    calories_f = float(calories) if calories is not None else None
    nutrition = _nutrition_per_serving(
        calories_f,
        digest if isinstance(digest, list) else None,
        servings_f,
    )

    url = (row.get("url") or "").strip()
    instructions = (
        [f"Follow the original recipe at: {url}"]
        if url
        else ["See source for detailed instructions."]
    )

    rid = f"d{idx:06d}-{_slugify(name)}"
    return {
        "id": rid,
        "name": name,
        "ingredients": ingredients,
        "cuisine": cuisine,
        "dietary_tags": dietary_tags,
        "prep_time_min": prep,
        "cook_time_min": cook,
        "difficulty": difficulty,
        "nutrition": nutrition,
        "flavor_profile": flavor,
        "instructions": instructions,
    }


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", type=Path,
                        default=ROOT / "data" / "raw" / "datahive_train.parquet")
    parser.add_argument("--out", type=Path,
                        default=ROOT / "data" / "recipes.json")
    parser.add_argument("--limit", type=int, default=None,
                        help="Convert only N rows (for debugging).")
    args = parser.parse_args()

    if not args.parquet.exists():
        raise SystemExit(f"parquet not found: {args.parquet}")

    table = pq.read_table(args.parquet)
    df = table.to_pandas()
    if args.limit:
        df = df.head(args.limit)

    out: list[dict] = []
    dropped_empty = 0
    dropped_invalid = 0
    errors: list[str] = []

    for idx, row in enumerate(df.to_dict(orient="records"), start=1):
        converted = convert_row(row, idx)
        if converted is None:
            dropped_empty += 1
            continue
        try:
            Recipe.model_validate(converted)
        except Exception as e:
            dropped_invalid += 1
            if len(errors) < 5:
                errors.append(f"{converted.get('name', '??')}: {str(e)[:150]}")
            continue
        out.append(converted)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(out, ensure_ascii=False),
        encoding="utf-8",
    )

    by_cuisine: dict[str, int] = {}
    for r in out:
        by_cuisine[r["cuisine"]] = by_cuisine.get(r["cuisine"], 0) + 1
    size_mb = args.out.stat().st_size / 1e6

    print(f"Wrote {len(out):,} recipes → {args.out} ({size_mb:.1f} MB)")
    print(f"Dropped: {dropped_empty} empty, {dropped_invalid} schema-invalid")
    print("Top cuisines: " + ", ".join(
        f"{k}={v}" for k, v in sorted(by_cuisine.items(), key=lambda p: -p[1])[:8]
    ))
    if errors:
        print("\nSample validation errors:")
        for e in errors:
            print(f"  - {e}")


if __name__ == "__main__":
    main()
