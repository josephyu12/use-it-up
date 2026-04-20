"""
Build data/recipes.json from Food.com RAW_recipes.csv.

Usage:
    Place RAW_recipes.csv in data/raw/, then run:
        python scripts/build_dataset.py
"""

import ast
import json
import re
import sys
import uuid
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
RAW_CSV = ROOT / "data" / "raw" / "RAW_recipes.csv"
OUT_JSON = ROOT / "data" / "recipes.json"

TARGET_COUNT = 600
TARGET_CUISINES = 8

# ---------------------------------------------------------------------------
# Keyword rulebooks
# ---------------------------------------------------------------------------

CATEGORY_RULES: list[tuple[list[str], str]] = [
    (["chicken", "beef", "pork", "lamb", "salmon", "tuna", "shrimp", "turkey",
      "bacon", "sausage", "tofu", "tempeh", "lentil", "chickpea", "black bean",
      "kidney bean", "egg", "cod", "tilapia", "crab", "lobster", "duck",
      "venison", "anchovy", "sardine", "ham", "pepperoni", "prosciutto"], "protein"),
    (["flour", "rice", "pasta", "bread", "oat", "quinoa", "barley", "corn",
      "wheat", "noodle", "tortilla", "couscous", "polenta", "rye", "bulgur",
      "spaghetti", "fettuccine", "penne", "breadcrumb", "cracker"], "grain"),
    (["milk", "cheese", "butter", "cream", "yogurt", "sour cream", "ghee",
      "mozzarella", "parmesan", "cheddar", "feta", "ricotta", "brie",
      "heavy cream", "half and half", "buttermilk", "whey"], "dairy"),
    (["salt", "pepper", "cumin", "paprika", "turmeric", "cinnamon", "oregano",
      "thyme", "basil", "rosemary", "ginger", "chili", "cayenne", "coriander",
      "cardamom", "nutmeg", "clove", "bay leaf", "saffron", "sumac",
      "five spice", "allspice", "fennel seed", "mustard seed", "curry"], "spice"),
    (["oil", "olive oil", "vegetable oil", "canola oil", "coconut oil",
      "sesame oil", "avocado", "lard", "shortening", "margarine",
      "peanut butter", "almond butter", "tahini"], "fat"),
    (["onion", "garlic", "tomato", "carrot", "celery", "spinach", "broccoli",
      "pepper", "mushroom", "zucchini", "eggplant", "cucumber", "lettuce",
      "kale", "cabbage", "potato", "sweet potato", "pea", "corn", "asparagus",
      "green bean", "cauliflower", "leek", "shallot", "scallion", "beet",
      "radish", "artichoke", "pumpkin", "squash", "chard"], "vegetable"),
    (["soy sauce", "ketchup", "mustard", "mayo", "mayonnaise", "vinegar",
      "hot sauce", "worcestershire", "fish sauce", "oyster sauce", "hoisin",
      "sriracha", "miso", "teriyaki", "salsa", "relish", "bbq sauce",
      "honey", "maple syrup", "molasses", "jam", "chutney", "tahini",
      "hummus", "pesto", "tomato sauce", "pasta sauce", "stock", "broth"], "condiment"),
]

MEAT_KEYWORDS = {
    "chicken", "beef", "pork", "lamb", "salmon", "tuna", "shrimp", "turkey",
    "bacon", "sausage", "cod", "tilapia", "crab", "lobster", "duck",
    "venison", "anchovy", "sardine", "ham", "pepperoni", "prosciutto",
    "meat", "fish", "seafood", "clam", "oyster", "mussel", "scallop",
    "gelatin", "lard",
}
DAIRY_KEYWORDS = {
    "milk", "cheese", "butter", "cream", "yogurt", "sour cream", "ghee",
    "mozzarella", "parmesan", "cheddar", "feta", "ricotta", "brie",
    "heavy cream", "half and half", "buttermilk", "whey", "casein",
}
GLUTEN_KEYWORDS = {
    "flour", "bread", "pasta", "wheat", "barley", "rye", "noodle",
    "spaghetti", "fettuccine", "penne", "breadcrumb", "cracker",
    "bulgur", "couscous", "semolina", "spelt", "kamut",
}
NUT_KEYWORDS = {
    "almond", "walnut", "pecan", "cashew", "pistachio", "hazelnut",
    "pine nut", "macadamia", "peanut", "nut",
}

FLAVOR_RULES: list[tuple[list[str], str]] = [
    (["chili", "jalapeño", "sriracha", "cayenne", "hot sauce", "pepper flake",
      "chipotle", "habanero", "tabasco", "spicy", "chilli"], "spicy"),
    (["soy sauce", "miso", "parmesan", "mushroom", "anchovy", "fish sauce",
      "worcestershire", "umami", "nutritional yeast"], "umami"),
    (["salt", "savory", "broth", "stock", "herb", "thyme", "rosemary",
      "oregano", "garlic", "onion", "sage"], "savory"),
    (["sugar", "honey", "maple syrup", "molasses", "chocolate", "vanilla",
      "caramel", "cinnamon", "sweet", "jam"], "sweet"),
    (["lemon", "lime", "vinegar", "tamarind", "yogurt", "sour cream",
      "cranberry", "tart", "citrus"], "sour"),
    (["smoked", "smoke", "bbq", "chipotle", "liquid smoke", "paprika",
      "bacon", "charred", "grilled"], "smoky"),
    (["fresh", "mint", "basil", "cilantro", "dill", "cucumber", "raw",
      "salad", "parsley", "lime zest", "lemon zest"], "fresh"),
    (["cream", "butter", "coconut milk", "heavy cream", "cheese", "rich",
      "avocado", "olive oil", "tahini"], "rich"),
]

CUISINE_KEYWORDS: dict[str, list[str]] = {
    "Italian": ["pasta", "parmesan", "mozzarella", "pesto", "risotto", "marinara",
                "lasagna", "polenta", "prosciutto", "basil", "oregano"],
    "Mexican": ["tortilla", "salsa", "cumin", "cilantro", "jalapeño", "taco",
                "enchilada", "tamale", "chipotle", "avocado", "lime"],
    "Asian": ["soy sauce", "ginger", "sesame oil", "rice vinegar", "tofu",
              "bok choy", "miso", "hoisin", "five spice", "noodle", "stir fry"],
    "Indian": ["curry", "turmeric", "garam masala", "cumin", "coriander",
               "cardamom", "ghee", "naan", "basmati", "dal", "paneer"],
    "American": ["bbq", "ketchup", "bacon", "cheddar", "cornmeal", "maple syrup",
                 "ranch", "burger", "hot dog", "biscuit"],
    "Mediterranean": ["feta", "olive oil", "lemon", "sumac", "tahini", "hummus",
                      "chickpea", "eggplant", "kalamata", "pita"],
    "French": ["beurre", "crème", "dijon", "herbes de provence", "baguette",
               "gruyère", "shallot", "tarragon", "brie", "camembert", "wine"],
    "Middle Eastern": ["za'atar", "sumac", "tahini", "pomegranate", "harissa",
                       "ras el hanout", "preserved lemon", "flatbread"],
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def classify_category(ingredient_name: str) -> str:
    lower = ingredient_name.lower()
    for keywords, category in CATEGORY_RULES:
        if any(kw in lower for kw in keywords):
            return category
    return "other"


def infer_flavor_profile(ingredient_names: list[str]) -> list[str]:
    text = " ".join(ingredient_names).lower()
    return [
        tag for keywords, tag in FLAVOR_RULES
        if any(kw in text for kw in keywords)
    ]


def infer_dietary_tags(
    ingredient_names: list[str],
    total_time_min: int,
) -> list[str]:
    lower_names = [n.lower() for n in ingredient_names]
    combined = " ".join(lower_names)
    tags: list[str] = []

    has_meat = any(any(kw in n for kw in MEAT_KEYWORDS) for n in lower_names)
    has_dairy = any(any(kw in n for kw in DAIRY_KEYWORDS) for n in lower_names)
    has_gluten = any(any(kw in n for kw in GLUTEN_KEYWORDS) for n in lower_names)
    has_nuts = any(any(kw in n for kw in NUT_KEYWORDS) for n in lower_names)

    if not has_meat and not has_dairy:
        tags.append("vegan")
    if not has_meat:
        tags.append("vegetarian")
    if not has_dairy:
        tags.append("dairy-free")
    if not has_gluten:
        tags.append("gluten-free")
    if not has_nuts:
        tags.append("nut-free")
    if total_time_min <= 30:
        tags.append("quick")

    # high-protein: multiple protein-category ingredients
    protein_count = sum(
        1 for n in lower_names
        if any(kw in n for kw in [
            "chicken", "beef", "pork", "salmon", "tuna", "shrimp", "turkey",
            "egg", "tofu", "lentil", "chickpea", "black bean", "kidney bean",
        ])
    )
    if protein_count >= 2:
        tags.append("high-protein")

    # low-carb: no grain ingredients
    if not has_gluten and "rice" not in combined and "potato" not in combined:
        tags.append("low-carb")

    return list(dict.fromkeys(tags))  # preserve order, deduplicate


def guess_cuisine(ingredient_names: list[str], tags_str: str) -> str:
    text = (" ".join(ingredient_names) + " " + tags_str).lower()
    scores: dict[str, int] = {c: 0 for c in CUISINE_KEYWORDS}
    for cuisine, kws in CUISINE_KEYWORDS.items():
        for kw in kws:
            if kw in text:
                scores[cuisine] += 1
    best = max(scores, key=lambda c: scores[c])
    return best if scores[best] > 0 else "American"


def parse_ingredient_list(raw: str) -> list[dict]:
    """Parse Food.com ingredient string list into Ingredient dicts."""
    try:
        items: list[str] = ast.literal_eval(raw)
    except Exception:
        return []
    results = []
    unit_pattern = re.compile(
        r"^([\d./\s]+)?\s*"
        r"(cups?|tablespoons?|tbsp|teaspoons?|tsp|oz|ounces?|lbs?|pounds?|g|grams?|"
        r"kg|ml|liters?|l|cloves?|stalks?|slices?|pieces?|cans?|packages?)?\s*(.+)$",
        re.IGNORECASE,
    )
    for item in items:
        item = item.strip()
        if not item:
            continue
        m = unit_pattern.match(item)
        if m:
            qty_str, unit, name = m.group(1), m.group(2), m.group(3)
        else:
            qty_str, unit, name = None, None, item
        try:
            quantity: float | None = float(qty_str.strip()) if qty_str and qty_str.strip() else None
        except ValueError:
            quantity = None
        name = re.sub(r"\(.*?\)", "", name).strip().lower()
        if not name:
            continue
        results.append({
            "name": name,
            "quantity": quantity,
            "unit": unit.strip().lower() if unit else None,
            "category": classify_category(name),
        })
    return results


def safe_int(val: object, default: int = 0) -> int:
    try:
        return int(float(str(val)))  # type: ignore[arg-type]
    except (ValueError, TypeError):
        return default


def difficulty_from_time(minutes: int, n_steps: int) -> int:
    score = minutes / 30 + n_steps / 5
    if score < 2:
        return 1
    elif score < 4:
        return 2
    elif score < 6:
        return 3
    elif score < 9:
        return 4
    return 5


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build(raw_csv: Path = RAW_CSV, out_json: Path = OUT_JSON) -> None:
    if not raw_csv.exists():
        print(f"ERROR: {raw_csv} not found.")
        print("Please download RAW_recipes.csv from Kaggle (Food.com dataset)")
        print("and place it in data/raw/ before running this script.")
        sys.exit(1)

    print(f"Loading {raw_csv} …")
    df = pd.read_csv(raw_csv, low_memory=False)

    required = {"id", "name", "ingredients", "steps", "minutes", "tags"}
    missing = required - set(df.columns)
    if missing:
        print(f"ERROR: CSV is missing columns: {missing}")
        sys.exit(1)

    # Drop rows with nulls in key columns
    df = df.dropna(subset=["name", "ingredients", "steps"])
    df["minutes"] = df["minutes"].apply(lambda x: safe_int(x))
    df = df[df["minutes"] > 0]

    print(f"Rows after cleaning: {len(df)}")

    recipes: list[dict] = []
    cuisine_buckets: dict[str, list[dict]] = {c: [] for c in CUISINE_KEYWORDS}
    cuisine_buckets["Other"] = []

    for _, row in df.iterrows():
        ingredients_raw = parse_ingredient_list(str(row["ingredients"]))
        if len(ingredients_raw) < 2:
            continue

        try:
            steps: list[str] = ast.literal_eval(str(row["steps"]))
        except Exception:
            steps = [str(row["steps"])]

        total_min = safe_int(row["minutes"])
        prep_min = total_min // 3
        cook_min = total_min - prep_min

        ing_names = [i["name"] for i in ingredients_raw]
        tags_str = str(row.get("tags", ""))
        cuisine = guess_cuisine(ing_names, tags_str)
        dietary_tags = infer_dietary_tags(ing_names, total_min)
        flavor_profile = infer_flavor_profile(ing_names)
        n_steps = len(steps) if isinstance(steps, list) else 5
        difficulty = difficulty_from_time(total_min, n_steps)

        recipe_dict = {
            "id": str(row.get("id", uuid.uuid4())),
            "name": str(row["name"]),
            "ingredients": ingredients_raw,
            "cuisine": cuisine,
            "dietary_tags": dietary_tags,
            "prep_time_min": prep_min,
            "cook_time_min": cook_min,
            "difficulty": difficulty,
            "nutrition": {
                "calories": 0.0,
                "protein_g": 0.0,
                "carbs_g": 0.0,
                "fat_g": 0.0,
            },
            "flavor_profile": flavor_profile,
            "instructions": steps if isinstance(steps, list) else [str(steps)],
        }
        cuisine_buckets.get(cuisine, cuisine_buckets["Other"]).append(recipe_dict)

    # Sample across cuisines
    per_cuisine = TARGET_COUNT // max(len(cuisine_buckets), 1)
    sampled: list[dict] = []
    for bucket in cuisine_buckets.values():
        sampled.extend(bucket[:per_cuisine])

    # Fill up to TARGET_COUNT
    remaining = [r for bucket in cuisine_buckets.values() for r in bucket[per_cuisine:]]
    need = TARGET_COUNT - len(sampled)
    sampled.extend(remaining[:need])
    sampled = sampled[:TARGET_COUNT]

    # Validate with Pydantic
    from useitup.schemas import Recipe as RecipeModel

    valid: list[dict] = []
    for r in sampled:
        try:
            RecipeModel.model_validate(r)
            valid.append(r)
        except Exception:
            pass

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(valid, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(valid)} recipes to {out_json}")

    cuisines_present = {r["cuisine"] for r in valid}
    print(f"Cuisines: {cuisines_present}")


if __name__ == "__main__":
    build()
