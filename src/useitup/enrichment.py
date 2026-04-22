"""Shared inference helpers for building recipe datasets.

Produces the category / dietary-tag / flavor / is_core labels that downstream
matching, CBR, and explanation code expect. Kept here (rather than in the
build scripts) so every dataset generator — curated or scraped — applies the
same rulebook.
"""

from __future__ import annotations

import re

# Rule order matters: earlier categories win ties. Multi-word keys MUST match
# as a phrase; single-word keys match only as a whole token (word boundaries)
# — this prevents "ham" in "champagne", "butter" in "butternut", etc.
#
# For ambiguous single words (e.g. "pepper" = black pepper OR bell pepper),
# keep the more common meaning in whichever category appears FIRST below.
# "pepper" lives in vegetable so bell pepper wins; black pepper is matched by
# the explicit "black pepper" phrase in spice.
_CATEGORY_RULES: list[tuple[list[str], str]] = [
    # Multi-word vegetable phrases FIRST so "bell pepper" / "cherry tomato" /
    # "butternut squash" don't get snared by single-word rules below.
    (["bell pepper", "red bell pepper", "yellow bell pepper", "green bell pepper",
      "cherry tomato", "cherry tomatoes", "grape tomato", "heirloom tomato",
      "roma tomato", "canned tomato", "sun-dried tomato", "butternut squash",
      "summer squash", "spaghetti squash", "acorn squash", "red onion",
      "yellow onion", "white onion", "green onion", "pearl onion", "sweet potato",
      "sweet potatoes", "green bean", "green beans", "french bean", "haricot vert",
      "haricots verts", "bok choy", "brussels sprout", "brussels sprouts",
      "snow pea", "snow peas", "sugar snap pea", "english cucumber",
      "baby spinach", "romaine lettuce", "iceberg lettuce", "boston lettuce",
      "swiss chard", "rainbow chard", "basil leaves", "bay leaves",
      "portobello mushroom", "cremini mushroom", "shiitake mushroom"],
     "vegetable"),
    # Multi-word grain phrases so "italian bread", "sourdough bread", etc. win
    (["italian bread", "french bread", "sourdough bread", "whole wheat bread",
      "whole-wheat bread", "gluten-free bread", "stale bread", "white bread",
      "corn tortilla", "flour tortilla", "pita bread", "rice noodles",
      "egg noodles", "whole-grain pita", "english muffin"], "grain"),
    # Multi-word dairy
    (["heavy cream", "half and half", "sour cream", "cream cheese",
      "cottage cheese", "goat cheese", "blue cheese", "swiss cheese",
      "feta cheese", "cheddar cheese", "parmesan cheese",
      "mozzarella cheese", "fresh mozzarella"], "dairy"),
    # Multi-word protein phrases so "chicken breast", "pork belly" etc. match
    (["chicken breast", "chicken thigh", "chicken wing", "chicken leg",
      "ground beef", "ground turkey", "ground pork", "ground chicken",
      "beef sirloin", "beef chuck", "pork belly", "pork shoulder",
      "pork chop", "lamb shank", "lamb shoulder", "turkey breast",
      "cannellini bean", "cannellini beans", "garbanzo bean", "garbanzo beans",
      "pinto bean", "pinto beans", "navy bean", "navy beans",
      "black bean", "black beans", "white bean", "white beans",
      "kidney bean", "kidney beans", "great northern bean", "refried beans",
      "canned chickpea", "canned chickpeas", "canned tuna", "canned salmon",
      "canned sardine", "canned sardines", "pine nut", "pine nuts"], "protein"),
    # Multi-word fat
    (["olive oil", "vegetable oil", "canola oil", "coconut oil", "sesame oil",
      "avocado oil", "peanut butter", "almond butter", "sunflower oil",
      "grapeseed oil", "extra virgin olive oil", "extra-virgin olive oil"], "fat"),
    # Multi-word condiment
    (["soy sauce", "fish sauce", "oyster sauce", "hot sauce", "bbq sauce",
      "barbecue sauce", "tomato sauce", "pasta sauce", "marinara sauce",
      "maple syrup", "brown sugar", "canned tomato", "canned tomatoes",
      "red wine", "white wine", "rice vinegar", "red wine vinegar",
      "white wine vinegar", "apple cider vinegar", "balsamic vinegar",
      "champagne vinegar", "sherry vinegar", "wine vinegar",
      "dijon mustard", "whole grain mustard", "tomato paste"], "condiment"),
    # Multi-word spice
    (["black pepper", "white pepper", "cracked pepper", "ground pepper",
      "freshly ground pepper", "red pepper flakes", "chili powder",
      "chili flakes", "curry powder", "garam masala", "five spice",
      "five-spice", "bay leaf", "mustard seed", "sea salt", "kosher salt",
      "table salt", "fine sea salt", "ras el hanout", "za'atar"], "spice"),
    # Single-word protein (whole-token match)
    (["chicken", "beef", "pork", "lamb", "salmon", "tuna", "shrimp", "turkey",
      "bacon", "sausage", "tofu", "tempeh", "lentil", "lentils", "chickpea",
      "chickpeas", "egg", "eggs", "cod", "tilapia", "crab", "lobster",
      "duck", "venison", "anchovy", "anchovies", "sardine", "sardines",
      "ham", "pepperoni", "prosciutto", "paneer", "pancetta",
      "bean", "beans", "almond", "almonds", "walnut", "walnuts",
      "pecan", "pecans", "cashew", "cashews", "pistachio", "pistachios",
      "hazelnut", "hazelnuts", "peanut", "peanuts", "macadamia",
      "scallop", "scallops", "oyster", "oysters", "mussel", "mussels",
      "clam", "clams", "trout", "mackerel", "halibut", "tilapia"], "protein"),
    # Single-word grain (whole-token match)
    (["flour", "rice", "pasta", "bread", "oat", "oats", "quinoa", "barley",
      "corn", "wheat", "noodle", "noodles", "tortilla", "tortillas",
      "couscous", "polenta", "rye", "bulgur", "farro", "millet",
      "spaghetti", "fettuccine", "penne", "rigatoni", "lasagna", "gnocchi",
      "ramen", "udon", "soba", "orzo", "risotto", "breadcrumb", "breadcrumbs",
      "cracker", "crackers", "pita", "naan", "baguette", "bun", "buns",
      "wrap", "crouton", "croutons", "loaf", "biscuit", "biscuits",
      "muffin", "muffins", "bagel", "bagels", "ciabatta", "boule",
      "cornbread", "cornmeal", "semolina", "sourdough"], "grain"),
    # Single-word dairy (whole-token match)
    (["milk", "cheese", "butter", "cream", "yogurt", "ghee",
      "mozzarella", "parmesan", "cheddar", "feta", "ricotta", "brie",
      "buttermilk", "gruyere", "pecorino", "mascarpone"], "dairy"),
    # Single-word spice
    (["cumin", "paprika", "turmeric", "cinnamon", "oregano", "thyme",
      "rosemary", "ginger", "cayenne", "coriander", "cardamom", "nutmeg",
      "clove", "cloves", "saffron", "sumac", "allspice", "fennel",
      "chili", "chilli", "chipotle", "harissa"], "spice"),
    # Single-word fat
    (["lard", "margarine", "tahini", "shortening"], "fat"),
    # Single-word vegetable (whole-token match; includes fruits — we have no
    # `fruit` category, and treating them as vegetable-class produce is
    # defensible for matching/coverage purposes).
    (["onion", "garlic", "tomato", "tomatoes", "carrot", "carrots",
      "celery", "spinach", "broccoli", "pepper", "peppers", "mushroom",
      "mushrooms", "zucchini", "eggplant", "cucumber", "cucumbers",
      "lettuce", "kale", "cabbage", "potato", "potatoes", "pea", "peas",
      "asparagus", "cauliflower", "leek", "leeks", "shallot", "shallots",
      "scallion", "scallions", "beet", "beets", "radish", "radishes",
      "artichoke", "artichokes", "pumpkin", "squash", "chard",
      "parsley", "cilantro", "mint", "lime", "limes", "lemon", "lemons",
      "arugula", "olive", "olives", "kalamata", "pickle", "pickles",
      "jalapeño", "jalapeno", "avocado", "avocados", "parsnip", "parsnips",
      "turnip", "turnips", "rutabaga", "fennel", "endive", "watercress",
      "tomatillo", "tomatillos", "okra", "yam", "yams",
      "apple", "apples", "pear", "pears", "berry", "berries",
      "strawberry", "strawberries", "raspberry", "raspberries",
      "blueberry", "blueberries", "blackberry", "blackberries",
      "cherry", "cherries", "peach", "peaches", "nectarine", "nectarines",
      "plum", "plums", "mango", "mangoes", "pineapple", "pineapples",
      "grape", "grapes", "melon", "watermelon", "cantaloupe",
      "honeydew", "kiwi", "banana", "bananas", "orange", "oranges",
      "tangerine", "clementine", "grapefruit", "fig", "figs",
      "date", "dates", "apricot", "apricots", "coconut",
      "basil", "marjoram", "dill", "tarragon", "chive", "chives", "sage",
      "amaranth", "sprout", "sprouts", "cress"], "vegetable"),
    # Single-word condiment (last, so specific items above win)
    (["ketchup", "mustard", "mayo", "mayonnaise", "vinegar",
      "worcestershire", "hoisin", "sriracha", "miso", "teriyaki",
      "salsa", "relish", "honey", "molasses", "jam", "chutney",
      "hummus", "pesto", "stock", "broth", "wine", "sugar", "capers",
      "caper"], "condiment"),
    # Standalone salt (garnish, but needs a home) — kept in spice via its
    # multi-word forms above; bare "salt" here catches the plain case.
    (["salt"], "spice"),
]

_MEAT_KW = {
    "chicken", "beef", "pork", "lamb", "salmon", "tuna", "shrimp", "turkey",
    "bacon", "sausage", "cod", "tilapia", "crab", "lobster", "duck",
    "venison", "anchovy", "sardine", "ham", "pepperoni", "prosciutto",
    "meat", "fish", "seafood", "pancetta",
}
_DAIRY_KW = {
    "milk", "cheese", "butter", "cream", "yogurt", "sour cream", "ghee",
    "mozzarella", "parmesan", "cheddar", "feta", "ricotta", "brie",
    "heavy cream", "buttermilk", "gruyere", "paneer", "pecorino",
}
_GLUTEN_KW = {
    "flour", "bread", "pasta", "wheat", "barley", "rye", "noodle",
    "spaghetti", "fettuccine", "penne", "rigatoni", "lasagna", "breadcrumb",
    "cracker", "bulgur", "couscous", "semolina", "pita", "naan", "baguette",
    "bun", "wrap", "ramen", "udon", "tortilla wrap", "phyllo",
}
_NUT_KW = {
    "almond", "walnut", "pecan", "cashew", "pistachio", "hazelnut",
    "pine nut", "macadamia", "peanut",
}

_FLAVOR_RULES: list[tuple[list[str], str]] = [
    (["chili", "jalapeño", "jalapeno", "sriracha", "cayenne", "hot sauce",
      "red pepper flakes", "chipotle", "habanero", "tabasco", "harissa"], "spicy"),
    (["soy sauce", "miso", "parmesan", "mushroom", "anchovy", "fish sauce",
      "worcestershire", "nutritional yeast", "tomato paste"], "umami"),
    (["salt", "broth", "stock", "thyme", "rosemary", "oregano", "garlic",
      "onion", "sage", "herbes"], "savory"),
    (["sugar", "honey", "maple syrup", "molasses", "chocolate", "vanilla",
      "brown sugar", "caramel", "cinnamon"], "sweet"),
    (["lemon", "lime", "vinegar", "tamarind", "yogurt", "sour cream",
      "balsamic"], "sour"),
    (["smoked", "bbq", "chipotle", "liquid smoke", "paprika", "bacon",
      "charred", "grilled"], "smoky"),
    (["mint", "basil", "cilantro", "dill", "cucumber", "parsley",
      "lemon zest", "arugula"], "fresh"),
    (["cream", "butter", "coconut milk", "heavy cream", "cheese",
      "avocado", "olive oil", "tahini", "ghee"], "rich"),
]

_GARNISH_KEYWORDS = frozenset({
    "salt", "pepper", "parsley", "cilantro", "chives", "scallion", "green onion",
    "sesame seeds", "red pepper flakes", "lemon wedges", "lime wedges",
    "black pepper", "to taste", "garnish",
})

_TOKEN_STOPWORDS = frozenset({
    "fresh", "dried", "ground", "crushed", "chopped", "minced", "large", "small",
    "extra", "virgin", "boneless", "skinless", "whole", "plain", "low", "fat",
    "optional", "for", "serving", "and", "or", "the",
})


_WORD_RE = re.compile(r"[a-z0-9']+")


def _name_tokens(name: str) -> list[str]:
    return _WORD_RE.findall(name.lower())


def _matches_keyword(name_low: str, name_tokens: list[str], kw: str) -> bool:
    """Keyword hit with word-boundary semantics.

    Multi-word keywords: phrase substring match (e.g. "bell pepper").
    Single-word keywords: only hit when the token appears whole in the name,
    so "ham" does not match "champagne" and "butter" does not match
    "butternut". Simple plurals are tolerated (token == kw or token == kw+s
    or stripping trailing s).
    """
    if " " in kw or "-" in kw:
        return kw in name_low
    for tok in name_tokens:
        if tok == kw:
            return True
        if tok.endswith("s") and tok[:-1] == kw:
            return True
        if tok.endswith("es") and tok[:-2] == kw:
            return True
    return False


def classify_category(name: str) -> str:
    """Map an ingredient name to one of the 8 IngredientCategory values.

    Uses word-boundary matching for single-word keywords to avoid spurious
    substring hits ("ham" in "champagne", "butter" in "butternut").
    Multi-word phrases are checked first so compound names like
    "butternut squash" resolve to vegetable before `butter` can fire.
    """
    low = name.lower()
    tokens = _name_tokens(name)
    for keywords, cat in _CATEGORY_RULES:
        for kw in keywords:
            if _matches_keyword(low, tokens, kw):
                return cat
    return "other"


def _tokens(text: str) -> set[str]:
    return {
        t for t in re.split(r"[^a-z0-9]+", text.lower())
        if t and t not in _TOKEN_STOPWORDS
    }


def _is_garnish(name: str) -> bool:
    low = name.lower().strip()
    if low in _GARNISH_KEYWORDS:
        return True
    return any(kw in low for kw in _GARNISH_KEYWORDS)


_NOMINAL_KEYWORDS = frozenset({
    "water", "ice", "ice cube", "ice cubes",
    "salt", "kosher salt", "sea salt", "fine sea salt", "table salt",
    "pepper", "black pepper", "white pepper", "ground pepper",
    "cracked pepper", "freshly ground pepper",
    "sugar", "to taste",
})


def _is_nominal(name: str) -> bool:
    """True for water / salt / pepper / sugar-only ingredients.

    Excluded from weight-share accounting in `infer_is_core` so that e.g.
    473 g of water doesn't dilute the core-staple signal.
    """
    low = name.lower().strip()
    return low in _NOMINAL_KEYWORDS or any(
        low == f"{prefix} salt" or low == f"{prefix} pepper"
        for prefix in ("kosher", "sea", "fine", "coarse", "table", "fine-grain")
    )


def compute_weight_shares(
    raw_ingredients: list[dict],
) -> list[float | None]:
    """Given Edamam-style ingredient dicts with optional `weight` (grams),
    return each item's share of the non-nominal total.

    Returns `None` entries where weight data is missing. Water/salt/pepper/sugar
    are excluded from the denominator so bread's 200 g isn't dwarfed by
    500 g of water.
    """
    weights: list[float] = []
    nominals: list[bool] = []
    for raw in raw_ingredients:
        w = raw.get("weight") if isinstance(raw, dict) else None
        try:
            weights.append(float(w) if w is not None else 0.0)
        except (TypeError, ValueError):
            weights.append(0.0)
        nominals.append(_is_nominal(
            (raw.get("food") if isinstance(raw, dict) else "") or ""
        ))
    total = sum(w for w, nom in zip(weights, nominals) if not nom)
    if total <= 0:
        return [None] * len(raw_ingredients)
    shares: list[float | None] = []
    for w, nom in zip(weights, nominals):
        if nom or w <= 0:
            shares.append(None)
        else:
            shares.append(w / total)
    return shares


def infer_is_core(
    name: str,
    category: str,
    idx: int,
    recipe_name: str,
    *,
    weight_share: float | None = None,
    total_ingredients: int | None = None,
) -> bool:
    """Decide whether an ingredient is core/essential for this recipe.

    Priority order:

      1. Garnishes (salt, pepper, parsley, …) are never core.
      2. Weight share ≥ 10 % of the non-nominal recipe total → core (when
         weight data is available, from Edamam).
      3. Tokens overlapping the recipe name → core (e.g. "chicken" in
         "Chicken Tikka Masala").
      4. Proteins and grains are always core.
      5. The first non-garnish ingredient is core — Edamam orders ingredients
         by centrality, and plain category rules miss "other"-class staples
         like croutons or amaranth.
      6. Positions 1–2, if in a major category, are core.
    """
    low = name.lower()
    if _is_garnish(low):
        return False
    if weight_share is not None and weight_share >= 0.10:
        return True
    if _tokens(name) & _tokens(recipe_name):
        return True
    if category in {"protein", "grain"}:
        return True
    if idx == 0:
        return True
    if idx < 2 and category in {"protein", "grain", "vegetable", "dairy", "fat"}:
        return True
    if idx < 3 and category in {"fat", "dairy", "vegetable"}:
        return True
    return False


def infer_dietary_tags(ingredient_names: list[str], total_minutes: int) -> list[str]:
    """Derive DietaryTag list from ingredient names + total cooking time."""
    lowered = [n.lower() for n in ingredient_names]
    combined = " ".join(lowered)

    def has(kws: set[str]) -> bool:
        return any(any(kw in n for kw in kws) for n in lowered)

    has_meat = has(_MEAT_KW)
    has_dairy = has(_DAIRY_KW)
    has_gluten = has(_GLUTEN_KW)
    has_nuts = has(_NUT_KW)

    tags: list[str] = []
    if not has_meat and not has_dairy and "honey" not in combined and "egg" not in combined:
        tags.append("vegan")
    if not has_meat:
        tags.append("vegetarian")
    if not has_dairy:
        tags.append("dairy-free")
    if not has_gluten:
        tags.append("gluten-free")
    if not has_nuts:
        tags.append("nut-free")
    if total_minutes <= 30:
        tags.append("quick")

    protein_keywords = [
        "chicken", "beef", "pork", "salmon", "tuna", "shrimp", "turkey",
        "eggs", "tofu", "lentils", "chickpea", "black bean", "paneer",
        "pancetta", "white beans", "lamb",
    ]
    protein_count = sum(
        1 for n in lowered
        if any(kw in n for kw in protein_keywords)
    )
    if protein_count >= 2:
        tags.append("high-protein")

    if not has_gluten and "rice" not in combined and "potato" not in combined \
            and "corn tortilla" not in combined and "sugar" not in combined:
        tags.append("low-carb")

    return list(dict.fromkeys(tags))


def infer_flavor(ingredient_names: list[str]) -> list[str]:
    """Derive FlavorTag list from ingredient names."""
    text = " ".join(ingredient_names).lower()
    out: list[str] = []
    for kws, tag in _FLAVOR_RULES:
        if any(kw in text for kw in kws):
            out.append(tag)
    return list(dict.fromkeys(out))
