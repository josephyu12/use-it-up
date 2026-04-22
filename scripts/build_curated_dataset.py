"""
Build data/recipes.json from a curated, hand-authored recipe list.

No external dataset required — recipe data is embedded below in a compact form.
Ingredient categories, dietary tags, and flavor profiles are derived from the
same rulebooks used at runtime, so downstream code stays consistent.

Run:
    python scripts/build_curated_dataset.py
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_JSON = ROOT / "data" / "recipes_curated.json"

sys.path.insert(0, str(ROOT / "src"))

from useitup.enrichment import infer_is_core  # noqa: E402
from useitup.schemas import Recipe  # noqa: E402

# --------------------------------------------------------------------------
# Rulebooks (kept local so the script stays self-contained)
# --------------------------------------------------------------------------

_CATEGORY_RULES: list[tuple[list[str], str]] = [
    (["chicken", "beef", "pork", "lamb", "salmon", "tuna", "shrimp", "turkey",
      "bacon", "sausage", "tofu", "tempeh", "lentil", "chickpea", "black bean",
      "kidney bean", "white bean", "egg", "cod", "tilapia", "crab", "lobster",
      "duck", "venison", "anchovy", "sardine", "ham", "pepperoni", "prosciutto",
      "paneer", "pancetta"], "protein"),
    (["flour", "rice", "pasta", "bread", "oat", "quinoa", "barley", "corn",
      "wheat", "noodle", "tortilla", "couscous", "polenta", "rye", "bulgur",
      "spaghetti", "fettuccine", "penne", "rigatoni", "lasagna", "gnocchi",
      "ramen", "udon", "soba", "breadcrumb", "cracker", "pita", "naan",
      "baguette", "bun", "wrap"], "grain"),
    (["milk", "cheese", "butter", "cream", "yogurt", "sour cream", "ghee",
      "mozzarella", "parmesan", "cheddar", "feta", "ricotta", "brie",
      "heavy cream", "half and half", "buttermilk", "gruyere"], "dairy"),
    (["salt", "pepper", "cumin", "paprika", "turmeric", "cinnamon", "oregano",
      "thyme", "basil", "rosemary", "ginger", "chili", "cayenne", "coriander",
      "cardamom", "nutmeg", "clove", "bay leaf", "saffron", "sumac",
      "five spice", "allspice", "fennel", "mustard seed", "curry powder",
      "garam masala", "za'atar", "harissa", "ras el hanout"], "spice"),
    (["olive oil", "vegetable oil", "canola oil", "coconut oil", "sesame oil",
      "avocado", "lard", "margarine", "peanut butter", "almond butter",
      "tahini"], "fat"),
    (["onion", "garlic", "tomato", "carrot", "celery", "spinach", "broccoli",
      "pepper", "mushroom", "zucchini", "eggplant", "cucumber", "lettuce",
      "kale", "cabbage", "potato", "sweet potato", "pea", "asparagus",
      "green bean", "cauliflower", "leek", "shallot", "scallion", "scallions",
      "green onion", "beet", "radish", "artichoke", "pumpkin", "squash",
      "chard", "bok choy", "parsley", "cilantro", "mint", "lime", "lemon",
      "basil leaves", "arugula", "olive", "olives", "kalamata", "pickles",
      "cherry tomatoes", "red onion", "jalapeño", "jalapeno", "avocado",
      "corn", "bell pepper", "chickpeas"], "vegetable"),
    (["soy sauce", "ketchup", "mustard", "mayo", "mayonnaise", "vinegar",
      "hot sauce", "worcestershire", "fish sauce", "oyster sauce", "hoisin",
      "sriracha", "miso", "teriyaki", "salsa", "relish", "bbq sauce",
      "honey", "maple syrup", "molasses", "jam", "chutney", "hummus",
      "pesto", "tomato sauce", "pasta sauce", "stock", "broth", "wine",
      "canned tomatoes", "sugar", "brown sugar"], "condiment"),
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
    "heavy cream", "buttermilk", "gruyere", "paneer",
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


def classify_category(name: str) -> str:
    low = name.lower()
    for keywords, cat in _CATEGORY_RULES:
        if any(kw in low for kw in keywords):
            return cat
    return "other"


def infer_dietary_tags(names: list[str], total_min: int) -> list[str]:
    lowered = [n.lower() for n in names]
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
    if total_min <= 30:
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


def infer_flavor(names: list[str]) -> list[str]:
    text = " ".join(names).lower()
    out: list[str] = []
    for kws, tag in _FLAVOR_RULES:
        if any(kw in text for kw in kws):
            out.append(tag)
    return list(dict.fromkeys(out))


# --------------------------------------------------------------------------
# Compact recipe format
# --------------------------------------------------------------------------

@dataclass
class Ing:
    name: str
    qty: float | None = None
    unit: str | None = None


def i(name: str, qty: float | None = None, unit: str | None = None) -> Ing:
    return Ing(name, qty, unit)


@dataclass
class R:
    name: str
    cuisine: str
    prep: int
    cook: int
    difficulty: int
    ings: list[Ing]
    steps: list[str]
    calories: float
    protein_g: float
    carbs_g: float
    fat_g: float


# --------------------------------------------------------------------------
# Recipe corpus
# --------------------------------------------------------------------------

RECIPES: list[R] = [
    # ---------- Italian (12) ----------
    R("Spaghetti Aglio e Olio", "Italian", 5, 15, 1, [
        i("spaghetti", 400, "g"), i("garlic", 6, "cloves"),
        i("olive oil", 4, "tbsp"), i("red pepper flakes", 1, "tsp"),
        i("parsley"), i("salt"),
    ], [
        "Boil spaghetti in salted water until al dente.",
        "Gently fry sliced garlic in olive oil until golden.",
        "Add red pepper flakes; toss drained pasta in the oil.",
        "Finish with chopped parsley and serve.",
    ], 520, 14, 72, 18),

    R("Spaghetti Carbonara", "Italian", 10, 15, 2, [
        i("spaghetti", 400, "g"), i("pancetta", 150, "g"),
        i("eggs", 3), i("parmesan", 80, "g"),
        i("black pepper"), i("salt"),
    ], [
        "Boil spaghetti until al dente; reserve 1 cup pasta water.",
        "Crisp pancetta in a wide pan, then remove from heat.",
        "Whisk eggs with grated parmesan and pepper.",
        "Toss hot pasta with pancetta, then off-heat egg mix, loosening with pasta water.",
    ], 680, 32, 72, 28),

    R("Fettuccine Alfredo", "Italian", 5, 15, 2, [
        i("fettuccine", 400, "g"), i("butter", 80, "g"),
        i("heavy cream", 250, "ml"), i("parmesan", 100, "g"),
        i("black pepper"), i("salt"),
    ], [
        "Cook fettuccine until al dente.",
        "Melt butter with cream over low heat.",
        "Toss pasta with sauce and grated parmesan off heat.",
        "Season with black pepper; serve immediately.",
    ], 760, 22, 68, 44),

    R("Penne Arrabbiata", "Italian", 5, 20, 1, [
        i("penne", 400, "g"), i("canned tomatoes", 400, "g"),
        i("garlic", 4, "cloves"), i("red pepper flakes", 2, "tsp"),
        i("olive oil", 3, "tbsp"), i("parsley"), i("salt"),
    ], [
        "Sauté garlic and chilli flakes in olive oil.",
        "Add canned tomatoes; simmer 15 minutes.",
        "Cook penne until al dente.",
        "Toss pasta in sauce; top with parsley.",
    ], 490, 14, 78, 13),

    R("Cacio e Pepe", "Italian", 5, 12, 2, [
        i("spaghetti", 400, "g"), i("pecorino", 120, "g"),
        i("black pepper", 2, "tsp"), i("salt"),
    ], [
        "Cook spaghetti in minimal salted water.",
        "Toast cracked pepper in a dry pan.",
        "Add pasta water to the pan, then pasta.",
        "Remove from heat, stir in pecorino until creamy.",
    ], 560, 24, 72, 18),

    R("Classic Lasagna", "Italian", 30, 60, 4, [
        i("lasagna noodles", 12), i("ground beef", 500, "g"),
        i("tomato sauce", 600, "g"), i("ricotta", 400, "g"),
        i("mozzarella", 300, "g"), i("parmesan", 80, "g"),
        i("onion", 1), i("garlic", 3, "cloves"), i("oregano"), i("salt"),
    ], [
        "Brown beef with onion and garlic; add tomato sauce, simmer 20 min.",
        "Layer noodles, meat sauce, ricotta, mozzarella in a dish.",
        "Top with parmesan; bake 40 min at 375°F.",
        "Rest 10 minutes before slicing.",
    ], 780, 44, 52, 42),

    R("Margherita Pizza", "Italian", 20, 12, 3, [
        i("pizza dough", 500, "g"), i("tomato sauce", 200, "g"),
        i("fresh mozzarella", 250, "g"), i("fresh basil"),
        i("olive oil", 2, "tbsp"), i("salt"),
    ], [
        "Stretch dough on a floured surface.",
        "Spread tomato sauce and torn mozzarella.",
        "Bake at 500°F for 10–12 minutes.",
        "Finish with basil and olive oil.",
    ], 620, 22, 78, 22),

    R("Caprese Salad", "Italian", 10, 0, 1, [
        i("fresh mozzarella", 250, "g"), i("tomato", 3),
        i("fresh basil"), i("olive oil", 3, "tbsp"),
        i("balsamic vinegar", 1, "tbsp"), i("salt"), i("black pepper"),
    ], [
        "Slice mozzarella and tomatoes into rounds.",
        "Alternate slices with basil leaves.",
        "Drizzle with oil and balsamic.",
        "Season and serve immediately.",
    ], 350, 18, 8, 26),

    R("Mushroom Risotto", "Italian", 10, 30, 3, [
        i("arborio rice", 300, "g"), i("mushroom", 400, "g"),
        i("onion", 1), i("garlic", 2, "cloves"),
        i("vegetable broth", 1, "l"), i("white wine", 120, "ml"),
        i("parmesan", 60, "g"), i("butter", 40, "g"), i("olive oil", 2, "tbsp"),
    ], [
        "Sauté mushrooms in olive oil until browned; set aside.",
        "Cook onion and garlic in butter; toast rice 2 min.",
        "Deglaze with wine; add broth a ladle at a time, stirring, for 20 min.",
        "Fold in mushrooms and parmesan.",
    ], 560, 18, 82, 16),

    R("Minestrone Soup", "Italian", 15, 35, 2, [
        i("white beans", 400, "g"), i("carrot", 2),
        i("celery", 2, "stalks"), i("onion", 1), i("garlic", 2, "cloves"),
        i("canned tomatoes", 400, "g"), i("zucchini", 1),
        i("small pasta", 100, "g"), i("vegetable broth", 1, "l"),
        i("olive oil", 2, "tbsp"), i("parmesan"), i("oregano"),
    ], [
        "Sauté onion, carrot, celery, garlic in olive oil.",
        "Add tomatoes and broth; simmer 20 min.",
        "Add zucchini, beans, pasta; cook 10 more min.",
        "Top with parmesan.",
    ], 360, 16, 58, 8),

    R("Chicken Parmesan", "Italian", 15, 25, 3, [
        i("chicken breast", 600, "g"), i("breadcrumbs", 120, "g"),
        i("eggs", 2), i("flour", 80, "g"),
        i("tomato sauce", 400, "g"), i("mozzarella", 200, "g"),
        i("parmesan", 60, "g"), i("olive oil", 3, "tbsp"), i("salt"),
    ], [
        "Pound chicken thin; dredge in flour, egg, breadcrumbs.",
        "Pan-fry until golden on both sides.",
        "Top with tomato sauce and cheeses; bake 10 min at 400°F.",
        "Serve over pasta.",
    ], 720, 58, 42, 34),

    R("Pesto Gnocchi", "Italian", 5, 10, 1, [
        i("potato gnocchi", 500, "g"), i("pesto", 120, "g"),
        i("parmesan", 40, "g"), i("pine nuts", 30, "g"),
        i("olive oil", 1, "tbsp"), i("salt"),
    ], [
        "Boil gnocchi until they float.",
        "Drain, reserving a splash of water.",
        "Toss with pesto, olive oil, and a bit of pasta water.",
        "Top with parmesan and pine nuts.",
    ], 540, 16, 82, 18),

    # ---------- Mexican (10) ----------
    R("Black Bean Tacos", "Mexican", 5, 10, 1, [
        i("black beans", 400, "g"), i("corn tortillas", 8),
        i("avocado", 2), i("salsa", 200, "g"),
        i("cumin", 1, "tsp"), i("lime", 1), i("cilantro"),
    ], [
        "Heat black beans with cumin and salt.",
        "Warm tortillas in a dry skillet.",
        "Fill with beans, avocado, salsa, cilantro.",
        "Squeeze lime on top.",
    ], 410, 16, 60, 12),

    R("Chicken Quesadillas", "Mexican", 10, 10, 1, [
        i("chicken breast", 400, "g"), i("flour tortillas", 4),
        i("cheddar cheese", 200, "g"), i("bell pepper", 1),
        i("onion", 1), i("cumin"), i("olive oil", 1, "tbsp"),
        i("salsa"), i("sour cream"),
    ], [
        "Sauté chicken with onion, pepper, cumin until cooked.",
        "Layer filling and cheese between tortillas.",
        "Griddle 2 min per side until cheese melts.",
        "Slice and serve with salsa and sour cream.",
    ], 650, 42, 48, 32),

    R("Beef Burritos", "Mexican", 15, 20, 2, [
        i("ground beef", 500, "g"), i("flour tortillas", 4),
        i("rice", 150, "g"), i("black beans", 300, "g"),
        i("cheddar cheese", 120, "g"), i("salsa"),
        i("cumin"), i("paprika"), i("onion", 1),
    ], [
        "Brown beef with onion, cumin, paprika.",
        "Cook rice separately.",
        "Fill tortillas with rice, beans, beef, cheese, salsa.",
        "Roll tightly and sear seam-side down.",
    ], 780, 42, 78, 28),

    R("Chicken Enchiladas", "Mexican", 20, 25, 3, [
        i("chicken breast", 500, "g"), i("corn tortillas", 8),
        i("enchilada sauce", 500, "g"), i("cheddar cheese", 200, "g"),
        i("onion", 1), i("cumin"), i("cilantro"), i("sour cream"),
    ], [
        "Simmer chicken with onion and cumin; shred.",
        "Dip tortillas in warm enchilada sauce.",
        "Fill with chicken, roll, place in baking dish.",
        "Top with remaining sauce and cheese; bake 20 min at 375°F.",
    ], 620, 42, 48, 26),

    R("Chicken Fajitas", "Mexican", 15, 15, 2, [
        i("chicken breast", 500, "g"), i("bell pepper", 2),
        i("onion", 1), i("flour tortillas", 6),
        i("cumin"), i("paprika"), i("chili powder"),
        i("olive oil", 2, "tbsp"), i("lime", 1),
    ], [
        "Slice chicken and vegetables into strips.",
        "Sear chicken with spices in a hot pan.",
        "Add peppers and onion; cook 5 minutes.",
        "Squeeze lime and serve in warm tortillas.",
    ], 540, 38, 44, 20),

    R("Guacamole and Chips", "Mexican", 10, 0, 1, [
        i("avocado", 3), i("lime", 1), i("red onion", 0.5),
        i("cilantro"), i("jalapeño", 1), i("salt"),
        i("tortilla chips", 200, "g"),
    ], [
        "Mash avocado with lime juice and salt.",
        "Fold in diced onion, jalapeño, and cilantro.",
        "Serve with tortilla chips.",
    ], 420, 6, 38, 26),

    R("Huevos Rancheros", "Mexican", 5, 10, 1, [
        i("eggs", 4), i("corn tortillas", 4),
        i("black beans", 300, "g"), i("salsa", 300, "g"),
        i("avocado", 1), i("cilantro"), i("olive oil", 1, "tbsp"),
    ], [
        "Warm tortillas and beans.",
        "Fry eggs sunny-side up in olive oil.",
        "Layer tortilla, beans, egg, salsa, avocado.",
        "Top with cilantro.",
    ], 480, 22, 44, 22),

    R("Pozole Rojo", "Mexican", 20, 60, 3, [
        i("pork shoulder", 800, "g"), i("hominy", 500, "g"),
        i("dried chili", 4), i("onion", 1), i("garlic", 4, "cloves"),
        i("oregano"), i("lime", 2), i("cilantro"), i("salt"),
    ], [
        "Simmer pork with onion, garlic 45 min.",
        "Blend soaked chilies with a ladle of broth.",
        "Return to pot; add hominy, simmer 15 min.",
        "Serve with lime, oregano, cilantro.",
    ], 520, 36, 42, 20),

    R("Chicken Tortilla Soup", "Mexican", 15, 30, 2, [
        i("chicken breast", 400, "g"), i("canned tomatoes", 400, "g"),
        i("black beans", 200, "g"), i("corn", 200, "g"),
        i("chicken broth", 1, "l"), i("cumin"), i("paprika"),
        i("onion", 1), i("garlic", 2, "cloves"), i("tortilla chips"),
        i("avocado", 1), i("lime", 1),
    ], [
        "Sauté onion and garlic; add spices.",
        "Add broth, tomatoes, beans, chicken; simmer 20 min.",
        "Shred chicken; add corn.",
        "Top bowls with chips, avocado, lime.",
    ], 520, 36, 42, 18),

    R("Carne Asada Tacos", "Mexican", 15, 10, 2, [
        i("flank steak", 600, "g"), i("corn tortillas", 8),
        i("lime", 2), i("garlic", 3, "cloves"), i("cumin"),
        i("onion", 1), i("cilantro"), i("olive oil", 2, "tbsp"),
    ], [
        "Marinate steak in lime, garlic, cumin 30 min.",
        "Grill steak to medium-rare; rest and slice.",
        "Char tortillas; fill with steak.",
        "Top with onion and cilantro.",
    ], 560, 44, 36, 22),

    # ---------- Asian (12) ----------
    R("Miso Ramen", "Asian", 15, 25, 3, [
        i("ramen noodles", 200, "g"), i("miso paste", 3, "tbsp"),
        i("soy sauce", 2, "tbsp"), i("soft-boiled egg", 2),
        i("tofu", 150, "g"), i("bok choy", 200, "g"),
        i("sesame oil", 1, "tsp"), i("green onion"),
        i("chicken broth", 800, "ml"),
    ], [
        "Whisk miso and soy into hot broth.",
        "Cook noodles separately.",
        "Blanch bok choy; halve the eggs.",
        "Assemble bowls with noodles, broth, toppings.",
    ], 540, 28, 62, 18),

    R("Pad Thai", "Asian", 15, 15, 2, [
        i("rice noodles", 300, "g"), i("shrimp", 300, "g"),
        i("eggs", 2), i("bean sprouts", 150, "g"),
        i("fish sauce", 2, "tbsp"), i("tamarind paste", 2, "tbsp"),
        i("brown sugar", 2, "tbsp"), i("peanuts", 60, "g"),
        i("garlic", 3, "cloves"), i("lime", 1), i("green onion"),
    ], [
        "Soak rice noodles in warm water.",
        "Sear shrimp with garlic; push aside and scramble eggs.",
        "Add drained noodles and sauce; toss.",
        "Finish with sprouts, peanuts, lime, scallions.",
    ], 620, 32, 78, 22),

    R("Vegetable Fried Rice", "Asian", 10, 10, 1, [
        i("cooked rice", 500, "g"), i("eggs", 2),
        i("peas", 100, "g"), i("carrot", 1),
        i("soy sauce", 2, "tbsp"), i("sesame oil", 1, "tsp"),
        i("green onion"), i("garlic", 2, "cloves"),
    ], [
        "Scramble eggs, set aside.",
        "Stir-fry carrot and peas with garlic.",
        "Add cold rice; break up and toss.",
        "Stir in eggs, soy sauce, sesame oil, scallions.",
    ], 420, 12, 68, 10),

    R("Beef Stir Fry", "Asian", 15, 10, 2, [
        i("flank steak", 500, "g"), i("broccoli", 300, "g"),
        i("bell pepper", 1), i("soy sauce", 3, "tbsp"),
        i("oyster sauce", 2, "tbsp"), i("garlic", 3, "cloves"),
        i("ginger", 1, "tbsp"), i("sesame oil", 1, "tsp"),
        i("cornstarch", 1, "tbsp"),
    ], [
        "Slice beef thin; toss with cornstarch and soy.",
        "Sear beef in a hot wok; remove.",
        "Stir-fry garlic, ginger, vegetables.",
        "Return beef; add sauces and toss.",
    ], 520, 44, 22, 26),

    R("Chicken Teriyaki", "Asian", 10, 15, 2, [
        i("chicken thigh", 600, "g"), i("soy sauce", 4, "tbsp"),
        i("mirin", 2, "tbsp"), i("sugar", 2, "tbsp"),
        i("ginger", 1, "tbsp"), i("rice", 300, "g"),
        i("sesame seeds"), i("green onion"),
    ], [
        "Cook rice.",
        "Pan-sear chicken skin-side down until crisp.",
        "Add soy, mirin, sugar, ginger; reduce to glaze.",
        "Slice; serve over rice with sesame and scallions.",
    ], 640, 42, 62, 18),

    R("Vietnamese Pho", "Asian", 20, 60, 3, [
        i("beef bones", 1, "kg"), i("rice noodles", 300, "g"),
        i("sirloin", 300, "g"), i("star anise", 2),
        i("cinnamon stick", 1), i("onion", 1), i("ginger", 50, "g"),
        i("fish sauce", 2, "tbsp"), i("bean sprouts"), i("basil"),
        i("lime"), i("jalapeño"),
    ], [
        "Char onion and ginger; simmer with bones and spices 2+ hrs.",
        "Cook rice noodles; slice raw sirloin thin.",
        "Ladle boiling broth over noodles and beef to cook.",
        "Garnish with sprouts, basil, lime, chili.",
    ], 520, 32, 62, 14),

    R("Kung Pao Chicken", "Asian", 15, 10, 2, [
        i("chicken breast", 500, "g"), i("peanuts", 80, "g"),
        i("dried chili", 6), i("soy sauce", 3, "tbsp"),
        i("rice vinegar", 1, "tbsp"), i("sugar", 1, "tsp"),
        i("garlic", 3, "cloves"), i("ginger", 1, "tbsp"),
        i("sesame oil", 1, "tsp"), i("green onion"),
    ], [
        "Marinate diced chicken in soy and cornstarch.",
        "Sear chicken; remove from wok.",
        "Toast chilies and aromatics; add sauce.",
        "Return chicken with peanuts and scallions.",
    ], 560, 42, 22, 30),

    R("Sesame Noodles", "Asian", 10, 10, 1, [
        i("soba noodles", 300, "g"), i("soy sauce", 3, "tbsp"),
        i("tahini", 2, "tbsp"), i("sesame oil", 2, "tbsp"),
        i("rice vinegar", 1, "tbsp"), i("honey", 1, "tbsp"),
        i("garlic", 2, "cloves"), i("cucumber", 1), i("green onion"),
        i("sesame seeds"),
    ], [
        "Cook soba; rinse cold.",
        "Whisk sauce ingredients.",
        "Toss noodles with sauce, cucumber, scallions.",
        "Top with sesame seeds.",
    ], 440, 14, 68, 14),

    R("Kimchi Fried Rice", "Asian", 10, 10, 1, [
        i("cooked rice", 500, "g"), i("kimchi", 250, "g"),
        i("eggs", 2), i("soy sauce", 2, "tbsp"),
        i("sesame oil", 1, "tsp"), i("green onion"), i("gochujang", 1, "tbsp"),
    ], [
        "Sauté kimchi 3 minutes.",
        "Add rice and gochujang; toss.",
        "Fry eggs sunny-side up.",
        "Top rice with eggs and scallions.",
    ], 460, 14, 72, 12),

    R("Thai Green Curry", "Asian", 10, 20, 2, [
        i("chicken thigh", 500, "g"), i("green curry paste", 3, "tbsp"),
        i("coconut milk", 400, "ml"), i("fish sauce", 1, "tbsp"),
        i("sugar", 1, "tsp"), i("bamboo shoots", 200, "g"),
        i("thai basil"), i("lime"), i("rice", 300, "g"),
    ], [
        "Cook rice.",
        "Fry curry paste in a splash of coconut milk.",
        "Add chicken; cook 5 min.",
        "Pour remaining coconut milk, fish sauce, sugar; simmer 10 min.",
    ], 680, 38, 62, 32),

    R("Beef Bulgogi", "Asian", 30, 10, 2, [
        i("ribeye", 500, "g"), i("soy sauce", 4, "tbsp"),
        i("brown sugar", 2, "tbsp"), i("sesame oil", 1, "tbsp"),
        i("garlic", 3, "cloves"), i("pear", 0.5),
        i("green onion"), i("rice", 300, "g"), i("sesame seeds"),
    ], [
        "Slice beef paper-thin; marinate 30 min in soy, sugar, pear puree, garlic.",
        "Sear in a very hot pan 1–2 min per batch.",
        "Serve over rice with scallions and sesame seeds.",
    ], 640, 42, 58, 22),

    R("Egg Drop Soup", "Asian", 5, 10, 1, [
        i("chicken broth", 1, "l"), i("eggs", 2),
        i("cornstarch", 1, "tbsp"), i("soy sauce", 1, "tbsp"),
        i("green onion"), i("ginger", 1, "tsp"), i("sesame oil"),
    ], [
        "Bring broth to a simmer with ginger and soy.",
        "Stir in cornstarch slurry to thicken.",
        "Drizzle in beaten eggs while stirring.",
        "Finish with sesame oil and scallions.",
    ], 150, 10, 8, 8),

    # ---------- Indian (10) ----------
    R("Chicken Tikka Masala", "Indian", 20, 30, 3, [
        i("chicken breast", 600, "g"), i("tomato sauce", 400, "g"),
        i("heavy cream", 200, "ml"), i("garam masala", 2, "tsp"),
        i("turmeric", 1, "tsp"), i("garlic", 4, "cloves"),
        i("ginger", 1, "tbsp"), i("onion", 1), i("butter", 2, "tbsp"),
    ], [
        "Marinate chicken in yogurt and spices 30 min.",
        "Sear chicken until just cooked; set aside.",
        "Sauté onion in butter; add tomato sauce and spices.",
        "Return chicken; stir in cream and simmer 5 min.",
    ], 680, 45, 22, 42),

    R("Lentil Dal", "Indian", 10, 25, 2, [
        i("red lentils", 300, "g"), i("onion", 1),
        i("tomato", 2), i("turmeric", 1, "tsp"),
        i("cumin", 1, "tsp"), i("coriander", 1, "tsp"),
        i("ghee", 2, "tbsp"), i("garlic", 3, "cloves"),
        i("ginger", 1, "tsp"),
    ], [
        "Sauté onion, garlic, ginger in ghee.",
        "Add spices; cook 1 min.",
        "Add lentils, tomatoes, water; simmer 20 min until thick.",
        "Season and serve with rice.",
    ], 390, 22, 54, 10),

    R("Palak Paneer", "Indian", 15, 20, 3, [
        i("spinach", 400, "g"), i("paneer", 300, "g"),
        i("heavy cream", 100, "ml"), i("onion", 1),
        i("garlic", 3, "cloves"), i("ginger", 1, "tbsp"),
        i("garam masala", 1, "tsp"), i("cumin"), i("ghee", 2, "tbsp"),
    ], [
        "Blanch spinach; blend smooth.",
        "Sauté onion, garlic, ginger, spices in ghee.",
        "Stir in spinach puree and cream.",
        "Add cubed paneer; simmer 5 min.",
    ], 480, 24, 16, 36),

    R("Chana Masala", "Indian", 10, 25, 2, [
        i("chickpeas", 500, "g"), i("canned tomatoes", 400, "g"),
        i("onion", 1), i("garlic", 3, "cloves"), i("ginger", 1, "tbsp"),
        i("garam masala", 2, "tsp"), i("cumin"), i("coriander"),
        i("olive oil", 2, "tbsp"),
    ], [
        "Sauté onion with garlic and ginger.",
        "Add spices; cook 1 min.",
        "Add tomatoes and chickpeas; simmer 20 min.",
        "Adjust salt and serve with rice or naan.",
    ], 380, 16, 56, 10),

    R("Chicken Biryani", "Indian", 25, 40, 4, [
        i("chicken thigh", 600, "g"), i("basmati rice", 400, "g"),
        i("yogurt", 200, "g"), i("onion", 2),
        i("ginger", 1, "tbsp"), i("garlic", 4, "cloves"),
        i("garam masala", 2, "tsp"), i("saffron"), i("ghee", 3, "tbsp"),
        i("mint"), i("cilantro"),
    ], [
        "Marinate chicken in yogurt and spices 30 min.",
        "Fry sliced onions until crisp.",
        "Layer rice, chicken, onions, herbs in a pot.",
        "Steam covered 30 min on low heat.",
    ], 760, 48, 76, 24),

    R("Butter Chicken", "Indian", 20, 25, 3, [
        i("chicken breast", 600, "g"), i("tomato sauce", 400, "g"),
        i("heavy cream", 200, "ml"), i("butter", 60, "g"),
        i("garam masala", 2, "tsp"), i("fenugreek", 1, "tsp"),
        i("garlic", 4, "cloves"), i("ginger", 1, "tbsp"),
    ], [
        "Marinate chicken in yogurt and spices.",
        "Grill or broil chicken until lightly charred.",
        "Simmer tomato sauce with butter and spices.",
        "Add chicken and cream; simmer 10 min.",
    ], 720, 46, 18, 48),

    R("Aloo Gobi", "Indian", 10, 25, 2, [
        i("potato", 500, "g"), i("cauliflower", 400, "g"),
        i("onion", 1), i("tomato", 2), i("garlic", 3, "cloves"),
        i("ginger", 1, "tbsp"), i("turmeric"), i("cumin"),
        i("garam masala"), i("olive oil", 2, "tbsp"),
    ], [
        "Sauté onion with cumin.",
        "Add garlic, ginger, tomatoes, spices.",
        "Add potatoes and cauliflower; cover and cook 20 min.",
        "Finish with garam masala.",
    ], 320, 10, 52, 10),

    R("Tandoori Chicken", "Indian", 20, 30, 3, [
        i("chicken drumsticks", 8), i("yogurt", 250, "g"),
        i("garam masala", 2, "tsp"), i("paprika", 1, "tbsp"),
        i("garlic", 4, "cloves"), i("ginger", 1, "tbsp"),
        i("lemon", 1), i("olive oil", 2, "tbsp"),
    ], [
        "Marinate chicken in yogurt, spices, garlic 4+ hours.",
        "Roast at 450°F for 25–30 min turning once.",
        "Squeeze lemon before serving.",
    ], 540, 48, 6, 32),

    R("Saag Aloo", "Indian", 10, 20, 2, [
        i("spinach", 400, "g"), i("potato", 400, "g"),
        i("onion", 1), i("garlic", 3, "cloves"),
        i("turmeric"), i("cumin"), i("ghee", 2, "tbsp"),
    ], [
        "Boil cubed potato until tender.",
        "Sauté onion and garlic in ghee; add spices.",
        "Add potatoes and spinach; cook until wilted.",
    ], 320, 10, 44, 12),

    R("Vegetable Curry", "Indian", 15, 25, 2, [
        i("mixed vegetables", 500, "g"), i("coconut milk", 400, "ml"),
        i("onion", 1), i("garlic", 3, "cloves"),
        i("ginger", 1, "tbsp"), i("curry powder", 2, "tbsp"),
        i("tomato", 2), i("olive oil", 2, "tbsp"),
    ], [
        "Sauté onion, garlic, ginger.",
        "Add curry powder and tomatoes.",
        "Stir in vegetables and coconut milk; simmer 20 min.",
    ], 420, 10, 40, 26),

    # ---------- American (10) ----------
    R("BBQ Pulled Pork Sandwiches", "American", 20, 180, 3, [
        i("pork shoulder", 1.5, "lbs"), i("bbq sauce", 300, "g"),
        i("brioche buns", 4), i("coleslaw mix", 200, "g"),
        i("apple cider vinegar", 2, "tbsp"), i("brown sugar", 2, "tbsp"),
        i("paprika", 1, "tbsp"),
    ], [
        "Rub pork with sugar, paprika, salt; roast at 300°F for 3 hrs.",
        "Shred and toss with BBQ sauce and vinegar.",
        "Pile onto buns with coleslaw.",
    ], 620, 38, 54, 28),

    R("Classic Cheeseburger", "American", 10, 10, 2, [
        i("ground beef", 500, "g"), i("hamburger buns", 4),
        i("cheddar cheese", 4, "slices"), i("lettuce"),
        i("tomato", 1), i("onion", 1), i("pickles"),
        i("ketchup"), i("mustard"),
    ], [
        "Form 4 patties; season with salt and pepper.",
        "Sear 3 min per side; add cheese in last minute.",
        "Toast buns; build with lettuce, tomato, onion, pickles.",
        "Finish with condiments.",
    ], 700, 38, 42, 38),

    R("Southern Fried Chicken", "American", 20, 20, 3, [
        i("chicken thigh", 800, "g"), i("buttermilk", 400, "ml"),
        i("flour", 250, "g"), i("paprika", 1, "tbsp"),
        i("garlic powder", 1, "tsp"), i("cayenne", 1, "tsp"),
        i("vegetable oil", 1, "l"),
    ], [
        "Soak chicken in buttermilk 2+ hours.",
        "Dredge in seasoned flour.",
        "Fry at 350°F for 12–15 min until golden.",
        "Rest on a rack; sprinkle with salt.",
    ], 780, 46, 48, 42),

    R("Mac and Cheese", "American", 10, 20, 2, [
        i("elbow pasta", 400, "g"), i("cheddar cheese", 300, "g"),
        i("milk", 500, "ml"), i("butter", 40, "g"),
        i("flour", 30, "g"), i("breadcrumbs", 60, "g"),
    ], [
        "Cook pasta.",
        "Make a roux with butter and flour; whisk in milk.",
        "Stir in cheese until smooth.",
        "Toss with pasta, top with breadcrumbs, bake 10 min.",
    ], 720, 28, 68, 38),

    R("Beef Chili", "American", 15, 45, 2, [
        i("ground beef", 500, "g"), i("kidney beans", 400, "g"),
        i("canned tomatoes", 400, "g"), i("onion", 1),
        i("garlic", 3, "cloves"), i("chili powder", 2, "tbsp"),
        i("cumin", 1, "tsp"), i("bell pepper", 1),
    ], [
        "Brown beef with onion.",
        "Add garlic, peppers, spices.",
        "Add tomatoes and beans; simmer 40 min.",
    ], 560, 36, 38, 26),

    R("Cobb Salad", "American", 15, 10, 1, [
        i("romaine lettuce", 1), i("chicken breast", 300, "g"),
        i("bacon", 4, "slices"), i("hard boiled eggs", 2),
        i("avocado", 1), i("blue cheese", 80, "g"),
        i("tomato", 2), i("ranch dressing"),
    ], [
        "Cook chicken and bacon; chop.",
        "Arrange toppings in rows over lettuce.",
        "Drizzle with ranch.",
    ], 580, 42, 12, 38),

    R("Grilled Cheese Sandwich", "American", 3, 7, 1, [
        i("bread", 4, "slices"), i("cheddar cheese", 150, "g"),
        i("butter", 30, "g"),
    ], [
        "Butter bread on both sides.",
        "Layer cheese between slices.",
        "Griddle 3 min per side until golden.",
    ], 520, 22, 36, 32),

    R("Buffalo Wings", "American", 10, 40, 2, [
        i("chicken wings", 1, "kg"), i("hot sauce", 120, "ml"),
        i("butter", 60, "g"), i("garlic powder"),
        i("blue cheese dressing"), i("celery"),
    ], [
        "Roast wings at 400°F for 35 min.",
        "Melt butter with hot sauce and garlic.",
        "Toss hot wings in sauce.",
        "Serve with celery and blue cheese.",
    ], 620, 48, 4, 42),

    R("Meatloaf", "American", 15, 60, 2, [
        i("ground beef", 800, "g"), i("breadcrumbs", 80, "g"),
        i("eggs", 2), i("onion", 1), i("ketchup", 120, "g"),
        i("worcestershire", 1, "tbsp"), i("garlic powder"),
    ], [
        "Mix beef, breadcrumbs, eggs, onion, spices.",
        "Shape loaf; top with ketchup.",
        "Bake at 375°F for 55 min.",
    ], 540, 38, 24, 30),

    R("Buttermilk Pancakes", "American", 10, 15, 1, [
        i("flour", 250, "g"), i("buttermilk", 400, "ml"),
        i("eggs", 2), i("sugar", 2, "tbsp"),
        i("baking powder", 2, "tsp"), i("butter", 40, "g"),
        i("maple syrup"),
    ], [
        "Whisk dry ingredients; whisk wet separately.",
        "Fold together without overmixing.",
        "Griddle ¼-cup portions 2 min per side.",
        "Serve with butter and syrup.",
    ], 420, 12, 68, 12),

    # ---------- Mediterranean (8) ----------
    R("Greek Salad", "Mediterranean", 10, 0, 1, [
        i("cucumber", 1), i("tomato", 3), i("red onion", 0.5),
        i("feta cheese", 150, "g"), i("kalamata olives", 80, "g"),
        i("olive oil", 3, "tbsp"), i("oregano", 1, "tsp"),
        i("lemon juice", 2, "tbsp"),
    ], [
        "Chop cucumber, tomatoes, red onion.",
        "Add olives and cubed feta.",
        "Dress with oil, lemon, oregano.",
    ], 320, 10, 14, 26),

    R("Classic Hummus", "Mediterranean", 5, 0, 1, [
        i("chickpeas", 400, "g"), i("tahini", 3, "tbsp"),
        i("lemon", 1), i("garlic", 2, "cloves"),
        i("olive oil", 3, "tbsp"), i("cumin", 0.5, "tsp"), i("salt"),
    ], [
        "Blend chickpeas, tahini, lemon, garlic until smooth.",
        "Thin with ice water if needed.",
        "Drizzle with oil and sprinkle cumin.",
    ], 260, 8, 22, 16),

    R("Falafel Wraps", "Mediterranean", 20, 15, 3, [
        i("chickpeas", 400, "g"), i("parsley"),
        i("onion", 1), i("garlic", 3, "cloves"),
        i("cumin"), i("coriander"), i("flour", 2, "tbsp"),
        i("pita bread", 4), i("tahini"), i("cucumber"), i("tomato", 2),
    ], [
        "Blend chickpeas, herbs, onion, garlic, spices coarsely.",
        "Form balls; fry in hot oil 4 min until brown.",
        "Stuff pitas with falafel, veg, tahini.",
    ], 540, 18, 72, 20),

    R("Tabbouleh", "Mediterranean", 15, 10, 1, [
        i("bulgur", 150, "g"), i("parsley"), i("mint"),
        i("tomato", 3), i("cucumber", 1), i("lemon juice", 3, "tbsp"),
        i("olive oil", 3, "tbsp"), i("scallions"), i("salt"),
    ], [
        "Soak bulgur in hot water 10 min; drain.",
        "Chop parsley, mint, tomato, cucumber finely.",
        "Toss everything with lemon and olive oil.",
    ], 280, 6, 38, 12),

    R("Mediterranean Chickpea Salad", "Mediterranean", 10, 0, 1, [
        i("chickpeas", 500, "g"), i("cucumber", 1),
        i("tomato", 2), i("red onion", 0.5),
        i("feta cheese", 100, "g"), i("kalamata olives", 60, "g"),
        i("olive oil", 3, "tbsp"), i("lemon", 1), i("oregano"),
    ], [
        "Combine chickpeas with chopped vegetables.",
        "Add feta and olives.",
        "Dress with oil, lemon, oregano.",
    ], 420, 16, 42, 20),

    R("Spanakopita", "Mediterranean", 20, 35, 3, [
        i("phyllo dough", 250, "g"), i("spinach", 500, "g"),
        i("feta cheese", 250, "g"), i("eggs", 2),
        i("onion", 1), i("garlic", 2, "cloves"),
        i("olive oil", 4, "tbsp"), i("dill"),
    ], [
        "Sauté onion and garlic; add spinach until wilted.",
        "Mix with feta, eggs, dill; cool.",
        "Layer phyllo with oil; spread filling; fold.",
        "Bake at 375°F for 30 min.",
    ], 460, 18, 32, 28),

    R("Chicken Souvlaki", "Mediterranean", 20, 15, 2, [
        i("chicken thigh", 600, "g"), i("olive oil", 3, "tbsp"),
        i("lemon", 1), i("oregano"), i("garlic", 3, "cloves"),
        i("pita bread", 4), i("tzatziki"), i("tomato", 2),
        i("red onion", 0.5),
    ], [
        "Marinate diced chicken in oil, lemon, garlic, oregano 30 min.",
        "Skewer and grill 10–12 min.",
        "Serve in pita with tzatziki and vegetables.",
    ], 620, 42, 48, 26),

    R("Baba Ganoush", "Mediterranean", 10, 30, 2, [
        i("eggplant", 2), i("tahini", 3, "tbsp"),
        i("lemon juice", 2, "tbsp"), i("garlic", 2, "cloves"),
        i("olive oil", 2, "tbsp"), i("cumin"), i("salt"),
    ], [
        "Roast whole eggplants 30 min at 400°F until soft.",
        "Scoop flesh; blend with tahini, lemon, garlic, cumin.",
        "Drizzle with olive oil.",
    ], 180, 4, 16, 12),

    # ---------- French (8) ----------
    R("Classic Omelette", "French", 5, 5, 2, [
        i("eggs", 3), i("butter", 1, "tbsp"),
        i("cheddar cheese", 40, "g"), i("chives"),
        i("salt"), i("pepper"),
    ], [
        "Beat eggs with salt and pepper.",
        "Melt butter in non-stick pan over medium heat.",
        "Pour in eggs; fold edges as they set.",
        "Add cheese, fold omelette, serve with chives.",
    ], 320, 20, 2, 25),

    R("Quiche Lorraine", "French", 20, 40, 3, [
        i("pie crust", 1), i("bacon", 200, "g"),
        i("eggs", 4), i("heavy cream", 300, "ml"),
        i("gruyere", 150, "g"), i("onion", 1), i("nutmeg"),
    ], [
        "Blind-bake crust 10 min.",
        "Crisp bacon with onion; drain.",
        "Whisk eggs, cream, cheese, nutmeg.",
        "Pour into crust with bacon; bake 30 min at 375°F.",
    ], 560, 22, 28, 42),

    R("French Onion Soup", "French", 10, 50, 3, [
        i("onion", 6), i("butter", 40, "g"),
        i("beef broth", 1, "l"), i("white wine", 120, "ml"),
        i("baguette", 1), i("gruyere", 150, "g"),
        i("thyme"), i("bay leaf"),
    ], [
        "Caramelize sliced onions in butter 30–40 min.",
        "Deglaze with wine; add broth and herbs; simmer 15 min.",
        "Top bowls with bread and cheese; broil until bubbly.",
    ], 420, 14, 42, 22),

    R("Ratatouille", "French", 20, 35, 2, [
        i("eggplant", 1), i("zucchini", 2), i("bell pepper", 2),
        i("tomato", 4), i("onion", 1), i("garlic", 4, "cloves"),
        i("olive oil", 4, "tbsp"), i("herbes de provence", 1, "tbsp"),
    ], [
        "Sauté onion and garlic in olive oil.",
        "Add diced peppers and eggplant; cook 10 min.",
        "Add zucchini, tomatoes, herbs; simmer 20 min.",
    ], 280, 6, 32, 16),

    R("Croque Monsieur", "French", 10, 10, 2, [
        i("bread", 4, "slices"), i("ham", 200, "g"),
        i("gruyere", 150, "g"), i("butter", 40, "g"),
        i("flour", 20, "g"), i("milk", 200, "ml"), i("nutmeg"),
    ], [
        "Make a béchamel with butter, flour, milk, nutmeg.",
        "Build sandwiches with ham, cheese, béchamel.",
        "Top with cheese; bake 8 min at 400°F.",
    ], 620, 32, 38, 36),

    R("Niçoise Salad", "French", 20, 10, 2, [
        i("tuna", 300, "g"), i("potato", 300, "g"),
        i("green beans", 200, "g"), i("hard boiled eggs", 3),
        i("tomato", 3), i("kalamata olives", 80, "g"),
        i("anchovy", 6), i("olive oil", 4, "tbsp"), i("lemon", 1),
    ], [
        "Boil potatoes and green beans until tender.",
        "Arrange lettuce, potatoes, beans, tomatoes, eggs on platter.",
        "Top with tuna, olives, anchovies.",
        "Drizzle with oil and lemon.",
    ], 480, 36, 28, 24),

    R("Coq au Vin", "French", 20, 75, 4, [
        i("chicken thigh", 1, "kg"), i("bacon", 150, "g"),
        i("mushroom", 300, "g"), i("pearl onions", 200, "g"),
        i("red wine", 500, "ml"), i("chicken broth", 300, "ml"),
        i("flour", 30, "g"), i("thyme"), i("bay leaf"), i("butter", 30, "g"),
    ], [
        "Crisp bacon; brown chicken in the fat.",
        "Deglaze with wine; add broth, herbs; braise 1 hr.",
        "Sauté mushrooms and onions; add to pot.",
        "Thicken with flour-butter paste.",
    ], 720, 52, 18, 42),

    R("Crêpes", "French", 15, 15, 2, [
        i("flour", 200, "g"), i("eggs", 3),
        i("milk", 400, "ml"), i("butter", 40, "g"),
        i("sugar", 1, "tbsp"), i("salt"),
    ], [
        "Whisk flour, eggs, milk, sugar, salt, melted butter.",
        "Rest batter 30 min.",
        "Cook thin crêpes in a buttered pan 1 min per side.",
        "Fill with jam or Nutella.",
    ], 380, 12, 54, 12),

    # ---------- Middle Eastern (8) ----------
    R("Shakshuka", "Middle Eastern", 10, 20, 2, [
        i("eggs", 4), i("canned tomatoes", 400, "g"),
        i("red bell pepper", 1), i("onion", 1),
        i("garlic", 3, "cloves"), i("cumin"), i("paprika"),
        i("olive oil", 2, "tbsp"), i("feta cheese", 60, "g"),
    ], [
        "Sauté onion and pepper in olive oil.",
        "Add garlic and spices; cook 1 min.",
        "Pour in tomatoes; simmer 10 min.",
        "Crack eggs into wells; cover until set. Top with feta.",
    ], 340, 18, 20, 20),

    R("Chicken Shawarma", "Middle Eastern", 30, 20, 3, [
        i("chicken thigh", 700, "g"), i("yogurt", 100, "g"),
        i("garlic", 4, "cloves"), i("lemon", 1),
        i("cumin"), i("paprika"), i("cinnamon"),
        i("pita bread", 4), i("tahini"), i("tomato", 2),
    ], [
        "Marinate chicken with yogurt, garlic, lemon, spices 1+ hr.",
        "Roast or grill 18 min, turning once.",
        "Slice; serve in pita with tahini and tomatoes.",
    ], 620, 48, 38, 28),

    R("Beef Kebabs", "Middle Eastern", 20, 12, 2, [
        i("sirloin", 600, "g"), i("bell pepper", 2),
        i("onion", 1), i("olive oil", 3, "tbsp"),
        i("lemon", 1), i("garlic", 3, "cloves"),
        i("cumin"), i("paprika"),
    ], [
        "Cube beef and vegetables; marinate with oil, lemon, spices 30 min.",
        "Skewer alternately.",
        "Grill 10–12 min, turning often.",
    ], 460, 40, 12, 26),

    R("Moroccan Tagine", "Middle Eastern", 20, 75, 3, [
        i("lamb shoulder", 800, "g"), i("onion", 2),
        i("garlic", 4, "cloves"), i("ras el hanout", 2, "tsp"),
        i("apricot", 150, "g"), i("chickpeas", 300, "g"),
        i("canned tomatoes", 400, "g"), i("olive oil", 2, "tbsp"),
        i("cilantro"),
    ], [
        "Brown lamb in batches.",
        "Sauté onion, garlic, spices.",
        "Add tomatoes, apricots, chickpeas; braise 1 hr.",
        "Top with cilantro.",
    ], 720, 48, 42, 36),

    R("Couscous Salad", "Middle Eastern", 15, 5, 1, [
        i("couscous", 250, "g"), i("cucumber", 1),
        i("tomato", 2), i("parsley"), i("mint"),
        i("lemon", 1), i("olive oil", 3, "tbsp"),
        i("chickpeas", 200, "g"),
    ], [
        "Pour boiling water over couscous; cover 5 min, then fluff.",
        "Chop vegetables and herbs.",
        "Toss everything with lemon and oil.",
    ], 380, 12, 62, 10),

    R("Lamb Kofta", "Middle Eastern", 15, 12, 2, [
        i("ground lamb", 600, "g"), i("onion", 1),
        i("parsley"), i("garlic", 3, "cloves"),
        i("cumin"), i("coriander"), i("cinnamon"),
        i("pita bread", 4), i("tzatziki"),
    ], [
        "Mix lamb with minced onion, herbs, spices.",
        "Form into oblong patties on skewers.",
        "Grill 10–12 min.",
        "Serve in pita with tzatziki.",
    ], 580, 38, 28, 36),

    R("Fattoush Salad", "Middle Eastern", 15, 5, 1, [
        i("pita bread", 2), i("cucumber", 1),
        i("tomato", 3), i("red onion", 0.5),
        i("parsley"), i("mint"), i("sumac", 1, "tsp"),
        i("lemon", 1), i("olive oil", 3, "tbsp"),
    ], [
        "Toast pita until crisp; break into pieces.",
        "Chop vegetables and herbs.",
        "Toss with lemon, oil, sumac; add pita.",
    ], 280, 6, 36, 12),

    R("Mujaddara", "Middle Eastern", 10, 35, 2, [
        i("green lentils", 250, "g"), i("rice", 200, "g"),
        i("onion", 3), i("olive oil", 4, "tbsp"),
        i("cumin", 1, "tsp"), i("salt"),
    ], [
        "Simmer lentils 20 min; add rice and cook 15 more min.",
        "Caramelize sliced onions deeply in oil (20 min).",
        "Fold onions into lentils and rice; season with cumin.",
    ], 440, 16, 68, 14),

    # ---------- Expansion set (24) ----------
    R("Pasta Puttanesca", "Italian", 10, 20, 2, [
        i("spaghetti", 400, "g"), i("canned tomatoes", 400, "g"),
        i("anchovy", 4), i("kalamata olives", 100, "g"),
        i("capers", 2, "tbsp"), i("garlic", 4, "cloves"),
        i("olive oil", 3, "tbsp"), i("red pepper flakes"),
    ], [
        "Cook spaghetti until al dente.",
        "Sauté garlic and anchovy in olive oil until fragrant.",
        "Add tomatoes, olives, capers, and chilli; simmer 10 minutes.",
        "Toss pasta in the sauce and serve.",
    ], 510, 18, 74, 14),

    R("Mushroom Risotto", "Italian", 15, 30, 3, [
        i("arborio rice", 300, "g"), i("mushroom", 400, "g"),
        i("onion", 1), i("garlic", 2, "cloves"),
        i("parmesan", 80, "g"), i("butter", 40, "g"),
        i("vegetable broth", 1, "l"), i("olive oil", 2, "tbsp"),
    ], [
        "Sauté onion, garlic, and mushrooms until browned.",
        "Toast arborio rice in the pan for 2 minutes.",
        "Add broth gradually, stirring until creamy and tender.",
        "Finish with butter and parmesan.",
    ], 540, 14, 70, 18),

    R("Pasta Primavera", "Italian", 15, 20, 2, [
        i("penne", 400, "g"), i("zucchini", 2), i("cherry tomatoes", 250, "g"),
        i("broccoli", 250, "g"), i("garlic", 3, "cloves"),
        i("olive oil", 3, "tbsp"), i("parmesan", 70, "g"), i("basil"),
    ], [
        "Cook penne until al dente.",
        "Sauté zucchini, broccoli, and garlic in olive oil.",
        "Add tomatoes and cook until just softened.",
        "Toss with pasta, parmesan, and basil.",
    ], 480, 16, 68, 14),

    R("Chicken Quesadillas", "Mexican", 10, 12, 1, [
        i("flour tortillas", 4), i("chicken breast", 400, "g"),
        i("cheddar cheese", 180, "g"), i("salsa", 120, "g"),
        i("red onion", 0.5), i("bell pepper", 1),
        i("cumin"), i("olive oil", 1, "tbsp"),
    ], [
        "Cook seasoned chicken with onion and pepper until done.",
        "Fill tortillas with chicken, cheese, and salsa.",
        "Toast in a skillet until crisp on both sides.",
        "Slice and serve hot.",
    ], 620, 38, 42, 28),

    R("Chilaquiles Rojos", "Mexican", 10, 15, 2, [
        i("corn tortillas", 8), i("eggs", 4), i("salsa", 300, "g"),
        i("onion", 0.5), i("avocado", 1), i("feta cheese", 60, "g"),
        i("olive oil", 2, "tbsp"), i("cilantro"),
    ], [
        "Fry tortilla pieces until crisp.",
        "Warm salsa with sliced onion until simmering.",
        "Toss chips lightly in salsa, then top with fried eggs.",
        "Finish with avocado, feta, and cilantro.",
    ], 460, 18, 40, 24),

    R("Carne Asada Bowls", "Mexican", 20, 15, 2, [
        i("sirloin", 500, "g"), i("rice", 250, "g"),
        i("black beans", 250, "g"), i("avocado", 1),
        i("lime", 1), i("cilantro"), i("garlic", 3, "cloves"),
        i("cumin"), i("olive oil", 2, "tbsp"),
    ], [
        "Marinate steak with garlic, lime, cumin, and oil.",
        "Cook rice and warm black beans.",
        "Sear steak until charred and slice thinly.",
        "Build bowls with rice, beans, steak, avocado, and cilantro.",
    ], 640, 42, 48, 26),

    R("Teriyaki Salmon Bowls", "Asian", 15, 15, 2, [
        i("salmon", 500, "g"), i("rice", 250, "g"),
        i("soy sauce", 4, "tbsp"), i("brown sugar", 1, "tbsp"),
        i("ginger", 1, "tbsp"), i("garlic", 2, "cloves"),
        i("broccoli", 250, "g"), i("sesame oil", 1, "tbsp"),
    ], [
        "Cook rice until fluffy.",
        "Simmer soy sauce, sugar, ginger, and garlic into a glaze.",
        "Roast or pan-sear salmon and brush with teriyaki glaze.",
        "Serve over rice with broccoli.",
    ], 590, 36, 44, 24),

    R("Kimchi Fried Rice", "Asian", 10, 12, 1, [
        i("rice", 300, "g"), i("kimchi", 200, "g"),
        i("eggs", 2), i("green onion"), i("soy sauce", 2, "tbsp"),
        i("sesame oil", 1, "tbsp"), i("garlic", 2, "cloves"),
    ], [
        "Fry garlic and kimchi in sesame oil.",
        "Add cold rice and soy sauce; stir-fry until hot.",
        "Top with fried eggs and green onion.",
    ], 430, 14, 52, 16),

    R("Garlic Tofu Stir-Fry", "Asian", 15, 12, 1, [
        i("tofu", 400, "g"), i("broccoli", 250, "g"),
        i("bell pepper", 1), i("soy sauce", 3, "tbsp"),
        i("garlic", 4, "cloves"), i("ginger", 1, "tbsp"),
        i("sesame oil", 1, "tbsp"), i("rice", 250, "g"),
    ], [
        "Press tofu and sear until golden.",
        "Stir-fry garlic, ginger, broccoli, and pepper.",
        "Add soy sauce and return tofu to the pan.",
        "Serve with rice.",
    ], 470, 22, 46, 18),

    R("Chana Masala", "Indian", 15, 25, 2, [
        i("chickpeas", 400, "g"), i("onion", 1), i("canned tomatoes", 400, "g"),
        i("garlic", 3, "cloves"), i("ginger", 1, "tbsp"),
        i("garam masala", 2, "tsp"), i("cumin", 1, "tsp"),
        i("olive oil", 2, "tbsp"), i("cilantro"),
    ], [
        "Sauté onion, garlic, and ginger until soft.",
        "Add spices and cook 1 minute.",
        "Add tomatoes and chickpeas; simmer 20 minutes.",
        "Finish with cilantro.",
    ], 390, 14, 48, 12),

    R("Palak Paneer", "Indian", 15, 20, 2, [
        i("paneer", 300, "g"), i("spinach", 400, "g"),
        i("onion", 1), i("garlic", 3, "cloves"),
        i("ginger", 1, "tbsp"), i("garam masala", 2, "tsp"),
        i("cumin", 1, "tsp"), i("heavy cream", 60, "ml"),
        i("olive oil", 2, "tbsp"),
    ], [
        "Cook onion, garlic, and ginger until soft.",
        "Wilt spinach and blend into a smooth sauce.",
        "Return sauce to the pan with spices and cream.",
        "Fold in paneer cubes and simmer briefly.",
    ], 510, 22, 18, 36),

    R("Vegetable Biryani", "Indian", 20, 30, 3, [
        i("basmati rice", 300, "g"), i("cauliflower", 250, "g"),
        i("peas", 150, "g"), i("carrot", 2), i("onion", 1),
        i("garlic", 3, "cloves"), i("ginger", 1, "tbsp"),
        i("garam masala", 2, "tsp"), i("yogurt", 120, "g"),
    ], [
        "Parboil basmati rice until nearly tender.",
        "Cook onion, garlic, ginger, and vegetables with spices.",
        "Layer rice and vegetables in a pot with yogurt.",
        "Steam on low heat until fragrant.",
    ], 480, 12, 74, 10),

    R("Turkey Chili", "American", 15, 35, 2, [
        i("ground turkey", 500, "g"), i("kidney beans", 300, "g"),
        i("canned tomatoes", 400, "g"), i("onion", 1),
        i("garlic", 3, "cloves"), i("chili powder", 2, "tbsp"),
        i("cumin", 1, "tsp"), i("olive oil", 1, "tbsp"),
    ], [
        "Brown turkey with onion and garlic.",
        "Add spices and toast briefly.",
        "Add tomatoes and beans; simmer 30 minutes.",
        "Serve with your favorite toppings.",
    ], 430, 34, 28, 16),

    R("Baked Mac and Cheese", "American", 15, 25, 2, [
        i("macaroni", 350, "g"), i("cheddar cheese", 250, "g"),
        i("milk", 400, "ml"), i("butter", 50, "g"),
        i("flour", 35, "g"), i("breadcrumb", 60, "g"),
        i("black pepper"), i("salt"),
    ], [
        "Cook macaroni until just tender.",
        "Make a cheese sauce with butter, flour, milk, and cheddar.",
        "Combine pasta and sauce in a baking dish.",
        "Top with breadcrumbs and bake until golden.",
    ], 690, 24, 70, 32),

    R("Sheet Pan Sausage and Veg", "American", 10, 30, 1, [
        i("sausage", 500, "g"), i("potatoes", 500, "g"),
        i("broccoli", 300, "g"), i("red onion", 1),
        i("olive oil", 2, "tbsp"), i("paprika", 1, "tsp"),
        i("garlic", 2, "cloves"),
    ], [
        "Toss potatoes, broccoli, onion, and garlic with oil and paprika.",
        "Spread on a sheet pan with sausage pieces.",
        "Roast at 425°F until browned and cooked through.",
    ], 560, 24, 34, 32),

    R("Greek Chicken Bowls", "Mediterranean", 20, 20, 2, [
        i("chicken breast", 500, "g"), i("rice", 250, "g"),
        i("cucumber", 1), i("tomato", 2), i("feta cheese", 80, "g"),
        i("lemon", 1), i("olive oil", 3, "tbsp"),
        i("garlic", 3, "cloves"), i("oregano"),
    ], [
        "Marinate chicken with lemon, garlic, oregano, and oil.",
        "Cook rice and grill the chicken.",
        "Slice cucumber and tomato for the bowl base.",
        "Assemble bowls with feta on top.",
    ], 560, 40, 38, 22),

    R("Lemon Herb Chicken Orzo", "Mediterranean", 10, 20, 2, [
        i("chicken breast", 450, "g"), i("orzo", 300, "g"),
        i("lemon", 1), i("garlic", 3, "cloves"),
        i("spinach", 150, "g"), i("olive oil", 2, "tbsp"),
        i("feta cheese", 60, "g"), i("parsley"),
    ], [
        "Brown chicken pieces in olive oil.",
        "Add garlic and orzo, then cook briefly.",
        "Pour in water and simmer until orzo is tender.",
        "Finish with lemon, spinach, feta, and parsley.",
    ], 540, 34, 46, 20),

    R("White Bean Vegetable Soup", "Mediterranean", 15, 30, 1, [
        i("white beans", 400, "g"), i("carrot", 2), i("celery", 2, "stalks"),
        i("onion", 1), i("garlic", 3, "cloves"), i("spinach", 150, "g"),
        i("vegetable broth", 1, "l"), i("olive oil", 2, "tbsp"),
    ], [
        "Cook onion, carrot, celery, and garlic in olive oil.",
        "Add beans and broth; simmer 20 minutes.",
        "Stir in spinach just before serving.",
    ], 310, 14, 38, 10),

    R("Quiche Lorraine", "French", 20, 35, 3, [
        i("pie crust", 1), i("eggs", 4), i("bacon", 150, "g"),
        i("gruyere", 120, "g"), i("heavy cream", 250, "ml"),
        i("onion", 0.5), i("black pepper"), i("salt"),
    ], [
        "Blind-bake the crust until lightly set.",
        "Cook bacon and onion until fragrant.",
        "Whisk eggs with cream and cheese.",
        "Fill crust and bake until just set.",
    ], 620, 24, 24, 44),

    R("Salmon en Papillote", "French", 15, 18, 2, [
        i("salmon", 500, "g"), i("asparagus", 250, "g"),
        i("lemon", 1), i("butter", 30, "g"),
        i("shallot", 1), i("olive oil", 1, "tbsp"), i("thyme"),
    ], [
        "Layer salmon, asparagus, shallot, lemon, and butter on parchment.",
        "Fold packets tightly.",
        "Bake until the salmon is tender and flaky.",
    ], 410, 34, 10, 24),

    R("Mushroom Gruyere Tartine", "French", 10, 15, 1, [
        i("bread", 4, "slices"), i("mushroom", 300, "g"),
        i("gruyere", 120, "g"), i("shallot", 1),
        i("butter", 20, "g"), i("thyme"), i("olive oil", 1, "tbsp"),
    ], [
        "Sauté mushrooms and shallot with butter and thyme.",
        "Toast bread slices until crisp.",
        "Pile mushrooms on bread, top with gruyere, and broil briefly.",
    ], 420, 16, 30, 24),

    R("Falafel Plate", "Middle Eastern", 20, 15, 2, [
        i("chickpeas", 400, "g"), i("garlic", 3, "cloves"),
        i("parsley"), i("cumin", 1, "tsp"),
        i("coriander", 1, "tsp"), i("pita bread", 4),
        i("tahini", 3, "tbsp"), i("tomato", 2), i("cucumber", 1),
    ], [
        "Pulse chickpeas with garlic, parsley, and spices into a coarse mixture.",
        "Form patties and pan-fry until crisp.",
        "Serve with pita, tahini, tomato, and cucumber.",
    ], 470, 16, 56, 16),

    R("Harissa Roasted Cauliflower", "Middle Eastern", 10, 30, 1, [
        i("cauliflower", 1), i("chickpeas", 250, "g"),
        i("harissa", 2, "tbsp"), i("olive oil", 2, "tbsp"),
        i("lemon", 1), i("tahini", 2, "tbsp"), i("parsley"),
    ], [
        "Toss cauliflower and chickpeas with harissa and olive oil.",
        "Roast until deeply caramelized.",
        "Drizzle with tahini and lemon, then finish with parsley.",
    ], 320, 12, 28, 18),

    R("Za'atar Chicken Rice Bowl", "Middle Eastern", 15, 20, 2, [
        i("chicken thigh", 500, "g"), i("rice", 250, "g"),
        i("za'atar", 2, "tsp"), i("cucumber", 1),
        i("tomato", 2), i("yogurt", 120, "g"),
        i("lemon", 1), i("olive oil", 2, "tbsp"),
    ], [
        "Season chicken with za'atar, lemon, and olive oil.",
        "Roast or pan-sear until cooked through.",
        "Serve over rice with cucumber, tomato, and yogurt sauce.",
    ], 560, 34, 44, 22),
]


# --------------------------------------------------------------------------
# Expand compact form → full Recipe dict
# --------------------------------------------------------------------------

def build_recipe(idx: int, r: R) -> dict:
    ingredients = []
    for i_idx, ing in enumerate(r.ings):
        name = ing.name.strip().lower()
        category = classify_category(ing.name)
        ingredients.append({
            "name": name,
            "quantity": ing.qty,
            "unit": ing.unit,
            "category": category,
            "is_core": infer_is_core(name, category, i_idx, r.name),
        })
    ing_names = [ing["name"] for ing in ingredients]
    total = r.prep + r.cook
    dietary_tags = infer_dietary_tags(ing_names, total)
    flavor = infer_flavor(ing_names)
    return {
        "id": f"r{idx:03d}",
        "name": r.name,
        "ingredients": ingredients,
        "cuisine": r.cuisine,
        "dietary_tags": dietary_tags,
        "prep_time_min": r.prep,
        "cook_time_min": r.cook,
        "difficulty": r.difficulty,
        "nutrition": {
            "calories": r.calories,
            "protein_g": r.protein_g,
            "carbs_g": r.carbs_g,
            "fat_g": r.fat_g,
        },
        "flavor_profile": flavor,
        "instructions": r.steps,
    }


def main() -> None:
    out: list[dict] = []
    errors: list[tuple[str, str]] = []
    for idx, r in enumerate(RECIPES, start=1):
        candidate = build_recipe(idx, r)
        try:
            Recipe.model_validate(candidate)
            out.append(candidate)
        except Exception as e:
            errors.append((r.name, str(e)))

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(
        json.dumps(out, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    by_cuisine: dict[str, int] = {}
    for r in out:
        by_cuisine[r["cuisine"]] = by_cuisine.get(r["cuisine"], 0) + 1

    print(f"Wrote {len(out)} recipes → {OUT_JSON}")
    print("By cuisine: " + ", ".join(f"{k}={v}" for k, v in sorted(by_cuisine.items())))
    if errors:
        print(f"\n{len(errors)} validation failure(s):")
        for name, msg in errors:
            print(f"  - {name}: {msg[:120]}")


if __name__ == "__main__":
    main()
