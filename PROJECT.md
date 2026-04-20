# UseItUp — Project Specification

## System Overview
UseItUp is a knowledge-based, explainable recipe recommendation system (CS 4580/5580 final project).
Users supply ingredients on hand plus dietary goals/constraints; the system recommends recipes
and explains every decision in human-readable language.

Team: Joseph Yu (Lead Engineer), Gavin Onghai (Explanation Engine & UI/UX), Helen Mao (Data & QA).

---

## Three-Stage Pipeline

### Stage 1 — Ingredient Matching & Rule-Based Filtering (`matching.py`)
1. Score every recipe in the database by **ingredient overlap** (fraction of required ingredients the user has).
2. Eliminate recipes below a configurable coverage threshold.
3. Apply a **rule engine** in priority order:
   - *Hard constraints* first: allergies, dietary restrictions (vegan, gluten-free, …).
   - *Soft constraints* next: max prep time, cost tier, cuisine preference.
4. Each rule that fires is appended to a **decision log** (rule name, recipes eliminated, reason).

### Stage 2 — Case-Based Reasoning (`cbr.py`)
Implements the classic **Retrieve–Reuse–Revise–Retain** cycle:
- **Retrieve**: find the candidate recipe most similar to the user's highest-rated past recipes.
  Similarity is computed over a feature vector: `(cuisine_type, protein_source, cooking_method, flavor_profile, difficulty)`.
- **Reuse**: propose the best match.
- **Revise**: adapt the proposal to the current context (e.g., substitute tofu for chicken when goal is Vegetarian).
- **Retain**: store successful recommendations in the user's case library for future retrieval.

### Stage 3 — Explanation Generation (`explain.py`)
Compiles the decision log from Stages 1–2 into four explanation types:
1. **Goal Trace** — why the top recipe was chosen (ingredients matched, goals satisfied, constraints respected).
2. **Counterfactual** — why the runner-up was not chosen and what constraint change would flip the decision.
3. **CBR Trace** — which past highly-rated recipe the recommendation derives from and what was adapted.
4. **Ingredient Utilization Report** — which on-hand ingredients are used/unused; follow-up recipe suggestions to minimize waste.

---

## Data Schema (Phase 1 — implemented)

### Ingredient
| Field | Type | Description |
|---|---|---|
| `name` | `str` | Normalized ingredient name (stripped, lowercase) |
| `quantity` | `float \| None` | Amount |
| `unit` | `str \| None` | Unit of measure |
| `category` | `IngredientCategory` | One of: `protein`, `vegetable`, `grain`, `dairy`, `spice`, `fat`, `condiment`, `other` |

### Recipe
| Field | Type | Description |
|---|---|---|
| `id` | `str` | Unique identifier |
| `name` | `str` | Display name |
| `ingredients` | `list[Ingredient]` | Structured ingredient list (≥1 required) |
| `cuisine` | `str` | e.g. `"Italian"`, `"Mexican"` |
| `dietary_tags` | `list[DietaryTag]` | Subset of `{vegan, vegetarian, gluten-free, dairy-free, nut-free, low-carb, high-protein, low-cost, quick}` |
| `prep_time_min` | `int` | Preparation time in minutes |
| `cook_time_min` | `int` | Cook time in minutes |
| `difficulty` | `Literal[1..5]` | Integer 1 (easiest) to 5 (hardest) |
| `nutrition` | `dict[str, float]` | e.g. `{calories, protein_g, carbs_g, fat_g}` |
| `flavor_profile` | `list[FlavorTag]` | Subset of `{spicy, savory, sweet, sour, umami, smoky, fresh, rich}` |
| `instructions` | `list[str]` | Step-by-step cooking instructions |

### UserProfile
| Field | Type | Description |
|---|---|---|
| `user_id` | `str` | Unique identifier |
| `pantry` | `list[str]` | Current on-hand ingredients |
| `dietary_restrictions` | `list[str]` | Hard constraints (allergies, diet) |
| `goals` | `list[str]` | Soft preferences (High Protein, Low Cost, …) |
| `cuisine_preferences` | `list[str]` | Preferred cuisine types |
| `max_prep_time_min` | `int \| None` | Hard prep-time limit |
| `ratings` | `dict[str, int]` | `{recipe_id: 1–5}` — CBR case library |

### DecisionLogEntry
| Field | Type | Description |
|---|---|---|
| `stage` | `int` | 1 or 2 |
| `rule_name` | `str` | Name of rule/step that fired |
| `recipes_eliminated` | `int` | Count eliminated by this step |
| `reason` | `str` | Human-readable explanation |

---

## Tagging Rulebooks (Phase 1)

### Ingredient Category Assignment (`classify_category`)
Keywords matched (substring) in priority order: protein → grain → dairy → spice → fat → vegetable → condiment → other.

### Dietary Tag Inference (`infer_dietary_tags`)
| Tag | Rule |
|---|---|
| `vegan` | No meat/fish/dairy keyword in any ingredient |
| `vegetarian` | No meat/fish keyword in any ingredient |
| `dairy-free` | No dairy keyword |
| `gluten-free` | No gluten keyword (flour, wheat, barley, rye, pasta variants…) |
| `nut-free` | No nut keyword |
| `quick` | `prep_time + cook_time ≤ 30 min` |
| `high-protein` | ≥2 distinct protein-category ingredients |
| `low-carb` | No gluten ingredient AND no rice or potato |

### Flavor Profile Inference (`infer_flavor_profile`)
Keyword scan of all ingredient names: spicy (chili/jalapeño/cayenne…), umami (soy/miso/parmesan/anchovy…), savory (salt/garlic/herb…), sweet (sugar/honey/maple…), sour (lemon/vinegar/tamarind…), smoky (smoked/bbq/chipotle…), fresh (mint/cilantro/cucumber…), rich (cream/butter/cheese/avocado…).

### Cuisine Assignment (`guess_cuisine`)
Scored keyword matching against 8 cuisine keyword lists: Italian, Mexican, Asian, Indian, American, Mediterranean, French, Middle Eastern. Highest score wins; ties default to American.

---

## Current Status

**Phase 4 complete** — `cbr.py` with `recipe_feature_vector`, `CBRRetriever`, `CBRAdapter`, `record_success`; `data/substitutions.json` with 20 entries; 30 passing tests.

### Rule Defaults
| Rule | Type | Default Weight | Default Threshold |
|---|---|---|---|
| `AllergyRule` | Hard | 1.0 | — |
| `DietaryRule` | Hard | 1.0 | — |
| `PantryCoverageRule` | Hard | 1.0 | `coverage ≥ 0.50` |
| `PrepTimeRule` | Soft | 0.4 | `prep_time_min ≤ profile.max_prep_time_min` |
| `CuisinePreferenceRule` | Soft | 0.3 | — |
| `GoalAlignmentRule` | Soft | 0.3 | — |

Fuzzy match threshold: `FUZZY_THRESHOLD = 75` (rapidfuzz `partial_ratio`).
Pantry coverage threshold: `COVERAGE_THRESHOLD = 0.50`.

### Completed Phases
| Phase | Summary |
|---|---|
| 0 | Scaffold: directory layout, packaging config, empty stubs, documentation. |
| 1 | Data layer: `schemas.py` Pydantic models, `data_loader.py`, `build_dataset.py`, `recipes_sample.json`, full test suite. |
| 2 | User profiles: `profile.py` with `UserProfile` (hard constraints, soft preferences, rating history, pantry), load/save JSON, immutable update helpers, `demo_user.json`, 20 tests. |
| 3 | Stage 1 matching: `matching.py` with `IngredientScorer` (rapidfuzz fuzzy matching), `Rule` protocol, 3 hard rules + 3 soft rules, `FilterEngine` producing `FilterResult` + `DecisionEntry` log, 40 tests. |
| 4 | Stage 2 CBR: `cbr.py` with `recipe_feature_vector` (cuisine/protein/method/flavor/difficulty/prep_time encoding), `CBRRetriever` (weighted centroid + cosine similarity + cold-start fallback), `CBRAdapter` (goal-based substitution from `data/substitutions.json`), `record_success`; 30 tests. |

### Schema / Interface Changes in Phase 1
- `ingredients` changed from `list[str]` → `list[Ingredient]` (structured model with `name`, `quantity`, `unit`, `category`).
- `difficulty` changed from `str` (`easy/medium/hard`) → `Literal[1..5]`.
- `cuisine_type` renamed → `cuisine`.
- `protein_source`, `cooking_method`, `cost_tier` removed (derivable from ingredients).
- `nutritional_estimates` renamed → `nutrition`.
- `dietary_tags` and `flavor_profile` are now validated against fixed vocabularies.

### Schema / Interface Changes in Phase 2
- `UserProfile` uses `hard_constraints: list[str]` (validated against `DietaryTag` vocabulary) instead of `dietary_restrictions`.
- `soft_preferences` is a nested `SoftPreferences` model (replaces flat `goals`, `cuisine_preferences`, `max_prep_time_min` fields).
- `rating_history: list[RatingEntry]` replaces `ratings: dict[str, int]` — each entry carries `recipe_id`, `rating`, and `timestamp`.
- `goals` validated against `VALID_GOALS` vocabulary in `SoftPreferences`.
- Profiles stored in `data/profiles/{user_id}.json`.

### Next: Phase 5
Implement `explain.py`: Explanation Generation (Goal Trace, Counterfactual, CBR Trace, Ingredient Utilization Report).
