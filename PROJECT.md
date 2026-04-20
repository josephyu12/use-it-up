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

## Data Schema (planned)

### Recipe
| Field | Type | Description |
|---|---|---|
| `id` | `str` | Unique identifier |
| `name` | `str` | Display name |
| `ingredients` | `list[str]` | Normalized ingredient names |
| `cuisine_type` | `str` | e.g. `"Italian"`, `"Mexican"` |
| `dietary_tags` | `list[str]` | e.g. `["vegan", "gluten-free"]` |
| `prep_time_min` | `int` | Preparation time in minutes |
| `cook_time_min` | `int` | Cook time in minutes |
| `difficulty` | `str` | `"easy"` / `"medium"` / `"hard"` |
| `protein_source` | `str` | e.g. `"chicken"`, `"tofu"`, `"lentils"` |
| `cooking_method` | `str` | e.g. `"stovetop"`, `"baked"`, `"raw"` |
| `flavor_profile` | `list[str]` | e.g. `["savory", "spicy"]` |
| `nutritional_estimates` | `dict[str, float]` | `{calories, protein_g, carbs_g, fat_g}` |
| `cost_tier` | `str` | `"low"` / `"medium"` / `"high"` |

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

## Current Status

**Phase 0 complete** — project scaffold created (directory structure, `pyproject.toml`, stub source modules, empty test suite, `CLAUDE.md`, `PROJECT.md`, `README.md`, skeleton notebook).

### Completed Phases
| Phase | Summary |
|---|---|
| 0 | Scaffold: directory layout, packaging config, empty stubs, documentation. |

### Next: Phase 1
Implement `schemas.py` (Pydantic models), load/validate a small sample recipe dataset in `data/`, and write tests for all models.
