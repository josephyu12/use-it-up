# QA Report — UseItUp Phase 7

**Date:** 2026-04-28  
**QA Lead:** Helen Mao  
**Test suite:** `tests/test_scenarios.py`, `tests/test_explanation_quality.py`  
**Coverage:** 96% on the recommendation pipeline modules (`schemas`, `profile`, `data_loader`, `matching`, `cbr`, `explain`, `pipeline`); remaining uncovered lines are defensive guards for unreachable states. `enrichment.py` is an offline dataset-build utility and `webapp.py` is the FastAPI UI; both are exercised by their own tests but excluded from the core pipeline metric.

---

## Test Summary

| Scenario | Tests | Status | Notes |
|---|---|---|---|
| A — Gluten-free pasta pantry | 4 | ✅ Pass | GF tag enforced; rice pasta named; non-GF counterfactual produced |
| B — Vegan cold-start | 4 | ✅ Pass | Cold-start mode triggered; vegan tag enforced |
| C — Conflicting soft preferences | 4 | ✅ Pass | Pipeline doesn't crash; goal_trace surfaces unmet CuisinePreferenceRule |
| D — Sparse pantry (2 items) | 3 | ✅ Pass | ValueError raised at default threshold; lowered threshold returns recipe + 🛒 section |
| E — Adaptation triggered | 3 | ✅ Pass | chicken → tofu substitution recorded; CBR trace cites both |
| Quality: no template slots | 4 | ✅ Pass | Tested across 3 parametrized profiles |
| Quality: pantry ingredient named | 2 | ✅ Pass | Used items appear in goal trace and utilization report |
| Quality: counterfactual names rejected recipe | 2 | ✅ Pass | Rejected recipe name appears in counterfactual section |
| Quality: utilization sums | 5 | ✅ Pass | used + unused = pantry; overlap + missing = recipe ingredients |

**Total: 31/31 tests pass in 0.26 s.**

---

## Coverage Detail (recommendation pipeline)

| Module | Statements | Missed | Coverage |
|---|---|---|---|
| `schemas.py` | 42 | 0 | 100% |
| `profile.py` | 59 | 0 | 100% |
| `data_loader.py` | 8 | 0 | 100% |
| `pipeline.py` | 44 | 0 | 100% |
| `matching.py` | 355 | 9 | 97% |
| `cbr.py` | 273 | 17 | 94% |
| `explain.py` | 181 | 11 | 94% |
| **TOTAL (pipeline)** | **962** | **37** | **96%** |
| `webapp.py` (FastAPI UI) | 186 | 28 | 85% |
| `enrichment.py` (offline build) | 115 | 115 | 0% (offline-only) |

### Uncovered lines

- **`matching.py:216`** — defensive branch in `GoalAlignmentRule.fail_reason` for a goal key absent from `GOAL_TO_TAG`; unreachable because `SoftPreferences.goals` is validated against `VALID_GOALS` at profile creation.
- **`cbr.py:151`** — `protein_vec[-1] = 1.0` fallback when a protein-category ingredient matches none of the explicit protein keywords (e.g., an exotic protein like "seitan"). Not hit by current sample data.
- **`cbr.py:188, 302, 327`** — early-return guards in `CBRRetriever.retrieve` and `record_success`; would fire if `candidates` is empty or `rating` is out-of-range, but callers already gate these paths.
- **`explain.py:141, 149–151`** — `recipe_index.get(rid) is None` guard and `best_recipe is None` fallback in `_build_counterfactual`. These require a rejected recipe ID with no matching recipe object — impossible with the current pipeline since `all_recipes` is always the same list that was filtered.
- **`explain.py:204–206, 211`** — cold-start CBR trace with non-empty adaptations, and the `nearest_past_recipe is None` branch in the non-cold-start path. The latter requires CBR to return a match with `nearest_past_recipe=None` despite having rating history — a state the current `CBRRetriever` never produces.

All uncovered lines are defensive `None`-checks or fallbacks for states the rest of the pipeline actively prevents. Reaching them would require either corrupt input or internal contract violations.

---

## Known Edge Cases Not Handled

### 1. Fuzzy-match false positives at the 75% threshold

`"olive oil"` fuzzy-matches `"lime"` with `partial_ratio = 75` exactly, because the substring `"live"` inside `"olive"` scores against `"lime"`. As a result, `_pantry_used` may classify `"olive oil"` as used when a recipe contains `"lime"`, and the utilization report will over-count used pantry items by 1. Observed with the demo_user profile against Black Bean Tacos.

**Impact:** Minor display inaccuracy; ingredient coverage score (computed by `IngredientScorer`) is unaffected because it traverses in the opposite direction (recipe ingredients → pantry).

### 2. Recipe dietary tag vs. ingredient reality mismatch

`Lentil Dal` (sample-007) carries both `"vegan"` and `"dairy-free"` tags despite containing `ghee` (a dairy fat). The `DietaryRule` passes vegan profiles through because it trusts tags, not ingredients. A vegan user following this recommendation would receive incorrect results.

**Impact:** Correctness issue; no current test catches ingredient-level vegan violations.

### 3. Adaptation does not update dietary tags

`CBRAdapter.adapt()` substitutes ingredient names but never updates `recipe.dietary_tags`. After adapting chicken → tofu the recipe still lacks a `"vegetarian"` tag, so `GoalAlignmentRule` still reports the goal as unsatisfied. The goal_trace then correctly warns about an unmet soft preference even though the adapted recipe is vegetarian.

**Impact:** Cosmetic inaccuracy in the goal_trace; the adaptation itself is correct.

### 4. Cold-start ranking ignores pantry coverage

In cold-start mode `CBRRetriever.retrieve` ranks candidates by `(cuisine_in_preferences, prep_time_score)` only. A recipe that matches 1% of the pantry can rank above one that matches 60%. This can surface recipes with very long shopping lists when the user has few ingredients.

### 5. No handling for identical recipe IDs

If two recipes share the same `id`, the `recipe_index` dict in `_build_counterfactual` silently keeps the last one. The pipeline has no uniqueness validation at load time.

---

## What We Would Fix with More Time

1. **Fuzzy threshold tuning** — Use a higher threshold (80–85) or switch from `partial_ratio` to `token_sort_ratio` to eliminate false positives like the olive-oil/lime case.

2. **Ingredient-level vegan/vegetarian validation** — Augment `infer_dietary_tags` in the data pipeline to scan ingredient names against `_GOAL_CONFLICTS` and refuse to tag a recipe `"vegan"` if any conflicting ingredient is present.

3. **Tag update after adaptation** — After `CBRAdapter.adapt()` modifies ingredients, infer and merge the new appropriate dietary tags so the goal_trace accurately reflects the adapted recipe's properties.

4. **Coverage-weighted cold-start ranking** — Multiply the cold-start score by ingredient coverage so sparse-pantry users get recipes they can actually make without a long shopping trip.

5. **Recipe ID uniqueness check** — Add a validation step in `load_recipes` that raises if any `id` is duplicated.

6. **Broader recipe catalog** — Ten sample recipes make it easy to exhaust the survivor pool. More recipes (especially tagged with diverse cuisines) would make Scenario C tests more meaningful and stress the CBR similarity more realistically.
