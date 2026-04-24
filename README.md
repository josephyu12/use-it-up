# UseItUp

An explainable, three-stage recipe recommendation system. Tell it what's in your kitchen and your dietary constraints; it picks a recipe you can actually cook and explains every decision in plain language — ingredients matched, rules satisfied, counterfactuals, past-recipe CBR trace, and what to buy next.

**CS 4580/5580 Final Project** — Joseph Yu, Gavin Onghai, Helen Mao.

---

## Quickstart on the Zoo

Only two dependencies are required (`numpy`, `pydantic`), both pre-installed on the Zoo.

```bash
cd use-it-up
python3 -m pip install --user -r requirements.txt      # numpy + pydantic only
python3 scripts/demo.py
```

`scripts/demo.py` is a zero-argument CLI that runs the full pipeline on three contrasting scenarios — gluten-free hard-constraint filtering, warm-start CBR retrieval from past ratings, and vegan ingredient adaptation — and prints the complete decision log, goal trace, counterfactual, CBR trace, and pantry utilization report for each. This is the fastest way to see every capability end-to-end.

The pipeline loads `data/recipes_curated.json` (102 recipes) by default. The much larger `data/recipes.json` (39 447 recipes, 55 MB) is present for stress testing but is not required for the demo.

---

## Run the tests

```bash
python3 -m pytest                       # 229 passing
python3 -m pytest --cov=useitup        # with coverage
```

There are no pytest flags required; all tests are runnable from the project root as declared in `CLAUDE.md`.

---

## Interactive notebook (richer demo)

```bash
python3 -m pip install --user jupyter ipywidgets matplotlib pandas
jupyter notebook notebooks/UseItUp.ipynb
```

The notebook provides an `ipywidgets`-driven form (pantry, constraints, goals, cuisine preferences, max prep time) and renders:
* the recipe card with tags, nutrition, and instructions,
* the full markdown explanation from `render_explanation()`,
* a bar chart of Stage 1 rule eliminations,
* a counterfactual "what if I relax X" explorer.

---

## Python API

```python
from useitup.data_loader import load_recipes
from useitup.profile import UserProfile, SoftPreferences
from useitup.pipeline import recommend

recipes = load_recipes("data/recipes_curated.json")
profile = UserProfile(
    user_id="me",
    hard_constraints=["gluten-free"],
    soft_preferences=SoftPreferences(goals=["high_protein"], max_prep_time_min=30),
    pantry=["chicken breast", "rice", "garlic", "olive oil", "onion"],
)

results = recommend(profile, recipes, top_k=3)
top = results[0]
print(top.adapted_recipe.recipe.name)
print(top.explanation.goal_trace)
print(top.explanation.counterfactual)
```

`recommend()` returns a list of `Recommendation` objects, each bundling the adapted recipe, the four-part `Explanation`, the Stage 1 `DecisionEntry` log, and the `CBRMatch` (similarity breakdown and nearest past recipe).

---

## What's in this repo

```
use-it-up/
  README.md                  ← this file
  PROJECT.md                 ← full design spec (schemas, rule defaults, phase log)
  docs/
    writeup.md               ← write-up (architecture, explanations, evaluation, refs)
    qa_report.md             ← QA scenarios + known edge cases
  scripts/
    demo.py                  ← zero-argument zoo demo (RUN THIS)
    build_*.py               ← dataset builders (not needed at runtime)
  notebooks/
    UseItUp.ipynb            ← interactive ipywidgets demo
  src/useitup/
    schemas.py               ← Pydantic models (Recipe, Ingredient, tags)
    data_loader.py           ← JSON ↔ Recipe[] conversion
    profile.py               ← UserProfile + SoftPreferences
    matching.py              ← Stage 1: asymmetric ingredient matching + rule engine
    cbr.py                   ← Stage 2: retrieve/reuse/revise/retain CBR
    explain.py               ← Stage 3: four-part explanation generation
    pipeline.py              ← Public API: recommend() + Recommendation
    webapp.py                ← Optional FastAPI UI (install extras: webapp)
  tests/                     ← 229 pytest tests covering every stage
  data/
    recipes_curated.json     ← 102-recipe demo corpus
    recipes.json             ← 39 447-recipe full corpus (optional)
    recipes_sample.json      ← 10-recipe smoke-test corpus
    substitutions.json       ← CBR adaptation rules
    profiles/                ← demo user profiles
```

---

## How the system explains its decisions

Every recommendation comes with four grounded explanations, each generated from structured data — not an LLM:

1. **Goal Trace.** "I recommend *Shakshuka* because you have 8 of 9 ingredients (88%), 2 of 2 core ingredients. Hard constraints respected: gluten-free, nut-free."
2. **Counterfactual.** "I did not recommend *Carne Asada Bowls* because essential coverage was 67% (missing: sirloin)."
3. **CBR Trace.** Identifies the highest-similarity past recipe and which feature groups drove the match (cuisine/protein/method/flavor/difficulty/prep time), or reports cold-start fallback.
4. **Ingredient Utilization Report.** Which pantry items were used, which were unused, which need to be bought, plus follow-up recipes that would consume the unused items.

Because each section is rule-driven, every claim is auditable against the decision log. The project explicitly avoids black-box neural recommendation — see `docs/writeup.md §4` for the design rationale.

---

## Further reading

* **`docs/writeup.md`** — full project write-up: architecture diagram, four real end-to-end explanations, evaluation table (98% line coverage, 31 scenario tests), annotated bibliography, division of labor, future work.
* **`docs/qa_report.md`** — QA scenarios pass/fail and known edge cases.
* **`PROJECT.md`** — internal design spec with per-phase implementation log.
