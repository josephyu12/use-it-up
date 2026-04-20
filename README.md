# UseItUp

An explainable recipe recommendation system. Tell it what's in your kitchen; it recommends what to cook and explains exactly why.

CS 4580/5580 Final Project — Joseph Yu, Gavin Onghai, Helen Mao.

## Install

Requires Python 3.10+. Uses [uv](https://github.com/astral-sh/uv) for dependency management.

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
uv pip install -e ".[dev]"
```

## Run the notebook

```bash
jupyter notebook notebooks/UseItUp.ipynb
```

Run cells 1–6 in order. Every cell is idempotent — re-run any cell without breaking state.
The notebook uses a single `state` dict and `ipywidgets` for interactive input.

To execute headlessly (smoke test):

```bash
jupyter nbconvert --to notebook --execute --output /tmp/executed.ipynb notebooks/UseItUp.ipynb
```

## Python API

```python
from useitup.pipeline import recommend
from useitup.data_loader import load_recipes
from useitup.profile import load_profile

recipes = load_recipes("data/recipes_sample.json")
profile = load_profile("demo_user", base_dir="data/profiles")

results = recommend(profile, recipes, top_k=3)
top = results[0]
print(top.adapted_recipe.recipe.name)
print(top.explanation.goal_trace)
```

## Run tests

```bash
pytest                        # all tests (includes notebook smoke test)
pytest -k "not notebook"     # fast unit tests only
```

## Project structure

```
useitup/
  data/               # Recipe JSON, user profiles, substitution rules
  src/useitup/
    schemas.py        # Pydantic models
    matching.py       # Stage 1: ingredient matching + rule filtering
    cbr.py            # Stage 2: case-based reasoning
    explain.py        # Stage 3: explanation generation
    profile.py        # User profile management
    pipeline.py       # Public API: recommend() + Recommendation dataclass
  notebooks/
    UseItUp.ipynb     # Interactive demo (ipywidgets + matplotlib)
  tests/
    test_pipeline.py  # Unit tests + notebook smoke test
```

See `PROJECT.md` for full design spec and current status.
