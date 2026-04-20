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

## Run tests

```bash
pytest
```

## Project structure

```
useitup/
  data/               # Recipe JSON, user profiles
  src/useitup/
    schemas.py        # Pydantic models
    matching.py       # Stage 1: ingredient matching + rule filtering
    cbr.py            # Stage 2: case-based reasoning
    explain.py        # Stage 3: explanation generation
    profile.py        # User profile management
    pipeline.py       # Top-level orchestrator
  notebooks/
    UseItUp.ipynb     # Main demo notebook
  tests/
```

See `PROJECT.md` for full design spec and current status.
