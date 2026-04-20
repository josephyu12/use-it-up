"""Load and save Recipe collections to/from JSON."""

import json
from pathlib import Path

from useitup.schemas import Recipe


def load_recipes(path: str | Path) -> list[Recipe]:
    """Load recipes from a JSON file."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [Recipe.model_validate(r) for r in data]


def save_recipes(recipes: list[Recipe], path: str | Path) -> None:
    """Save recipes to a JSON file."""
    Path(path).write_text(
        json.dumps([r.model_dump() for r in recipes], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
