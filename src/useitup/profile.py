"""User profile and preference management."""

import json
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, field_validator

from useitup.schemas import DietaryTag

_PROFILES_DIR = Path(__file__).parent.parent.parent / "data" / "profiles"

VALID_GOALS = frozenset(
    {"high_protein", "low_cost", "vegetarian", "vegan", "low_carb", "quick", "dairy_free"}
)


class SoftPreferences(BaseModel):
    max_prep_time_min: int | None = None
    max_cost_tier: int | None = None
    preferred_cuisines: list[str] = []
    goals: list[str] = []

    @field_validator("goals")
    @classmethod
    def goals_are_known(cls, v: list[str]) -> list[str]:
        unknown = [g for g in v if g not in VALID_GOALS]
        if unknown:
            raise ValueError(f"Unknown goals: {unknown}. Valid: {sorted(VALID_GOALS)}")
        return v


class RatingEntry(BaseModel):
    recipe_id: str
    rating: int
    timestamp: str

    @field_validator("rating")
    @classmethod
    def rating_in_range(cls, v: int) -> int:
        if not (1 <= v <= 5):
            raise ValueError(f"Rating must be 1–5, got {v}")
        return v


class UserProfile(BaseModel):
    user_id: str
    hard_constraints: list[str] = []
    soft_preferences: SoftPreferences = SoftPreferences()
    rating_history: list[RatingEntry] = []
    pantry: list[str] = []

    @field_validator("hard_constraints")
    @classmethod
    def constraints_are_valid_tags(cls, v: list[str]) -> list[str]:
        valid: set[str] = set(DietaryTag.__args__)  # type: ignore[attr-defined]
        invalid = [c for c in v if c not in valid]
        if invalid:
            raise ValueError(f"Unknown dietary tags: {invalid}. Valid: {sorted(valid)}")
        return v


def _profiles_dir(base: Path | None) -> Path:
    return base if base is not None else _PROFILES_DIR


def load_profile(user_id: str, base_dir: Path | None = None) -> UserProfile:
    path = _profiles_dir(base_dir) / f"{user_id}.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    return UserProfile.model_validate(data)


def save_profile(profile: UserProfile, base_dir: Path | None = None) -> None:
    directory = _profiles_dir(base_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{profile.user_id}.json"
    path.write_text(
        json.dumps(profile.model_dump(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def add_rating(profile: UserProfile, recipe_id: str, rating: int) -> UserProfile:
    entry = RatingEntry(
        recipe_id=recipe_id,
        rating=rating,
        timestamp=datetime.utcnow().isoformat(),
    )
    return profile.model_copy(update={"rating_history": [*profile.rating_history, entry]})


def update_pantry(profile: UserProfile, ingredients: list[str]) -> UserProfile:
    return profile.model_copy(update={"pantry": list(ingredients)})
