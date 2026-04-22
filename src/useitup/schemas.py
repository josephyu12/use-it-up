"""Pydantic models for UseItUp data structures."""

from typing import Literal

from pydantic import BaseModel, field_validator

IngredientCategory = Literal[
    "protein", "vegetable", "grain", "dairy", "spice", "fat", "condiment", "other"
]

DietaryTag = Literal[
    "vegan", "vegetarian", "gluten-free", "dairy-free", "nut-free",
    "low-carb", "high-protein", "low-cost", "quick",
]

FlavorTag = Literal[
    "spicy", "savory", "sweet", "sour", "umami", "smoky", "fresh", "rich"
]

Difficulty = Literal[1, 2, 3, 4, 5]


class Ingredient(BaseModel):
    name: str
    quantity: float | None = None
    unit: str | None = None
    category: IngredientCategory
    is_core: bool | None = None

    @field_validator("name")
    @classmethod
    def name_not_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Ingredient name must not be empty")
        return v


class Recipe(BaseModel):
    id: str
    name: str
    ingredients: list[Ingredient]
    cuisine: str
    dietary_tags: list[DietaryTag]
    prep_time_min: int
    cook_time_min: int
    difficulty: Difficulty
    nutrition: dict[str, float]
    flavor_profile: list[FlavorTag]
    instructions: list[str]

    @field_validator("dietary_tags", "flavor_profile", mode="before")
    @classmethod
    def deduplicate(cls, v: list) -> list:
        seen: set = set()
        return [x for x in v if not (x in seen or seen.add(x))]  # type: ignore[func-returns-value]

    @field_validator("ingredients")
    @classmethod
    def at_least_one_ingredient(cls, v: list[Ingredient]) -> list[Ingredient]:
        if not v:
            raise ValueError("Recipe must have at least one ingredient")
        return v
