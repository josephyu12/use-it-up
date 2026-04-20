"""Tests for profile.py — load/save, ratings, pantry, validation."""

import json

import pytest
from pydantic import ValidationError

from useitup.profile import (
    SoftPreferences,
    UserProfile,
    add_rating,
    load_profile,
    save_profile,
    update_pantry,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_profile(**kwargs) -> UserProfile:
    defaults = dict(
        user_id="test_user",
        hard_constraints=["gluten-free"],
        soft_preferences=SoftPreferences(
            max_prep_time_min=30,
            preferred_cuisines=["Italian"],
            goals=["high_protein"],
        ),
        pantry=["eggs", "chicken breast"],
    )
    defaults.update(kwargs)
    return UserProfile(**defaults)


# ---------------------------------------------------------------------------
# Load / save round-trip
# ---------------------------------------------------------------------------

def test_save_and_load_roundtrip(tmp_path):
    profile = _make_profile()
    save_profile(profile, base_dir=tmp_path)
    loaded = load_profile(profile.user_id, base_dir=tmp_path)
    assert loaded == profile


def test_save_creates_file(tmp_path):
    profile = _make_profile()
    save_profile(profile, base_dir=tmp_path)
    assert (tmp_path / "test_user.json").exists()


def test_load_demo_profile():
    from pathlib import Path
    base = Path(__file__).parent.parent / "data" / "profiles"
    profile = load_profile("demo_user", base_dir=base)
    assert profile.user_id == "demo_user"
    assert "gluten-free" in profile.hard_constraints
    assert len(profile.rating_history) == 8
    assert len(profile.pantry) > 0


def test_roundtrip_preserves_rating_history(tmp_path):
    profile = _make_profile()
    profile = add_rating(profile, "r001", 5)
    profile = add_rating(profile, "r002", 3)
    save_profile(profile, base_dir=tmp_path)
    loaded = load_profile(profile.user_id, base_dir=tmp_path)
    assert len(loaded.rating_history) == 2
    assert loaded.rating_history[0].recipe_id == "r001"
    assert loaded.rating_history[1].rating == 3


def test_load_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_profile("nonexistent", base_dir=tmp_path)


# ---------------------------------------------------------------------------
# Rating history
# ---------------------------------------------------------------------------

def test_add_rating_appends_entry():
    profile = _make_profile()
    updated = add_rating(profile, "r001", 4)
    assert len(updated.rating_history) == 1
    assert updated.rating_history[0].recipe_id == "r001"
    assert updated.rating_history[0].rating == 4


def test_add_rating_is_immutable():
    profile = _make_profile()
    updated = add_rating(profile, "r001", 4)
    assert len(profile.rating_history) == 0
    assert len(updated.rating_history) == 1


def test_add_multiple_ratings_accumulate():
    profile = _make_profile()
    for i, r in enumerate([5, 3, 1, 4, 2], start=1):
        profile = add_rating(profile, f"r{i:03d}", r)
    assert len(profile.rating_history) == 5
    ratings = [e.rating for e in profile.rating_history]
    assert ratings == [5, 3, 1, 4, 2]


def test_add_rating_invalid_value_raises():
    profile = _make_profile()
    with pytest.raises(ValidationError):
        add_rating(profile, "r001", 6)


def test_add_rating_zero_raises():
    profile = _make_profile()
    with pytest.raises(ValidationError):
        add_rating(profile, "r001", 0)


def test_rating_has_timestamp():
    profile = _make_profile()
    updated = add_rating(profile, "r001", 5)
    assert updated.rating_history[0].timestamp != ""


# ---------------------------------------------------------------------------
# Pantry
# ---------------------------------------------------------------------------

def test_update_pantry_replaces_contents():
    profile = _make_profile(pantry=["eggs"])
    updated = update_pantry(profile, ["chicken", "garlic", "olive oil"])
    assert updated.pantry == ["chicken", "garlic", "olive oil"]


def test_update_pantry_is_immutable():
    profile = _make_profile(pantry=["eggs"])
    update_pantry(profile, ["chicken"])
    assert profile.pantry == ["eggs"]


def test_update_pantry_empty():
    profile = _make_profile(pantry=["eggs", "milk"])
    updated = update_pantry(profile, [])
    assert updated.pantry == []


# ---------------------------------------------------------------------------
# Hard constraint validation
# ---------------------------------------------------------------------------

def test_valid_dietary_tags_accepted():
    profile = _make_profile(hard_constraints=["vegan", "nut-free"])
    assert "vegan" in profile.hard_constraints


def test_invalid_dietary_tag_raises():
    with pytest.raises(ValidationError):
        _make_profile(hard_constraints=["shellfish-free"])


def test_empty_hard_constraints_accepted():
    profile = _make_profile(hard_constraints=[])
    assert profile.hard_constraints == []


# ---------------------------------------------------------------------------
# Soft preferences validation
# ---------------------------------------------------------------------------

def test_invalid_goal_raises():
    with pytest.raises(ValidationError):
        SoftPreferences(goals=["keto"])


def test_valid_goals_accepted():
    prefs = SoftPreferences(goals=["high_protein", "low_cost"])
    assert "high_protein" in prefs.goals


def test_soft_preferences_defaults():
    prefs = SoftPreferences()
    assert prefs.max_prep_time_min is None
    assert prefs.preferred_cuisines == []
    assert prefs.goals == []
