# UseItUp — Claude Code Conventions

## Coding Conventions
- **Type hints everywhere**: every function parameter and return value must be annotated.
- **Dataclasses / Pydantic over plain dicts**: structured data must use a typed model defined in `schemas.py`.
- **Small, pure functions**: prefer functions with no side effects that take inputs and return outputs. Keep functions under ~40 lines.
- **No global state**: do not use module-level mutable variables. Pass dependencies explicitly.
- **Imports**: stdlib first, then third-party, then local — separated by blank lines.

## Testing Conventions
- Framework: **pytest** only (no unittest).
- One test file per source module: `tests/test_schemas.py`, `tests/test_matching.py`, `tests/test_cbr.py`, `tests/test_explain.py`, `tests/test_profile.py`, `tests/test_pipeline.py`.
- Tests must be runnable with `pytest` from the project root with no extra flags.
- Prefer small, focused unit tests with explicit assertions over broad integration tests.
- Use `tmp_path` (pytest fixture) for any file I/O in tests — never write to the `data/` directory from tests.

## End-of-Phase Checklist
At the end of every phase, update `PROJECT.md`:
1. Move the completed phase to the "Completed" section with a one-line summary.
2. Update "Current status" to reflect the next phase.
3. Note any schema or interface changes made during the phase.
