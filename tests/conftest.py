import pytest


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """Return exit code 0 when no tests are collected (empty test suite)."""
    if exitstatus == 5:
        session.exitstatus = 0
