"""Pytest path setup and shared fixtures."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
_BACKEND = _ROOT / "backend"

for _p in (_ROOT, _BACKEND):
    s = str(_p)
    if s not in sys.path:
        sys.path.insert(0, s)


@pytest.fixture
def minimal_state():
    from testing.fixtures_minimal_state import build_minimal_loaded_state

    return build_minimal_loaded_state()


@pytest.fixture
def api_client(monkeypatch, minimal_state):
    """FastAPI TestClient with `load_state` returning synthetic `LoadedState`."""
    import app.inference as inference_mod

    monkeypatch.setattr(inference_mod, "load_state", lambda: minimal_state)
    from app.main import app
    from fastapi.testclient import TestClient

    with TestClient(app) as client:
        yield client
