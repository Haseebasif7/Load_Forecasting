"""
HTTP contract tests for `backend.app.main` (maps to report TC-01–TC-08).

Uses synthetic `LoadedState` — no `artifacts/` or Modal required.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


def test_tc01_health(api_client: TestClient):
    r = api_client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert "single" in body["models"]
    assert body["n_test_windows"] > 0


def test_tc02_meta_window_horizon(api_client: TestClient):
    r = api_client.get("/api/meta")
    assert r.status_code == 200
    body = r.json()
    assert body["window"] == 336
    assert body["horizon"] == 24
    assert body["n_test_windows"] > 0
    assert "metrics" in body


def test_tc03_test_windows_pagination(api_client: TestClient):
    r = api_client.get("/api/test-windows", params={"limit": 5, "offset": 0})
    assert r.status_code == 200
    body = r.json()
    assert len(body["items"]) <= 5
    for it in body["items"]:
        assert "t" in it and "first_pred_time" in it and "last_pred_time" in it


def test_tc04_test_windows_filter_prefix(api_client: TestClient, minimal_state):
    prefix = minimal_state.times_iso[minimal_state.test_indices[0]][:7]
    r = api_client.get("/api/test-windows", params={"q": prefix})
    assert r.status_code == 200
    for it in r.json()["items"]:
        assert it["first_pred_time"].startswith(prefix)


def test_tc05_test_windows_random(api_client: TestClient, minimal_state):
    r = api_client.get("/api/test-windows", params={"random": "true"})
    assert r.status_code == 200
    body = r.json()
    assert body["total"] == 1
    assert len(body["items"]) == 1
    t = body["items"][0]["t"]
    assert t in set(minimal_state.test_indices.tolist())


def test_tc06_predict_valid_shape(api_client: TestClient, minimal_state):
    t = int(minimal_state.test_indices[0])
    r = api_client.post("/api/predict", json={"t": t, "model": "single"})
    assert r.status_code == 200
    body = r.json()
    assert len(body["forecast"]["pred_kw"]) == 24
    assert len(body["forecast"]["actual_kw"]) == 24
    assert len(body["forecast"]["naive_kw"]) == 24
    assert len(body["history"]["load_kw"]) == 336
    assert "mae_kw" in body["summary"]


def test_tc07_predict_invalid_t_returns_400(api_client: TestClient):
    r = api_client.post("/api/predict", json={"t": -1, "model": "single"})
    assert r.status_code == 400


def test_tc08_single_and_ensemble_roundtrip(api_client: TestClient, minimal_state):
    t = int(minimal_state.test_indices[0])
    r1 = api_client.post("/api/predict", json={"t": t, "model": "single"})
    r2 = api_client.post("/api/predict", json={"t": t, "model": "ensemble"})
    assert r1.status_code == 200 and r2.status_code == 200
    assert r1.json()["n_members"] >= 1
    assert r2.json()["n_members"] >= 1
