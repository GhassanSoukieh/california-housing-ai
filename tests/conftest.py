import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


BACKEND_DIR = Path(__file__).parents[1] / "back-end"
sys.path.insert(0, str(BACKEND_DIR))


@pytest.fixture
def client(monkeypatch):
    import main

    monkeypatch.setattr(main, "load_model", lambda: object())
    with TestClient(main.app) as test_client:
        yield test_client
