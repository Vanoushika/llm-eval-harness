import os

import pytest
from fastapi.testclient import TestClient

# Point the module-level store at a throwaway file before the app imports.
os.environ.setdefault("DATABASE_URL", "sqlite:///./test_evalharness.db")

from evalharness import main as main_module  # noqa: E402
from evalharness.main import app  # noqa: E402


@pytest.fixture()
def client(tmp_path):
    """A TestClient whose store writes to a fresh temp database."""
    db_path = tmp_path / "test.db"
    main_module.store = main_module.Store(f"sqlite:///{db_path}")
    with TestClient(app) as test_client:
        yield test_client
