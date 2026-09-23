from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from backend.app.db.database import Base, get_db
from backend.app.main import app


@pytest.fixture()
def db_session():
    """An isolated in-memory SQLite session, fresh for every test.

    StaticPool is required here: without it, every checkout from the
    pool opens a *new* `:memory:` database, so the tables created below
    would be invisible to the session actually used by the test.
    """

    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    from backend.app.db import models  # noqa: F401  (register tables on Base)

    Base.metadata.create_all(bind=engine)

    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(bind=engine)


@pytest.fixture()
def client(db_session):
    """A TestClient with the DB dependency overridden to the isolated session."""

    def _override_get_db():
        try:
            yield db_session
        finally:
            pass

    app.dependency_overrides[get_db] = _override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


SAMPLE_ASSESSMENT_REQUEST = {
    "A1_Score": 1,
    "A2_Score": 1,
    "A3_Score": 1,
    "A4_Score": 1,
    "A5_Score": 1,
    "A6_Score": 1,
    "A7_Score": 1,
    "A8_Score": 1,
    "A9_Score": 1,
    "A10_Score": 1,
    "age": 8,
    "gender": "m",
    "ethnicity": "White-European",
    "jundice": "no",
    "austim": "no",
    "contry_of_res": "United States",
    "used_app_before": "no",
    "relation": "Parent",
}
