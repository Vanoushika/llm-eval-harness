"""Storage — SQLAlchemy over SQLite by default, PostgreSQL-ready.

Suites and runs are stored as JSON payloads in two small tables, so the
same code works against SQLite locally and PostgreSQL in production via
the ``DATABASE_URL`` environment variable.
"""

from __future__ import annotations

import json
import os

from sqlalchemy import JSON, Column, DateTime, String, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from .models import EvalRun, EvalSuite

Base = declarative_base()


class SuiteRow(Base):
    __tablename__ = "suites"
    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    payload = Column(JSON, nullable=False)


class RunRow(Base):
    __tablename__ = "runs"
    id = Column(String, primary_key=True)
    suite_id = Column(String, nullable=False)
    created_at = Column(DateTime, nullable=False)
    payload = Column(JSON, nullable=False)


def _database_url() -> str:
    return os.environ.get("DATABASE_URL", "sqlite:///./evalharness.db")


def make_session_factory(database_url: str | None = None):
    url = database_url or _database_url()
    connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
    engine = create_engine(url, connect_args=connect_args)
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, autoflush=False, autocommit=False)


class Store:
    """Thin persistence layer for suites and runs."""

    def __init__(self, database_url: str | None = None):
        self._sessions = make_session_factory(database_url)

    def save_suite(self, suite: EvalSuite) -> EvalSuite:
        with self._sessions() as session:
            row = session.get(SuiteRow, suite.id)
            payload = json.loads(suite.model_dump_json())
            if row is None:
                session.add(SuiteRow(id=suite.id, name=suite.name, payload=payload))
            else:
                row.name = suite.name
                row.payload = payload
            session.commit()
        return suite

    def get_suite(self, suite_id: str) -> EvalSuite | None:
        with self._sessions() as session:
            row = session.get(SuiteRow, suite_id)
            return EvalSuite(**row.payload) if row else None

    def list_suites(self) -> list[EvalSuite]:
        with self._sessions() as session:
            rows = session.query(SuiteRow).order_by(SuiteRow.name).all()
            return [EvalSuite(**r.payload) for r in rows]

    def save_run(self, run: EvalRun) -> EvalRun:
        with self._sessions() as session:
            session.add(
                RunRow(
                    id=run.id,
                    suite_id=run.suite_id,
                    created_at=run.started_at,
                    payload=json.loads(run.model_dump_json()),
                )
            )
            session.commit()
        return run

    def get_run(self, run_id: str) -> EvalRun | None:
        with self._sessions() as session:
            row = session.get(RunRow, run_id)
            return EvalRun(**row.payload) if row else None

    def list_runs(self, suite_id: str | None = None) -> list[EvalRun]:
        with self._sessions() as session:
            query = session.query(RunRow).order_by(RunRow.created_at.desc())
            if suite_id:
                query = query.filter(RunRow.suite_id == suite_id)
            return [EvalRun(**r.payload) for r in query.all()]
