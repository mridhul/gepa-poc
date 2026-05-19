"""SQLite database schema and helpers."""

import sqlite3
import json
from datetime import datetime
from typing import Optional
from pathlib import Path


SCHEMA = """
CREATE TABLE IF NOT EXISTS applicants (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL,
    resume_json TEXT NOT NULL,
    hired       INTEGER NOT NULL,
    split       TEXT NOT NULL,
    source      TEXT NOT NULL,
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS prompt_candidates (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    iteration    INTEGER NOT NULL,
    parent_id    INTEGER,
    prompts_json TEXT NOT NULL,
    created_at   TEXT NOT NULL,
    FOREIGN KEY (parent_id) REFERENCES prompt_candidates(id)
);

CREATE TABLE IF NOT EXISTS evaluations (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    candidate_id     INTEGER NOT NULL,
    applicant_id     INTEGER NOT NULL,
    scores_json      TEXT NOT NULL,
    aggregate_score  REAL NOT NULL,
    predicted_hired  INTEGER NOT NULL,
    FOREIGN KEY (candidate_id) REFERENCES prompt_candidates(id),
    FOREIGN KEY (applicant_id) REFERENCES applicants(id)
);

CREATE TABLE IF NOT EXISTS iteration_logs (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    iteration     INTEGER NOT NULL,
    candidate_id  INTEGER NOT NULL,
    split         TEXT NOT NULL,
    accuracy      REAL,
    precision_h   REAL,
    recall_h      REAL,
    f1_h          REAL,
    spearman_rho  REAL,
    is_pareto     INTEGER DEFAULT 0,
    notes         TEXT,
    created_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS optimization_state (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    status              TEXT NOT NULL,
    current_iteration   INTEGER NOT NULL DEFAULT 0,
    pareto_pool_ids     TEXT NOT NULL DEFAULT '[]',
    total_llm_calls     INTEGER NOT NULL DEFAULT 0,
    total_cost_usd      REAL NOT NULL DEFAULT 0.0,
    optimal_threshold   REAL,
    created_at          TEXT NOT NULL,
    updated_at          TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS llm_call_log (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    phase           TEXT NOT NULL,
    model           TEXT NOT NULL,
    prompt_tokens   INTEGER,
    completion_tokens INTEGER,
    latency_ms      INTEGER,
    cost_usd        REAL,
    created_at      TEXT NOT NULL
);
"""


def get_db(path: str = "data/gepa.db", *, check_same_thread: bool = True) -> sqlite3.Connection:
    """Get or create SQLite connection."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, check_same_thread=check_same_thread)
    conn.row_factory = sqlite3.Row
    return conn


def init_schema(conn: sqlite3.Connection) -> None:
    """Initialize database schema (idempotent)."""
    conn.executescript(SCHEMA)
    conn.commit()


def insert_applicant(
    conn: sqlite3.Connection,
    name: str,
    resume_json: dict,
    hired: int,
    split: str,
    source: str,
) -> int:
    """Insert applicant and return id."""
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO applicants (name, resume_json, hired, split, source, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (name, json.dumps(resume_json), hired, split, source, datetime.utcnow().isoformat()),
    )
    conn.commit()
    return cursor.lastrowid


def get_applicants_by_split(conn: sqlite3.Connection, split: str) -> list[dict]:
    """Get all applicants in a split."""
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM applicants WHERE split = ? ORDER BY id", (split,))
    return [dict(row) for row in cursor.fetchall()]


def get_all_applicants(conn: sqlite3.Connection) -> list[dict]:
    """Get all applicants."""
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM applicants ORDER BY id")
    return [dict(row) for row in cursor.fetchall()]


def count_applicants(conn: sqlite3.Connection) -> int:
    """Count total applicants."""
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM applicants")
    return cursor.fetchone()[0]


def insert_prompt_candidate(
    conn: sqlite3.Connection,
    iteration: int,
    parent_id: Optional[int],
    prompts: list[dict],
) -> int:
    """Insert prompt candidate and return id."""
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO prompt_candidates (iteration, parent_id, prompts_json, created_at)
        VALUES (?, ?, ?, ?)
        """,
        (iteration, parent_id, json.dumps(prompts), datetime.utcnow().isoformat()),
    )
    conn.commit()
    return cursor.lastrowid


def get_candidate(conn: sqlite3.Connection, candidate_id: int) -> Optional[dict]:
    """Get a prompt candidate by id."""
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM prompt_candidates WHERE id = ?", (candidate_id,))
    row = cursor.fetchone()
    return dict(row) if row else None


def insert_evaluation(
    conn: sqlite3.Connection,
    candidate_id: int,
    applicant_id: int,
    scores: list[dict],
    aggregate_score: float,
    predicted_hired: int,
) -> int:
    """Insert evaluation and return id."""
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO evaluations (candidate_id, applicant_id, scores_json, aggregate_score, predicted_hired)
        VALUES (?, ?, ?, ?, ?)
        """,
        (candidate_id, applicant_id, json.dumps(scores), aggregate_score, predicted_hired),
    )
    conn.commit()
    return cursor.lastrowid


def get_evaluations_for_candidate(
    conn: sqlite3.Connection,
    candidate_id: int,
    split: Optional[str] = None,
) -> list[dict]:
    """Get evaluations for a candidate, optionally filtered by split."""
    cursor = conn.cursor()
    if split:
        cursor.execute(
            """
            SELECT e.* FROM evaluations e
            JOIN applicants a ON e.applicant_id = a.id
            WHERE e.candidate_id = ? AND a.split = ?
            ORDER BY e.applicant_id
            """,
            (candidate_id, split),
        )
    else:
        cursor.execute(
            "SELECT * FROM evaluations WHERE candidate_id = ? ORDER BY applicant_id",
            (candidate_id,),
        )
    return [dict(row) for row in cursor.fetchall()]


def evaluations_exist_for_candidate(
    conn: sqlite3.Connection,
    candidate_id: int,
    split: Optional[str] = None,
) -> bool:
    """Check if evaluations exist for a candidate, optionally for one split."""
    cursor = conn.cursor()
    if split:
        cursor.execute(
            """
            SELECT COUNT(*) FROM evaluations e
            JOIN applicants a ON e.applicant_id = a.id
            WHERE e.candidate_id = ? AND a.split = ?
            """,
            (candidate_id, split),
        )
    else:
        cursor.execute(
            "SELECT COUNT(*) FROM evaluations WHERE candidate_id = ?",
            (candidate_id,),
        )
    return cursor.fetchone()[0] > 0


def insert_iteration_log(
    conn: sqlite3.Connection,
    iteration: int,
    candidate_id: int,
    split: str,
    accuracy: Optional[float],
    precision_h: Optional[float],
    recall_h: Optional[float],
    f1_h: Optional[float],
    spearman_rho: Optional[float],
    is_pareto: int,
    notes: Optional[str],
) -> int:
    """Insert iteration log and return id."""
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO iteration_logs
        (iteration, candidate_id, split, accuracy, precision_h, recall_h, f1_h, spearman_rho, is_pareto, notes, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            iteration,
            candidate_id,
            split,
            accuracy,
            precision_h,
            recall_h,
            f1_h,
            spearman_rho,
            is_pareto,
            notes,
            datetime.utcnow().isoformat(),
        ),
    )
    conn.commit()
    return cursor.lastrowid


def get_iteration_logs(conn: sqlite3.Connection) -> list[dict]:
    """Get all iteration logs."""
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM iteration_logs ORDER BY iteration, candidate_id")
    return [dict(row) for row in cursor.fetchall()]


def get_baseline_candidate_id(conn: sqlite3.Connection) -> Optional[int]:
    """Return the most recent human-authored baseline candidate id."""
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT candidate_id FROM iteration_logs
        WHERE iteration = 0 AND notes = 'Baseline (human-authored)'
        ORDER BY candidate_id DESC
        LIMIT 1
        """
    )
    row = cursor.fetchone()
    return row[0] if row else None


def get_optimization_state(conn: sqlite3.Connection) -> Optional[dict]:
    """Get optimization state (single row)."""
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM optimization_state LIMIT 1")
    row = cursor.fetchone()
    if row:
        row_dict = dict(row)
        row_dict["pareto_pool_ids"] = json.loads(row_dict["pareto_pool_ids"])
        return row_dict
    return None


def upsert_optimization_state(
    conn: sqlite3.Connection,
    status: str,
    current_iteration: int,
    pareto_pool_ids: list[int],
    total_llm_calls: int,
    total_cost_usd: float,
    optimal_threshold: Optional[float] = None,
) -> None:
    """Upsert optimization state (single row, idempotent)."""
    cursor = conn.cursor()
    now = datetime.utcnow().isoformat()
    existing = get_optimization_state(conn)
    if existing:
        cursor.execute(
            """
            UPDATE optimization_state
            SET status = ?, current_iteration = ?, pareto_pool_ids = ?,
                total_llm_calls = ?, total_cost_usd = ?, optimal_threshold = ?, updated_at = ?
            WHERE id = ?
            """,
            (
                status,
                current_iteration,
                json.dumps(pareto_pool_ids),
                total_llm_calls,
                total_cost_usd,
                optimal_threshold,
                now,
                existing["id"],
            ),
        )
    else:
        cursor.execute(
            """
            INSERT INTO optimization_state
            (status, current_iteration, pareto_pool_ids, total_llm_calls, total_cost_usd, optimal_threshold, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                status,
                current_iteration,
                json.dumps(pareto_pool_ids),
                total_llm_calls,
                total_cost_usd,
                optimal_threshold,
                now,
                now,
            ),
        )
    conn.commit()


def log_llm_call(
    conn: sqlite3.Connection,
    phase: str,
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    latency_ms: int,
    cost_usd: float,
) -> None:
    """Log an LLM call."""
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO llm_call_log
        (phase, model, prompt_tokens, completion_tokens, latency_ms, cost_usd, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            phase,
            model,
            prompt_tokens,
            completion_tokens,
            latency_ms,
            cost_usd,
            datetime.utcnow().isoformat(),
        ),
    )
    conn.commit()


def get_total_cost(conn: sqlite3.Connection) -> float:
    """Get total LLM cost so far."""
    cursor = conn.cursor()
    cursor.execute("SELECT COALESCE(SUM(cost_usd), 0.0) FROM llm_call_log")
    return cursor.fetchone()[0]


def get_total_calls(conn: sqlite3.Connection) -> int:
    """Get total LLM calls so far."""
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM llm_call_log")
    return cursor.fetchone()[0]


def get_llm_call_log(conn: sqlite3.Connection) -> list[dict]:
    """Get all LLM call logs."""
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM llm_call_log ORDER BY created_at")
    return [dict(row) for row in cursor.fetchall()]
