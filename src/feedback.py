"""SQLite feedback log — analysts mark model verdicts as correct or wrong."""
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path("artifacts/feedback.db")


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    con.execute(
        """CREATE TABLE IF NOT EXISTS feedback (
               id          INTEGER PRIMARY KEY AUTOINCREMENT,
               ts          TEXT NOT NULL,
               amount      REAL,
               probability REAL,
               is_fraud    INTEGER,
               correct     INTEGER,
               features    TEXT
           )"""
    )
    con.commit()
    return con


def log(amount: float, prob: float, is_fraud: bool, analyst_correct: bool, features: dict) -> None:
    with _connect() as con:
        con.execute(
            "INSERT INTO feedback (ts, amount, probability, is_fraud, correct, features) VALUES (?,?,?,?,?,?)",
            (
                datetime.now(timezone.utc).isoformat(),
                float(amount),
                float(prob),
                int(is_fraud),
                int(analyst_correct),
                json.dumps(features),
            ),
        )


def stats() -> dict:
    with _connect() as con:
        row = con.execute("SELECT COUNT(*), COALESCE(SUM(correct),0) FROM feedback").fetchone()
    total, correct = int(row[0]), int(row[1])
    return {
        "total_feedback": total,
        "correct_count": correct,
        "wrong_count": total - correct,
        "agreement_rate": round(correct / total, 3) if total else None,
    }
