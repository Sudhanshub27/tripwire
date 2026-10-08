import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import structlog
from config.settings import settings

logger = structlog.get_logger(__name__)

DEFAULT_DB_PATH = settings.DATA_PROCESSED_DIR / "resolution_reviews.db"
DEFAULT_JSONL_PATH = settings.DATA_PROCESSED_DIR / "resolution_review_queue.jsonl"


class ReviewQueue:
    """
    Persistent human-in-the-loop review queue for ambiguous entity resolution matches.
    Stores records in SQLite for ACID transactional consistency and keeps a JSONL mirror.
    """

    def __init__(
        self,
        db_path: Optional[Path] = None,
        jsonl_path: Optional[Path] = None,
    ):
        self.db_path = db_path or DEFAULT_DB_PATH
        self.jsonl_path = jsonl_path or DEFAULT_JSONL_PATH
        self._init_storage()

    def _get_connection(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_storage(self) -> None:
        """Initialize SQLite table and parent directories."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        with self._get_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS resolution_reviews (
                    id TEXT PRIMARY KEY,
                    candidate TEXT NOT NULL,
                    canonical TEXT NOT NULL,
                    matching_method TEXT NOT NULL,
                    fuzzy_score REAL NOT NULL,
                    confidence REAL NOT NULL,
                    decision TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    source TEXT NOT NULL,
                    match_reason TEXT,
                    reviewed_at TEXT,
                    reviewer_notes TEXT
                )
                """
            )
            conn.commit()

    def _sync_jsonl_mirror(self) -> None:
        """Write all reviews from SQLite to the JSONL mirror for transparent inspection."""
        try:
            reviews = self.list_all_reviews()
            with open(self.jsonl_path, "w", encoding="utf-8") as f:
                for r in reviews:
                    f.write(json.dumps(r) + "\n")
        except Exception as e:
            logger.warning("Failed syncing resolution review JSONL mirror", error=str(e))

    def add_review(
        self,
        candidate: str,
        canonical: str,
        matching_method: str,
        fuzzy_score: float,
        confidence: float,
        source: str = "pipeline",
        match_reason: str = "",
        notes: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Add an ambiguous entity resolution match to the review queue.
        If a review for the same candidate and canonical is already pending, returns existing entry.
        """
        with self._get_connection() as conn:
            # Check for existing pending review
            existing = conn.execute(
                """
                SELECT * FROM resolution_reviews
                WHERE candidate = ? AND canonical = ? AND decision = 'REVIEW'
                """,
                (candidate, canonical),
            ).fetchone()

            if existing:
                return dict(existing)

            # Generate unique deterministic review id
            clean_str = f"{candidate.strip().lower()}_{canonical.strip().lower()}"
            record_id = f"rev-{abs(hash(clean_str)) % 100000:05d}"
            now_iso = datetime.now(timezone.utc).isoformat()

            conn.execute(
                """
                INSERT OR REPLACE INTO resolution_reviews
                (id, candidate, canonical, matching_method, fuzzy_score, confidence, decision, timestamp, source, match_reason, reviewed_at, reviewer_notes)
                VALUES (?, ?, ?, ?, ?, ?, 'REVIEW', ?, ?, ?, NULL, ?)
                """,
                (
                    record_id,
                    candidate,
                    canonical,
                    matching_method,
                    round(float(fuzzy_score), 4),
                    round(float(confidence), 4),
                    now_iso,
                    source,
                    match_reason,
                    notes,
                ),
            )
            conn.commit()

        review_item = {
            "id": record_id,
            "candidate": candidate,
            "candidate_entity": candidate,
            "canonical": canonical,
            "matched_entity": canonical,
            "matching_method": matching_method,
            "fuzzy_score": round(float(fuzzy_score), 4),
            "confidence": round(float(confidence), 4),
            "confidence_score": round(float(confidence), 4),
            "decision": "REVIEW",
            "final_decision": "REVIEW",
            "timestamp": now_iso,
            "source": source,
            "match_reason": match_reason,
            "reviewed_at": None,
            "reviewer_notes": notes,
        }

        self._sync_jsonl_mirror()
        logger.info(
            "Enqueued ambiguous entity resolution match for review",
            id=record_id,
            candidate=candidate,
            canonical=canonical,
            confidence=round(float(confidence), 4),
        )
        return review_item

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
        """Convert sqlite3.Row to dict and ensure all required schema fields are present."""
        d = dict(row)
        d["candidate_entity"] = d.get("candidate")
        d["matched_entity"] = d.get("canonical")
        d["confidence_score"] = d.get("confidence")
        d["final_decision"] = d.get("decision")
        return d

    def get_pending_reviews(self) -> List[Dict[str, Any]]:
        """Return all reviews with decision == 'REVIEW'."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT * FROM resolution_reviews
                WHERE decision = 'REVIEW'
                ORDER BY timestamp DESC
                """
            )
            return [self._row_to_dict(row) for row in cursor.fetchall()]

    def get_review_by_id(self, review_id: str) -> Optional[Dict[str, Any]]:
        """Fetch a specific review record by ID."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM resolution_reviews WHERE id = ?",
                (review_id,),
            )
            row = cursor.fetchone()
            return self._row_to_dict(row) if row else None

    def approve_review(
        self,
        review_id: str,
        notes: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Approve an ambiguous match (candidate merges with canonical).
        Updates decision to 'APPROVED' and records reviewed_at timestamp.
        """
        now_iso = datetime.now(timezone.utc).isoformat()
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                UPDATE resolution_reviews
                SET decision = 'APPROVED', reviewed_at = ?, reviewer_notes = coalesce(?, reviewer_notes)
                WHERE id = ?
                """,
                (now_iso, notes, review_id),
            )
            conn.commit()
            if cursor.rowcount == 0:
                return None

        self._sync_jsonl_mirror()
        updated = self.get_review_by_id(review_id)
        if updated:
            logger.info(
                "Approved entity resolution review",
                id=review_id,
                candidate=updated.get("candidate"),
                canonical=updated.get("canonical"),
            )
        return updated

    def reject_review(
        self,
        review_id: str,
        notes: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Reject an ambiguous match (candidate treated as a distinct NEW entity).
        Updates decision to 'REJECTED' and records reviewed_at timestamp.
        """
        now_iso = datetime.now(timezone.utc).isoformat()
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                UPDATE resolution_reviews
                SET decision = 'REJECTED', reviewed_at = ?, reviewer_notes = coalesce(?, reviewer_notes)
                WHERE id = ?
                """,
                (now_iso, notes, review_id),
            )
            conn.commit()
            if cursor.rowcount == 0:
                return None

        self._sync_jsonl_mirror()
        updated = self.get_review_by_id(review_id)
        if updated:
            logger.info(
                "Rejected entity resolution review",
                id=review_id,
                candidate=updated.get("candidate"),
                canonical=updated.get("canonical"),
            )
        return updated

    def list_all_reviews(self) -> List[Dict[str, Any]]:
        """Return all review records sorted by timestamp descending."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM resolution_reviews ORDER BY timestamp DESC"
            )
            return [self._row_to_dict(row) for row in cursor.fetchall()]


review_queue = ReviewQueue()
