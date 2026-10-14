import json
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import structlog
from config.settings import settings

from src.observability.metrics import metrics_collector

logger = structlog.get_logger(__name__)

DEFAULT_DB_PATH = settings.DATA_PROCESSED_DIR / "resolution_reviews.db"
DEFAULT_JSONL_PATH = settings.DATA_PROCESSED_DIR / "resolution_review_queue.jsonl"


class ReviewStatus(str, Enum):
    """Review item state in review lifecycle."""

    PENDING = "PENDING"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"

    # Backwards compatibility alias
    REVIEW = "PENDING"


class ReviewAction(str, Enum):
    """Supported human review action outcomes."""

    APPROVE = "APPROVE"
    REJECT = "REJECT"


class ReviewQueue:
    """
    Persistent human-in-the-loop review queue for ambiguous entity resolution matches.
    Stores records and audit trail in SQLite for ACID transactional consistency
    and maintains a JSONL mirror for transparent inspection.
    """

    def __init__(
        self,
        db_path: Optional[Path] = None,
        jsonl_path: Optional[Path] = None,
    ):
        self.db_path = db_path or DEFAULT_DB_PATH
        self.jsonl_path = jsonl_path or DEFAULT_JSONL_PATH
        self._lock = threading.Lock()
        self._init_storage()

    def _get_connection(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_storage(self) -> None:
        """Initialize SQLite tables, migrations, and parent directories."""
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
                    status TEXT NOT NULL DEFAULT 'PENDING',
                    timestamp TEXT NOT NULL,
                    source TEXT NOT NULL,
                    match_reason TEXT,
                    matching_signals TEXT DEFAULT '{}',
                    resolver_version TEXT DEFAULT '2.0.0',
                    reviewed_at TEXT,
                    reviewer TEXT,
                    reviewer_notes TEXT
                )
                """
            )
            # Schema migration for existing tables missing new columns
            cursor = conn.execute("PRAGMA table_info(resolution_reviews)")
            cols = {row["name"] for row in cursor.fetchall()}
            if "status" not in cols:
                conn.execute("ALTER TABLE resolution_reviews ADD COLUMN status TEXT NOT NULL DEFAULT 'PENDING'")
            if "matching_signals" not in cols:
                conn.execute("ALTER TABLE resolution_reviews ADD COLUMN matching_signals TEXT DEFAULT '{}'")
            if "resolver_version" not in cols:
                conn.execute("ALTER TABLE resolution_reviews ADD COLUMN resolver_version TEXT DEFAULT '2.0.0'")
            if "reviewer" not in cols:
                conn.execute("ALTER TABLE resolution_reviews ADD COLUMN reviewer TEXT")

            # Immutable audit trail table
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS resolution_review_audit (
                    audit_id TEXT PRIMARY KEY,
                    review_id TEXT NOT NULL,
                    candidate TEXT NOT NULL,
                    canonical TEXT NOT NULL,
                    original_resolution TEXT NOT NULL,
                    original_confidence REAL NOT NULL,
                    reviewer_action TEXT NOT NULL,
                    previous_state TEXT NOT NULL,
                    new_state TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    reviewer TEXT,
                    notes TEXT,
                    signals TEXT
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_audit_review_id ON resolution_review_audit(review_id)"
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

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
        """Convert sqlite3.Row to dict and ensure schema fields and backwards compatibility."""
        d = dict(row)
        d["candidate_entity"] = d.get("candidate")
        d["matched_entity"] = d.get("canonical")
        d["confidence_score"] = d.get("confidence")
        d["final_decision"] = d.get("decision")
        if not d.get("status"):
            dec = str(d.get("decision", "REVIEW")).upper()
            d["status"] = "PENDING" if dec in ("REVIEW", "NEEDS_REVIEW") else dec

        sig_raw = d.get("matching_signals")
        if isinstance(sig_raw, str):
            try:
                d["matching_signals"] = json.loads(sig_raw)
            except Exception:
                d["matching_signals"] = {}
        elif sig_raw is None:
            d["matching_signals"] = {}

        return d

    def add_review(
        self,
        candidate: str,
        canonical: str,
        matching_method: str,
        fuzzy_score: float,
        confidence: float,
        source: str = "pipeline",
        match_reason: str = "",
        signals: Optional[Dict[str, Any]] = None,
        notes: Optional[str] = None,
        resolver_version: str = "2.0.0",
    ) -> Dict[str, Any]:
        """
        Add an ambiguous entity resolution match to the persistent review queue.
        If a review for the same candidate and canonical is already pending, returns existing entry.
        Records an auditable ENQUEUE entry in the audit trail.
        """
        signals_json = json.dumps(signals or {})
        with self._get_connection() as conn:
            # Check for existing pending review
            existing = conn.execute(
                """
                SELECT * FROM resolution_reviews
                WHERE candidate = ? AND canonical = ? AND (decision = 'REVIEW' OR status = 'PENDING')
                """,
                (candidate, canonical),
            ).fetchone()

            if existing:
                return self._row_to_dict(existing)

            clean_str = f"{candidate.strip().lower()}_{canonical.strip().lower()}"
            record_id = f"rev-{abs(hash(clean_str)) % 100000:05d}"
            now_iso = datetime.now(timezone.utc).isoformat()

            conn.execute(
                """
                INSERT OR REPLACE INTO resolution_reviews
                (id, candidate, canonical, matching_method, fuzzy_score, confidence, decision, status,
                 timestamp, source, match_reason, matching_signals, resolver_version, reviewed_at, reviewer, reviewer_notes)
                VALUES (?, ?, ?, ?, ?, ?, 'REVIEW', 'PENDING', ?, ?, ?, ?, ?, NULL, NULL, ?)
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
                    signals_json,
                    resolver_version,
                    notes,
                ),
            )

            # Audit record for enqueue
            audit_id = f"aud-{uuid.uuid4().hex[:8]}"
            conn.execute(
                """
                INSERT INTO resolution_review_audit
                (audit_id, review_id, candidate, canonical, original_resolution, original_confidence,
                 reviewer_action, previous_state, new_state, timestamp, reviewer, notes, signals)
                VALUES (?, ?, ?, ?, 'NEEDS_REVIEW', ?, 'ENQUEUE', 'NONE', 'PENDING', ?, 'system', ?, ?)
                """,
                (
                    audit_id,
                    record_id,
                    candidate,
                    canonical,
                    round(float(confidence), 4),
                    now_iso,
                    match_reason or notes,
                    signals_json,
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
            "status": "PENDING",
            "timestamp": now_iso,
            "source": source,
            "match_reason": match_reason,
            "matching_signals": signals or {},
            "resolver_version": resolver_version,
            "reviewed_at": None,
            "reviewer": None,
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

    def get_pending_reviews(self) -> List[Dict[str, Any]]:
        """Return all active review records requiring human attention."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT * FROM resolution_reviews
                WHERE status = 'PENDING' OR decision = 'REVIEW'
                ORDER BY timestamp DESC
                """
            )
            return [self._row_to_dict(row) for row in cursor.fetchall()]

    def get_review_by_id(self, review_id: str) -> Optional[Dict[str, Any]]:
        """Fetch a specific review record by ID including its audit history."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM resolution_reviews WHERE id = ?",
                (review_id,),
            )
            row = cursor.fetchone()
            if not row:
                return None
            data = self._row_to_dict(row)
            data["audit_trail"] = self.get_review_audit_trail(review_id)
            return data

    def approve_review(
        self,
        review_id: str,
        notes: Optional[str] = None,
        reviewer: Optional[str] = "human_reviewer",
    ) -> Optional[Dict[str, Any]]:
        """
        Approve an ambiguous match (candidate merges into canonical).
        Validates state transitions (fails safely if already rejected or approved).
        Persists decision, records audit entry, and updates metrics.
        """
        now_iso = datetime.now(timezone.utc).isoformat()
        reviewer_name = reviewer or "human_reviewer"

        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.execute("SELECT * FROM resolution_reviews WHERE id = ?", (review_id,))
                row = cursor.fetchone()
                if not row:
                    return None

                curr_status = (row["status"] or row["decision"]).upper()
                if curr_status == "REJECTED":
                    raise ValueError(
                        f"Cannot approve review '{review_id}' with status 'REJECTED'. Invalid state transition."
                    )
                if curr_status == "APPROVED":
                    raise ValueError(f"Review '{review_id}' is already approved.")

                cursor = conn.execute(
                    """
                    UPDATE resolution_reviews
                    SET decision = 'APPROVED',
                        status = 'APPROVED',
                        reviewed_at = ?,
                        reviewer = ?,
                        reviewer_notes = coalesce(?, reviewer_notes)
                    WHERE id = ? AND (status = 'PENDING' OR decision = 'REVIEW')
                    """,
                    (now_iso, reviewer_name, notes, review_id),
                )
                if cursor.rowcount == 0:
                    raise ValueError(f"Review '{review_id}' has already been processed by another concurrent request.")

                # Record immutable audit entry
                audit_id = f"aud-{uuid.uuid4().hex[:8]}"
                conn.execute(
                    """
                    INSERT INTO resolution_review_audit
                    (audit_id, review_id, candidate, canonical, original_resolution, original_confidence,
                     reviewer_action, previous_state, new_state, timestamp, reviewer, notes, signals)
                    VALUES (?, ?, ?, ?, ?, ?, 'APPROVE', ?, 'APPROVED', ?, ?, ?, ?)
                    """,
                    (
                        audit_id,
                        review_id,
                        row["candidate"],
                        row["canonical"],
                        row["decision"],
                        row["confidence"],
                        curr_status,
                        now_iso,
                        reviewer_name,
                        notes,
                        row["matching_signals"] if "matching_signals" in row.keys() else "{}",
                    ),
                )
                conn.commit()

            self._sync_jsonl_mirror()
            metrics_collector.record_er_review_action("approved")
            updated = self.get_review_by_id(review_id)
            if updated:
                logger.info(
                    "Approved entity resolution review",
                    id=review_id,
                    candidate=updated.get("candidate"),
                    canonical=updated.get("canonical"),
                    reviewer=reviewer_name,
                )
            return updated

    def reject_review(
        self,
        review_id: str,
        notes: Optional[str] = None,
        reviewer: Optional[str] = "human_reviewer",
    ) -> Optional[Dict[str, Any]]:
        """
        Reject an ambiguous match (candidate kept separate as distinct entity).
        Validates state transitions (fails safely if already approved or rejected).
        Persists decision, records audit entry, and updates metrics.
        """
        now_iso = datetime.now(timezone.utc).isoformat()
        reviewer_name = reviewer or "human_reviewer"

        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.execute("SELECT * FROM resolution_reviews WHERE id = ?", (review_id,))
                row = cursor.fetchone()
                if not row:
                    return None

                curr_status = (row["status"] or row["decision"]).upper()
                if curr_status == "APPROVED":
                    raise ValueError(
                        f"Cannot reject review '{review_id}' with status 'APPROVED'. Invalid state transition."
                    )
                if curr_status == "REJECTED":
                    raise ValueError(f"Review '{review_id}' is already rejected.")

                cursor = conn.execute(
                    """
                    UPDATE resolution_reviews
                    SET decision = 'REJECTED',
                        status = 'REJECTED',
                        reviewed_at = ?,
                        reviewer = ?,
                        reviewer_notes = coalesce(?, reviewer_notes)
                    WHERE id = ? AND (status = 'PENDING' OR decision = 'REVIEW')
                    """,
                    (now_iso, reviewer_name, notes, review_id),
                )
                if cursor.rowcount == 0:
                    raise ValueError(f"Review '{review_id}' has already been processed by another concurrent request.")

                # Record immutable audit entry
                audit_id = f"aud-{uuid.uuid4().hex[:8]}"
                conn.execute(
                    """
                    INSERT INTO resolution_review_audit
                    (audit_id, review_id, candidate, canonical, original_resolution, original_confidence,
                     reviewer_action, previous_state, new_state, timestamp, reviewer, notes, signals)
                    VALUES (?, ?, ?, ?, ?, ?, 'REJECT', ?, 'REJECTED', ?, ?, ?, ?)
                    """,
                    (
                        audit_id,
                        review_id,
                        row["candidate"],
                        row["canonical"],
                        row["decision"],
                        row["confidence"],
                        curr_status,
                        now_iso,
                        reviewer_name,
                        notes,
                        row["matching_signals"] if "matching_signals" in row.keys() else "{}",
                    ),
                )
                conn.commit()

            self._sync_jsonl_mirror()
            metrics_collector.record_er_review_action("rejected")
            updated = self.get_review_by_id(review_id)
            if updated:
                logger.info(
                    "Rejected entity resolution review",
                    id=review_id,
                    candidate=updated.get("candidate"),
                    canonical=updated.get("canonical"),
                    reviewer=reviewer_name,
                )
            return updated

    def get_review_audit_trail(self, review_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieve audit history records, optionally scoped to a single review item."""
        with self._get_connection() as conn:
            if review_id:
                cursor = conn.execute(
                    """
                    SELECT * FROM resolution_review_audit
                    WHERE review_id = ?
                    ORDER BY timestamp ASC
                    """,
                    (review_id,),
                )
            else:
                cursor = conn.execute(
                    "SELECT * FROM resolution_review_audit ORDER BY timestamp DESC"
                )
            rows = cursor.fetchall()
            audit_records = []
            for r in rows:
                item = dict(r)
                sig_raw = item.get("signals")
                if isinstance(sig_raw, str):
                    try:
                        item["signals"] = json.loads(sig_raw)
                    except Exception:
                        pass
                audit_records.append(item)
            return audit_records

    def list_all_reviews(self) -> List[Dict[str, Any]]:
        """Return all review records sorted by timestamp descending."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM resolution_reviews ORDER BY timestamp DESC"
            )
            return [self._row_to_dict(row) for row in cursor.fetchall()]

    def get_reviews_by_status(self, status: str) -> List[Dict[str, Any]]:
        """Filter review records by status (pending, approved, rejected, all)."""
        st = status.upper().strip()
        if st in ("ALL", "*"):
            return self.list_all_reviews()
        if st in ("PENDING", "REVIEW", "NEEDS_REVIEW"):
            return self.get_pending_reviews()
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT * FROM resolution_reviews
                WHERE status = ? OR decision = ?
                ORDER BY timestamp DESC
                """,
                (st, st),
            )
            return [self._row_to_dict(row) for row in cursor.fetchall()]

    def get_review_stats(self) -> Dict[str, Any]:
        """Return aggregated summary counts of review queue states."""
        all_revs = self.list_all_reviews()
        pending = sum(1 for r in all_revs if r.get("status") == "PENDING" or r.get("decision") == "REVIEW")
        approved = sum(1 for r in all_revs if r.get("status") == "APPROVED" or r.get("decision") == "APPROVED")
        rejected = sum(1 for r in all_revs if r.get("status") == "REJECTED" or r.get("decision") == "REJECTED")
        return {
            "total": len(all_revs),
            "pending": pending,
            "approved": approved,
            "rejected": rejected,
        }


review_queue = ReviewQueue()
