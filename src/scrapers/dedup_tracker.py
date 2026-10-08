r"""
================================================================================
DISTRIBUTED DEDUPLICATION TRACKER ARCHITECTURE
================================================================================

This module maintains persistent tracking of seen URLs and content hashes to
guarantee that re-running scrapers never reprocesses duplicate articles or jobs.

--------------------------------------------------------------------------------
1. LOCAL STANDALONE MODEL (SQLITE STORE)
--------------------------------------------------------------------------------
For single-node ingestion, state is persisted locally in SQLite at
`data/processed/dedup_tracker.db`. The table indexes normalized URL strings and
SHA-256 content fingerprints.

--------------------------------------------------------------------------------
2. DISTRIBUTED PRODUCTION MODEL (REDIS SETS & BLOOM FILTERS)
--------------------------------------------------------------------------------
When scaling to horizontal worker pods across multiple nodes, SQLite is upgraded
to a centralized distributed in-memory cache layer:

  +-----------------------+     SISMEMBER seen:urls <url>     +-------------------+
  | Crawler Worker Pod 1  | --------------------------------> | Redis Cluster /   |
  +-----------------------+ <-------------------------------- | Redis Bloom       |
                                  (0 = unseen, 1 = seen)      +-------------------+
  +-----------------------+                                             ^
  | Crawler Worker Pod N  | --------------------------------------------+
  +-----------------------+             SADD seen:urls <url>

Production Upgrade Interface:
  - Redis Data Structure:  Redis SET (`SADD`, `SISMEMBER`) or `BF.ADD` (Bloom Filter).
  - Time-To-Live (TTL):    Set key TTL to 30 days to bound memory allocation while
                           preventing duplicate crawling.
================================================================================
"""

import hashlib
import re
import sqlite3
import unicodedata
from enum import Enum
from pathlib import Path
from typing import NamedTuple, Optional

import structlog
from config.settings import settings

logger = structlog.get_logger(__name__)

DB_PATH = settings.DATA_PROCESSED_DIR / "dedup_tracker.db"


class ProcessingDecision(str, Enum):
    """
    Explicit incremental processing decision states:
      - NEW: No previous fingerprint exists for this URL -> Full processing required.
      - CHANGED: URL was previously processed, but normalized content hash has changed -> Reprocessing required.
      - UNCHANGED: URL was previously processed and current content hash matches -> Skip expensive downstream work.
    """
    NEW = "NEW"
    CHANGED = "CHANGED"
    UNCHANGED = "UNCHANGED"


class ProcessingAction(str, Enum):
    """Pipeline execution action based on incremental evaluation."""
    PROCESS = "PROCESS"
    SKIP = "SKIP"


# Backward-compatibility aliases for Phase 1 code
IncrementalDecision = ProcessingDecision
IncrementalAction = ProcessingAction


class IncrementalCheckResult(NamedTuple):
    decision: ProcessingDecision
    action: ProcessingAction
    url: str
    content_hash: Optional[str] = None
    stored_hash: Optional[str] = None

    @property
    def is_skip(self) -> bool:
        """Return True if decision directs the pipeline to skip downstream work."""
        return self.action == ProcessingAction.SKIP

    @property
    def should_process(self) -> bool:
        """Return True if record requires downstream processing."""
        return self.action == ProcessingAction.PROCESS


class DedupTracker:
    """Persistent SQLite deduplication and incremental processing tracker for cross-run checking."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or (settings.DATA_PROCESSED_DIR / "dedup_tracker.db")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self.db_path), timeout=30.0)

    def _init_db(self):
        """Create sqlite deduplication table and indexes if missing, migrating columns safely."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS seen_records (
                    url_hash TEXT PRIMARY KEY,
                    url TEXT NOT NULL,
                    content_hash TEXT,
                    source TEXT,
                    first_seen_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_seen_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                """
            )
            # Safely check and add last_seen_at if existing table lacked it
            cursor.execute("PRAGMA table_info(seen_records);")
            columns = [row[1] for row in cursor.fetchall()]
            if "last_seen_at" not in columns:
                cursor.execute("ALTER TABLE seen_records ADD COLUMN last_seen_at TIMESTAMP;")

            cursor.execute("CREATE INDEX IF NOT EXISTS idx_content_hash ON seen_records(content_hash);")
            conn.commit()

    @staticmethod
    def normalize_content(text: Optional[str]) -> str:
        """
        Normalize textual or HTML document content to produce deterministic, stable fingerprints.

        What is normalized and why:
        1. Unicode normalization (NFKC): Unifies equivalent Unicode representations (e.g., composite
           accented characters vs combining marks, ligature forms) so semantically identical text
           from different web scrapers/parsers produces identical byte sequences.
        2. HTML/XML tag gap stripping ('>\\s+<') -> '><': Eliminates arbitrary layout differences
           such as indentation, pretty-printing, and whitespace inserted between HTML tags that do not
           affect page semantics.
        3. Collapsing repeated whitespace (\\s+ -> ' '): Collapses spaces, tabs (\\t), carriage returns (\\r),
           and newlines (\\n) into a single space, neutralizing cosmetic formatting changes.
        4. Trimming: Strips leading and trailing whitespace from the document boundaries.

        Why: Web crawling frequently captures superficial formatting artifacts (e.g. server re-rendering,
        beautified HTML, CDN whitespace variations). Normalization guarantees that only genuine content
        updates trigger expensive downstream LLM extraction and vector indexing.
        """
        if not text:
            return ""
        # 1. Unicode NFKC normalization
        normalized = unicodedata.normalize("NFKC", text)
        # 2. Normalize whitespace inside tag brackets: < div >, </ div > -> <div>, </div>
        cleaned = re.sub(r"<\s*/\s*", "</", normalized)
        cleaned = re.sub(r"<\s+", "<", cleaned)
        cleaned = re.sub(r"\s+>", ">", cleaned)
        # 3. Strip whitespace immediately adjacent to tag boundaries and between tags
        cleaned = re.sub(r">\s+", ">", cleaned)
        cleaned = re.sub(r"\s+<", "<", cleaned)
        # 4. Collapse multiple internal whitespace characters (spaces, tabs, newlines, carriage returns)
        cleaned = re.sub(r"\s+", " ", cleaned)
        # 5. Strip edges
        return cleaned.strip()

    @classmethod
    def compute_content_hash(cls, content: Optional[str]) -> str:
        """
        Compute deterministic SHA-256 hex digest of normalized content.
        Content is passed through normalize_content(), UTF-8 encoded, and digested.
        """
        normalized = cls.normalize_content(content)
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    @staticmethod
    def hash_string(text: str) -> str:
        """Compute SHA-256 hash of raw string."""
        return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()

    def check_incremental(
        self,
        url: str,
        content: Optional[str] = None,
        content_hash: Optional[str] = None,
    ) -> IncrementalCheckResult:
        """
        Evaluate incremental processing decision for a given URL and content.

        Decisions:
          - NEW       -> PROCESS (URL not yet seen)
          - CHANGED   -> PROCESS (URL seen, but normalized content hash differs)
          - UNCHANGED -> SKIP    (URL seen, and normalized content hash matches stored hash)
        """
        url_key = self.hash_string(url)
        current_hash = content_hash or (self.compute_content_hash(content) if content is not None else None)

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT content_hash FROM seen_records WHERE url_hash = ?", (url_key,))
            row = cursor.fetchone()

        if row is None:
            decision = ProcessingDecision.NEW
            action = ProcessingAction.PROCESS
            stored_hash = None
        else:
            stored_hash = row[0]
            if current_hash is not None and stored_hash is not None:
                if current_hash == stored_hash:
                    decision = ProcessingDecision.UNCHANGED
                    action = ProcessingAction.SKIP
                else:
                    decision = ProcessingDecision.CHANGED
                    action = ProcessingAction.PROCESS
            elif current_hash is not None and stored_hash is None:
                decision = ProcessingDecision.CHANGED
                action = ProcessingAction.PROCESS
            else:
                # Content not provided for comparison; treat as UNCHANGED if URL exists
                decision = ProcessingDecision.UNCHANGED
                action = ProcessingAction.SKIP

        logger.info(
            "Incremental processing decision evaluated",
            url=url,
            decision=decision.value,
            action=action.value,
            content_hash=current_hash,
            stored_hash=stored_hash,
        )

        return IncrementalCheckResult(
            decision=decision,
            action=action,
            url=url,
            content_hash=current_hash,
            stored_hash=stored_hash,
        )

    def save_record(
        self,
        url: str,
        content: Optional[str] = None,
        content_hash: Optional[str] = None,
        source: Optional[str] = None,
    ) -> None:
        """Persist or update URL and content fingerprint in seen_records table."""
        url_key = self.hash_string(url)
        c_key = content_hash or (self.compute_content_hash(content) if content is not None else None)

        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO seen_records (url_hash, url, content_hash, source, first_seen_at, last_seen_at)
                    VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                    ON CONFLICT(url_hash) DO UPDATE SET
                        content_hash = COALESCE(excluded.content_hash, seen_records.content_hash),
                        source = COALESCE(excluded.source, seen_records.source),
                        last_seen_at = CURRENT_TIMESTAMP;
                    """,
                    (url_key, url, c_key, source),
                )
                conn.commit()
        except Exception as e:
            logger.error("Failed to save record in DedupTracker", url=url, error=str(e))

    def record_successful_processing(
        self,
        url: str,
        content: Optional[str] = None,
        content_hash: Optional[str] = None,
        source: Optional[str] = None,
    ) -> None:
        """
        Persist or update URL and content fingerprint in seen_records table ONLY
        after downstream processing completes successfully.
        """
        self.save_record(url=url, content=content, content_hash=content_hash, source=source)

    def record_failed_processing(
        self,
        url: str,
        reason: Optional[str] = None,
    ) -> None:
        """
        Audit log that downstream processing failed.
        CRITICAL FAILURE SEMANTICS:
        - If NEW processing fails: Does NOT create or update fingerprint in seen_records.
        - If CHANGED processing fails: Does NOT replace previous successful fingerprint.
        - Preserves clean retry state for subsequent pipeline runs.
        """
        logger.warning(
            "Downstream processing failed; fingerprint not committed",
            url=url,
            reason=reason,
        )

    def get_stored_hash(self, url: str) -> Optional[str]:
        """Retrieve stored content hash for a given URL from seen_records, if present."""
        url_key = self.hash_string(url)
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT content_hash FROM seen_records WHERE url_hash = ?", (url_key,))
            row = cursor.fetchone()
            return row[0] if row else None

    def is_seen(self, url: str, content_hash: Optional[str] = None) -> bool:
        """Check if URL or content hash has already been processed."""
        url_key = self.hash_string(url)
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Check URL hash match
            cursor.execute("SELECT 1 FROM seen_records WHERE url_hash = ?", (url_key,))
            if cursor.fetchone() is not None:
                return True

            # Check content hash match if provided
            if content_hash:
                c_key = self.hash_string(content_hash)
                cursor.execute("SELECT 1 FROM seen_records WHERE content_hash = ?", (c_key,))
                if cursor.fetchone() is not None:
                    return True

        return False

    def mark_seen(self, url: str, content_hash: Optional[str] = None, source: Optional[str] = None):
        """Record URL and optional content hash as seen."""
        self.save_record(url=url, content_hash=content_hash, source=source)
