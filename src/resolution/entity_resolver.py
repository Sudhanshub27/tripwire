import json
import re
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import structlog
from config.settings import settings
from pydantic import BaseModel
from rapidfuzz import fuzz

from src.resolution.review_queue import ReviewQueue
from src.resolution.review_queue import review_queue as default_review_queue

logger = structlog.get_logger(__name__)

SEED_FILE = settings.BASE_DIR / "data" / "seed" / "canonical_startups.json"
MAPPING_LOG_FILE = settings.DATA_PROCESSED_DIR / "entity_mapping_log.jsonl"

# Corporate legal suffix regex pattern
LEGAL_SUFFIXES_REGEX = re.compile(
    r"\b(inc|incorporated|ltd|limited|corp|corporation|llc|co|company|pte|sas|pbc|gmbh|se|bv)\b",
    re.IGNORECASE,
)

# Empirically calibrated thresholds derived from evaluation/reports/resolution_eval.json:
# Score >= 85.0: Precision 100.0%, 0 false positives -> Auto MATCH
# Score in [65.0, 85.0): Ambiguous edge cases -> Human REVIEW
# Score < 65.0: Negligible similarity -> Distinct NEW entity
DEFAULT_MATCH_THRESHOLD = 85.0
DEFAULT_REVIEW_THRESHOLD = 65.0


class ResolutionDecision(str, Enum):
    MATCH = "MATCH"
    REVIEW = "REVIEW"
    NEW = "NEW"


def normalize_entity_name(name: str) -> str:
    """
    Normalize entity name for matching:
    - Convert to lowercase
    - Strip corporate legal suffixes (Inc, Ltd, LLC, Corp, etc.)
    - Remove punctuation
    - Collapse whitespace
    """
    if not name:
        return ""
    # Lowercase
    s = name.lower()
    # Strip legal suffixes
    s = LEGAL_SUFFIXES_REGEX.sub("", s)
    # Strip punctuation
    s = re.sub(r"[^\w\s]", "", s)
    # Collapse whitespace
    s = " ".join(s.split())
    return s


class ResolutionResult:
    """
    Result of entity resolution decision supporting both tuple unpacking
    (canonical, method, conf) for backwards compatibility and attribute access.
    """

    def __init__(
        self,
        canonical: Optional[str],
        method: str,
        confidence: float,
        decision: str,
        fuzzy_score: float,
        candidate: str,
        match_reason: str = "",
    ):
        self.canonical = canonical
        self.canonical_name = canonical
        self.matched_entity = canonical
        self.candidate = candidate
        self.candidate_name = candidate
        self.method = method
        self.method_used = method
        self.matching_method = method
        self.confidence = round(float(confidence), 4)
        self.confidence_score = round(float(confidence), 4)
        self.fuzzy_score = round(float(fuzzy_score), 4)
        self.decision = decision
        self.final_decision = decision
        self.match_reason = match_reason

    def __iter__(self):
        # Unpacks (canonical_for_merge, method, confidence)
        canonical_val = self.canonical if self.decision == ResolutionDecision.MATCH.value else None
        return iter((canonical_val, self.method_used, self.confidence_score))

    def __getitem__(self, idx: int):
        canonical_val = self.canonical if self.decision == ResolutionDecision.MATCH.value else None
        return (canonical_val, self.method_used, self.confidence_score)[idx]

    def __len__(self) -> int:
        return 3

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, tuple) and len(other) == 3:
            canonical_val = self.canonical if self.decision == ResolutionDecision.MATCH.value else None
            return (canonical_val, self.method_used, self.confidence_score) == other
        if isinstance(other, ResolutionResult):
            return (
                self.canonical == other.canonical
                and self.method_used == other.method_used
                and self.confidence_score == other.confidence_score
                and self.decision == other.decision
            )
        return False

    def __repr__(self) -> str:
        return (
            f"ResolutionResult(candidate={self.candidate!r}, "
            f"canonical={self.canonical!r}, decision={self.decision!r}, "
            f"method={self.method_used!r}, confidence={self.confidence_score}, "
            f"fuzzy_score={self.fuzzy_score})"
        )


class EntityResolver:
    """
    Multi-stage entity resolution engine mapping raw organization names
    to canonical entities using exact, normalized, and fuzzy matching,
    governed by an empirical confidence-based decision model (MATCH / REVIEW / NEW).
    """

    def __init__(
        self,
        seed_file: Optional[Path] = None,
        fuzzy_threshold: float = DEFAULT_MATCH_THRESHOLD,
        review_threshold: float = DEFAULT_REVIEW_THRESHOLD,
        queue: Optional[ReviewQueue] = None,
    ):
        self.seed_file = seed_file or SEED_FILE
        self.fuzzy_threshold = fuzzy_threshold
        self.review_threshold = review_threshold
        self.review_queue = queue or default_review_queue
        self.canonical_entities: List[Dict[str, Any]] = []
        # Fast lookup indexes
        self.exact_map: Dict[str, str] = {}  # lowercase raw alias -> canonical name
        self.norm_map: Dict[str, str] = {}   # normalized alias -> canonical name
        self.all_variants: List[Tuple[str, str]] = []  # [(variant_string, canonical_name)]
        self._load_seed_data()

    def _load_seed_data(self):
        """Load canonical startups and aliases into lookup index tables."""
        if not self.seed_file.exists():
            logger.warning("Seed file missing, initializing empty resolver", path=str(self.seed_file))
            return

        try:
            with open(self.seed_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.canonical_entities = data.get("startups", [])

            for item in self.canonical_entities:
                canonical = item["canonical_name"]
                aliases = item.get("aliases", [canonical])

                # Ensure canonical itself is in aliases
                if canonical not in aliases:
                    aliases.append(canonical)

                for alias in aliases:
                    self._register_alias(alias, canonical)

            logger.info("Loaded canonical entity index", total_canonical=len(self.canonical_entities))

        except Exception as e:
            logger.error("Failed to load canonical startups seed file", error=str(e))

    def _register_alias(self, alias: str, canonical: str) -> None:
        """Register an alias into fast lookup maps and variants list."""
        raw_lower = alias.lower().strip()
        norm_alias = normalize_entity_name(alias)

        if raw_lower:
            self.exact_map[raw_lower] = canonical
        if norm_alias:
            self.norm_map[norm_alias] = canonical
            if (norm_alias, canonical) not in self.all_variants:
                self.all_variants.append((norm_alias, canonical))

    def add_approved_alias(self, candidate: str, canonical: str) -> None:
        """
        Dynamically register an approved alias into the resolver index.
        Subsequent resolutions of candidate will resolve to canonical automatically as MATCH.
        """
        self._register_alias(candidate, canonical)
        self.log_decision(
            raw_name=candidate,
            canonical_name=canonical,
            method_used="approved_alias",
            confidence_score=1.0,
            fuzzy_score=1.0,
            decision=ResolutionDecision.MATCH.value,
            source="review_approval",
            match_reason="Approved through human review queue",
        )
        logger.info("Registered approved entity alias", candidate=candidate, canonical=canonical)

    def log_decision(
        self,
        raw_name: str,
        canonical_name: Optional[str],
        method_used: str,
        confidence_score: float,
        fuzzy_score: float = 0.0,
        decision: str = ResolutionDecision.MATCH.value,
        source: str = "pipeline",
        match_reason: str = "",
    ) -> Dict[str, Any]:
        """
        Log entity resolution decision with complete audit fields to
        data/processed/entity_mapping_log.jsonl.
        """
        settings.DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        record_id = f"er-{abs(hash(raw_name)) % 10000:04d}"
        status = (
            "merged"
            if decision == ResolutionDecision.MATCH.value
            else ("needs_review" if decision == ResolutionDecision.REVIEW.value else "kept_separate")
        )

        record = {
            "id": record_id,
            "candidate": raw_name,
            "canonical": canonical_name,
            "candidate_entity": raw_name,
            "matched_entity": canonical_name,
            "matching_method": method_used,
            "method_used": method_used,
            "fuzzy_score": round(float(fuzzy_score), 4),
            "confidence_score": round(float(confidence_score), 4),
            "confidence": round(float(confidence_score), 4),
            "final_decision": decision,
            "decision": decision,
            "status": status,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source": source,
            "match_reason": match_reason,
            "entity_name": canonical_name or raw_name,
            "raw_name": raw_name,
            "canonical_name": canonical_name,
            "entity_type": "startup",
        }

        try:
            with open(MAPPING_LOG_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        except Exception as e:
            logger.error("Failed to write to entity_mapping_log.jsonl", error=str(e))

        logger.info(
            "Entity resolution decision",
            candidate=raw_name,
            canonical=canonical_name,
            decision=decision,
            method=method_used,
            confidence=record["confidence_score"],
            fuzzy=record["fuzzy_score"],
        )
        return record

    def resolve(
        self,
        raw_name: str,
        source: str = "pipeline",
    ) -> ResolutionResult:
        """
        Resolve raw organization name against canonical entities using
        empirical confidence decision model: MATCH, REVIEW, or NEW.

        Returns ResolutionResult object (which unpacks as (canonical, method, confidence)).
        """
        if not raw_name or not raw_name.strip():
            self.log_decision(
                raw_name=raw_name,
                canonical_name=None,
                method_used="unresolved",
                confidence_score=0.0,
                fuzzy_score=0.0,
                decision=ResolutionDecision.NEW.value,
                source=source,
                match_reason="Empty input name",
            )
            return ResolutionResult(
                canonical=None,
                method="unresolved",
                confidence=0.0,
                decision=ResolutionDecision.NEW.value,
                fuzzy_score=0.0,
                candidate=raw_name,
                match_reason="Empty input name",
            )

        clean_raw = raw_name.strip()
        raw_lower = clean_raw.lower()

        # Step 1: Exact Match (Fast Path -> High Precision MATCH)
        if raw_lower in self.exact_map:
            canonical = self.exact_map[raw_lower]
            reason = "Exact alias match in canonical index"
            self.log_decision(
                raw_name=clean_raw,
                canonical_name=canonical,
                method_used="exact",
                confidence_score=1.0,
                fuzzy_score=1.0,
                decision=ResolutionDecision.MATCH.value,
                source=source,
                match_reason=reason,
            )
            return ResolutionResult(
                canonical=canonical,
                method="exact",
                confidence=1.0,
                decision=ResolutionDecision.MATCH.value,
                fuzzy_score=1.0,
                candidate=clean_raw,
                match_reason=reason,
            )

        # Step 2: Normalized Match (Legal Suffixes & Punctuation Stripped -> High Precision MATCH)
        norm_raw = normalize_entity_name(clean_raw)
        if norm_raw in self.norm_map:
            canonical = self.norm_map[norm_raw]
            reason = "Normalized string match after legal suffix and punctuation removal"
            self.log_decision(
                raw_name=clean_raw,
                canonical_name=canonical,
                method_used="normalized",
                confidence_score=0.95,
                fuzzy_score=1.0,
                decision=ResolutionDecision.MATCH.value,
                source=source,
                match_reason=reason,
            )
            return ResolutionResult(
                canonical=canonical,
                method="normalized",
                confidence=0.95,
                decision=ResolutionDecision.MATCH.value,
                fuzzy_score=1.0,
                candidate=clean_raw,
                match_reason=reason,
            )

        # Step 3: Fuzzy Match (RapidFuzz token_sort_ratio across canonical variants)
        best_canonical = None
        best_score = 0.0

        for norm_variant, canonical in self.all_variants:
            score = fuzz.token_sort_ratio(norm_raw, norm_variant)
            if score > best_score:
                best_score = score
                best_canonical = canonical

        fuzzy_ratio = round(best_score / 100.0, 4)
        confidence = fuzzy_ratio

        # Step 4: Confidence-Based Decision Routing
        # Branch 4a: Automatic MATCH (Empirically verified >= 85.0 has 100% precision)
        if best_score >= self.fuzzy_threshold and best_canonical:
            reason = f"High fuzzy similarity ({best_score:.1f}%) >= threshold ({self.fuzzy_threshold:.1f}%)"
            self.log_decision(
                raw_name=clean_raw,
                canonical_name=best_canonical,
                method_used="fuzzy",
                confidence_score=confidence,
                fuzzy_score=fuzzy_ratio,
                decision=ResolutionDecision.MATCH.value,
                source=source,
                match_reason=reason,
            )
            return ResolutionResult(
                canonical=best_canonical,
                method="fuzzy",
                confidence=confidence,
                decision=ResolutionDecision.MATCH.value,
                fuzzy_score=fuzzy_ratio,
                candidate=clean_raw,
                match_reason=reason,
            )

        # Branch 4b: Ambiguous Match -> Flag for Human REVIEW and enqueue
        if best_score >= self.review_threshold and best_canonical:
            reason = (
                f"Ambiguous fuzzy similarity ({best_score:.1f}%) in review range "
                f"[{self.review_threshold:.1f}%, {self.fuzzy_threshold:.1f}%)"
            )
            # Enqueue into persistent review queue
            self.review_queue.add_review(
                candidate=clean_raw,
                canonical=best_canonical,
                matching_method="fuzzy",
                fuzzy_score=fuzzy_ratio,
                confidence=confidence,
                source=source,
                match_reason=reason,
            )
            self.log_decision(
                raw_name=clean_raw,
                canonical_name=best_canonical,
                method_used="fuzzy",
                confidence_score=confidence,
                fuzzy_score=fuzzy_ratio,
                decision=ResolutionDecision.REVIEW.value,
                source=source,
                match_reason=reason,
            )
            return ResolutionResult(
                canonical=best_canonical,
                method="fuzzy",
                confidence=confidence,
                decision=ResolutionDecision.REVIEW.value,
                fuzzy_score=fuzzy_ratio,
                candidate=clean_raw,
                match_reason=reason,
            )

        # Branch 4c: Low Similarity -> Treat as distinct NEW entity
        reason = f"Low fuzzy similarity ({best_score:.1f}%) below review threshold ({self.review_threshold:.1f}%)"
        self.log_decision(
            raw_name=clean_raw,
            canonical_name=None,
            method_used="unresolved",
            confidence_score=confidence,
            fuzzy_score=fuzzy_ratio,
            decision=ResolutionDecision.NEW.value,
            source=source,
            match_reason=reason,
        )
        return ResolutionResult(
            canonical=None,
            method="unresolved",
            confidence=confidence,
            decision=ResolutionDecision.NEW.value,
            fuzzy_score=fuzzy_ratio,
            candidate=clean_raw,
            match_reason=reason,
        )

    def resolve_record(
        self,
        record: BaseModel,
        source: str = "pipeline",
    ) -> Tuple[BaseModel, Dict[str, Any]]:
        """
        Apply entity resolver to a Startup or Product Pydantic schema model object,
        updating entityName / maker_company to canonical form when resolved with MATCH decision.
        """
        record_type = getattr(record, "recordType", "")
        raw_target_name = None
        target_field = None

        if record_type == "startup" and hasattr(record, "name"):
            raw_target_name = getattr(record, "name")
            target_field = "name"
        elif record_type == "product" and hasattr(record, "maker_company"):
            raw_target_name = getattr(record, "maker_company")
            target_field = "maker_company"

        if not raw_target_name or not target_field:
            return record, {
                "method_used": "bypassed",
                "matching_method": "bypassed",
                "confidence_score": 1.0,
                "confidence": 1.0,
                "fuzzy_score": 1.0,
                "decision": ResolutionDecision.NEW.value,
            }

        res = self.resolve(raw_target_name, source=source)

        # Only merge into canonical form if final decision is explicit MATCH
        if res.decision == ResolutionDecision.MATCH.value and res.canonical and target_field:
            setattr(record, target_field, res.canonical)

        decision_info = {
            "raw_name": raw_target_name,
            "candidate": raw_target_name,
            "canonical_name": res.canonical if res.decision == ResolutionDecision.MATCH.value else None,
            "matched_entity": res.canonical,
            "method_used": res.method_used,
            "matching_method": res.method_used,
            "confidence_score": res.confidence_score,
            "confidence": res.confidence_score,
            "fuzzy_score": res.fuzzy_score,
            "decision": res.decision,
            "final_decision": res.decision,
            "match_reason": res.match_reason,
        }
        return record, decision_info
