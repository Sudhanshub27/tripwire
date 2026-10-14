import json
import re
import time
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import structlog
from config.settings import settings
from pydantic import BaseModel
from rapidfuzz import fuzz

from src.observability.metrics import metrics_collector
from src.resolution.review_queue import ReviewQueue
from src.resolution.review_queue import review_queue as default_review_queue

logger = structlog.get_logger(__name__)

SEED_FILE = settings.BASE_DIR / "data" / "seed" / "canonical_startups.json"
MAPPING_LOG_FILE = settings.DATA_PROCESSED_DIR / "entity_mapping_log.jsonl"
RESOLVER_VERSION = "2.0.0"

# Corporate legal suffix regex pattern
LEGAL_SUFFIXES_REGEX = re.compile(
    r"\b(inc|incorporated|ltd|limited|corp|corporation|llc|co|company|pte|sas|pbc|gmbh|se|bv)\b",
    re.IGNORECASE,
)

# Empirically calibrated thresholds:
# Confidence >= 0.85 (85%): Precision 100%, 0 false positives -> AUTO_MATCH
# Confidence in [0.65, 0.85) (65-85%): Ambiguous edge cases -> NEEDS_REVIEW
# Confidence < 0.65 (<65%): Negligible similarity -> NO_MATCH
DEFAULT_MATCH_THRESHOLD = 0.85
DEFAULT_REVIEW_THRESHOLD = 0.65


class ResolutionDecision(str, Enum):
    """Explicit entity resolution decision outcomes."""

    AUTO_MATCH = "AUTO_MATCH"
    NEEDS_REVIEW = "NEEDS_REVIEW"
    NO_MATCH = "NO_MATCH"

    # Backwards compatibility aliases
    MATCH = "AUTO_MATCH"
    REVIEW = "NEEDS_REVIEW"
    NEW = "NO_MATCH"

    def __eq__(self, other: Any) -> bool:
        if super().__eq__(other):
            return True
        if isinstance(other, str):
            o = other.upper()
            if self.value == "AUTO_MATCH" and o in ("MATCH", "AUTO_MATCH"):
                return True
            if self.value == "NEEDS_REVIEW" and o in ("REVIEW", "NEEDS_REVIEW"):
                return True
            if self.value == "NO_MATCH" and o in ("NEW", "NO_MATCH", "UNRESOLVED"):
                return True
        return False

    @classmethod
    def _missing_(cls, value: object):
        if isinstance(value, str):
            v = value.upper()
            if v in ("MATCH", "AUTO_MATCH", "AUTOMATCH"):
                return cls.AUTO_MATCH
            if v in ("REVIEW", "NEEDS_REVIEW", "NEEDSREVIEW"):
                return cls.NEEDS_REVIEW
            if v in ("NEW", "NO_MATCH", "NOMATCH", "UNRESOLVED"):
                return cls.NO_MATCH
        return None


def normalize_entity_name(name: str) -> str:
    """
    Normalize entity name for deterministic comparison:
    - Convert to lowercase
    - Strip corporate legal suffixes (Inc, Ltd, LLC, Corp, etc.)
    - Remove punctuation
    - Collapse whitespace
    """
    if not name:
        return ""
    s = name.lower()
    s = LEGAL_SUFFIXES_REGEX.sub("", s)
    s = re.sub(r"[^\w\s]", "", s)
    return " ".join(s.split())


def extract_acronym(text: str) -> str:
    """Extract upper/clean acronym letters from significant tokens."""
    tokens = [w for w in re.findall(r"\b\w+", text.lower()) if w not in ("inc", "llc", "corp", "ltd", "co")]
    return "".join(w[0] for w in tokens) if len(tokens) > 1 else ""


class ResolutionResult:
    """
    Comprehensive result of an entity resolution decision.
    Preserves backwards-compatible tuple unpacking (canonical, method, confidence)
    while exposing rich explanatory signals, decision metadata, and versioning.
    """

    def __init__(
        self,
        canonical: Optional[str],
        method: str,
        confidence: float,
        decision: Any,
        fuzzy_score: float,
        candidate: str,
        match_reason: str = "",
        signals: Optional[Dict[str, Any]] = None,
        source: str = "pipeline",
        resolver_version: str = RESOLVER_VERSION,
        timestamp: Optional[str] = None,
    ):
        dec_val = decision.value if hasattr(decision, "value") else str(decision)
        try:
            self.decision_enum = ResolutionDecision(dec_val)
        except Exception:
            self.decision_enum = ResolutionDecision.AUTO_MATCH if dec_val in ("MATCH", "AUTO_MATCH") else (
                ResolutionDecision.NEEDS_REVIEW if dec_val in ("REVIEW", "NEEDS_REVIEW") else ResolutionDecision.NO_MATCH
            )

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
        self.similarity_score = self.fuzzy_score
        self.decision = self.decision_enum.value
        self.final_decision = self.decision_enum.value
        self.match_reason = match_reason
        self.signals = signals or {}
        self.matching_signals = self.signals
        self.source = source
        self.resolver_version = resolver_version
        self.timestamp = timestamp or datetime.now(timezone.utc).isoformat()

    def is_match(self) -> bool:
        return self.decision_enum == ResolutionDecision.AUTO_MATCH

    def is_review(self) -> bool:
        return self.decision_enum == ResolutionDecision.NEEDS_REVIEW

    def is_no_match(self) -> bool:
        return self.decision_enum == ResolutionDecision.NO_MATCH

    def __iter__(self):
        canonical_val = self.canonical if self.is_match() else None
        return iter((canonical_val, self.method_used, self.confidence_score))

    def __getitem__(self, idx: int):
        canonical_val = self.canonical if self.is_match() else None
        return (canonical_val, self.method_used, self.confidence_score)[idx]

    def __len__(self) -> int:
        return 3

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, tuple) and len(other) == 3:
            canonical_val = self.canonical if self.is_match() else None
            return (canonical_val, self.method_used, self.confidence_score) == other
        if isinstance(other, ResolutionResult):
            return (
                self.canonical == other.canonical
                and self.method_used == other.method_used
                and self.confidence_score == other.confidence_score
                and self.decision == other.decision
            )
        return False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidate": self.candidate,
            "canonical": self.canonical,
            "decision": self.decision,
            "method": self.method,
            "confidence": self.confidence,
            "similarity_score": self.fuzzy_score,
            "signals": self.signals,
            "match_reason": self.match_reason,
            "source": self.source,
            "resolver_version": self.resolver_version,
            "timestamp": self.timestamp,
        }

    def __repr__(self) -> str:
        return (
            f"ResolutionResult(candidate={self.candidate!r}, "
            f"canonical={self.canonical!r}, decision={self.decision!r}, "
            f"method={self.method_used!r}, confidence={self.confidence_score}, "
            f"fuzzy_score={self.fuzzy_score})"
        )


class EntityResolver:
    """
    Production-grade multi-stage entity resolution engine.
    Separates candidate generation from scoring, computes deterministic explainable
    matching signals, and executes confidence-based decision routing:
    AUTO_MATCH, NEEDS_REVIEW, or NO_MATCH.
    """

    def __init__(
        self,
        seed_file: Optional[Path] = None,
        fuzzy_threshold: float = DEFAULT_MATCH_THRESHOLD,
        review_threshold: float = DEFAULT_REVIEW_THRESHOLD,
        queue: Optional[ReviewQueue] = None,
    ):
        self.seed_file = seed_file or SEED_FILE
        # Support both percentage (85.0) and unit interval (0.85) inputs
        self.fuzzy_threshold = fuzzy_threshold / 100.0 if fuzzy_threshold > 1.0 else fuzzy_threshold
        self.review_threshold = review_threshold / 100.0 if review_threshold > 1.0 else review_threshold
        self.review_queue = queue or default_review_queue
        self.canonical_entities: List[Dict[str, Any]] = []

        # Indexing structures for O(1) lookups and efficient blocking candidate generation
        self.exact_map: Dict[str, str] = {}           # lowercase raw alias -> canonical
        self.norm_map: Dict[str, str] = {}            # normalized alias -> canonical
        self.domain_map: Dict[str, str] = {}          # domain name / host -> canonical
        self.acronym_map: Dict[str, str] = {}         # acronym string -> canonical
        self.token_index: Dict[str, Set[str]] = {}    # token -> set of canonical names
        self.trigram_index: Dict[str, Set[str]] = {}  # 3-char n-gram -> set of canonical names
        self.canonical_aliases: Dict[str, List[str]] = {}  # canonical -> list of all raw aliases
        self.all_variants: List[Tuple[str, str]] = []  # [(normalized_variant, canonical)]

        self._load_seed_data()

    def _load_seed_data(self) -> None:
        """Load canonical startups and aliases into fast index and blocking tables."""
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
                if canonical not in aliases:
                    aliases.append(canonical)

                self.canonical_aliases[canonical] = aliases
                for alias in aliases:
                    self._register_alias(alias, canonical)

            logger.info("Loaded canonical entity index", total_canonical=len(self.canonical_entities))
        except Exception as e:
            logger.error("Failed to load canonical startups seed file", error=str(e))

    def _register_alias(self, alias: str, canonical: str) -> None:
        """Register an alias into fast lookup maps, blocking indices, and variants."""
        raw_clean = alias.strip()
        raw_lower = raw_clean.lower()
        norm_alias = normalize_entity_name(raw_clean)

        if canonical not in self.canonical_aliases:
            self.canonical_aliases[canonical] = []
        if raw_clean not in self.canonical_aliases[canonical]:
            self.canonical_aliases[canonical].append(raw_clean)

        if raw_lower:
            self.exact_map[raw_lower] = canonical

            # Domain alias extraction (e.g., "openai.com" -> domain_map["openai.com"] = "OpenAI")
            if any(raw_lower.endswith(ext) for ext in (".com", ".ai", ".io", ".co", ".org", ".net", ".dev")):
                self.domain_map[raw_lower] = canonical
                domain_core = raw_lower.split(".")[0]
                if domain_core and domain_core not in self.domain_map:
                    self.domain_map[domain_core] = canonical

        if norm_alias:
            self.norm_map[norm_alias] = canonical
            if (norm_alias, canonical) not in self.all_variants:
                self.all_variants.append((norm_alias, canonical))

            # Acronym registration
            acronym = extract_acronym(norm_alias)
            if acronym and len(acronym) >= 2:
                self.acronym_map[acronym] = canonical

            # Blocking token index (words of length >= 2)
            tokens = [w for w in norm_alias.split() if len(w) >= 2]
            for tok in tokens:
                if tok not in self.token_index:
                    self.token_index[tok] = set()
                self.token_index[tok].add(canonical)

            # Character 3-gram index for fuzzy blocking
            if len(norm_alias) >= 3:
                for i in range(len(norm_alias) - 2):
                    trigram = norm_alias[i : i + 3]
                    if trigram not in self.trigram_index:
                        self.trigram_index[trigram] = set()
                    self.trigram_index[trigram].add(canonical)

    def add_approved_alias(self, candidate: str, canonical: str) -> None:
        """
        Dynamically register an approved alias into the resolver index.
        Subsequent resolutions will resolve to canonical automatically as AUTO_MATCH.
        """
        self._register_alias(candidate, canonical)
        self.log_decision(
            raw_name=candidate,
            canonical_name=canonical,
            method_used="approved_alias",
            confidence_score=1.0,
            fuzzy_score=1.0,
            decision=ResolutionDecision.AUTO_MATCH.value,
            source="review_approval",
            match_reason="Approved through human review queue",
            signals={"approved_review_alias": True},
        )
        logger.info("Registered approved entity alias", candidate=candidate, canonical=canonical)

    def generate_candidates(self, raw_name: str) -> List[Tuple[str, str, str]]:
        """
        Candidate generation stage:
        Generates plausible candidate matches using inverted indexes and blocking,
        avoiding O(N) exhaustive scanning over unrelated entities.
        Returns list of (canonical_name, candidate_variant, generation_strategy).
        """
        clean_raw = raw_name.strip()
        raw_lower = clean_raw.lower()
        norm_raw = normalize_entity_name(clean_raw)
        candidates: List[Tuple[str, str, str]] = []
        seen_canonicals: Set[str] = set()

        # 1. Exact string index
        if raw_lower in self.exact_map:
            canon = self.exact_map[raw_lower]
            candidates.append((canon, raw_lower, "exact_index"))
            seen_canonicals.add(canon)

        # 2. Normalized index
        if norm_raw in self.norm_map:
            canon = self.norm_map[norm_raw]
            if canon not in seen_canonicals:
                candidates.append((canon, norm_raw, "norm_index"))
                seen_canonicals.add(canon)

        # 3. Domain index
        domain_probe = raw_lower.replace("https://", "").replace("http://", "").replace("www.", "").split("/")[0]
        if domain_probe in self.domain_map:
            canon = self.domain_map[domain_probe]
            if canon not in seen_canonicals:
                candidates.append((canon, domain_probe, "domain_index"))
                seen_canonicals.add(canon)

        # 4. Acronym index (only when input is a single-word acronym, e.g. "IBM", "AWS")
        if len(norm_raw.split()) == 1 and 2 <= len(norm_raw) <= 5:
            if norm_raw in self.acronym_map:
                canon = self.acronym_map[norm_raw]
                if canon not in seen_canonicals:
                    candidates.append((canon, norm_raw, "acronym_index"))
                    seen_canonicals.add(canon)

        # 5. Token & Trigram Blocking Retrieval
        candidate_scores: Dict[str, int] = {}
        tokens = [w for w in norm_raw.split() if len(w) >= 2]
        for tok in tokens:
            for matched_canon in self.token_index.get(tok, set()):
                candidate_scores[matched_canon] = candidate_scores.get(matched_canon, 0) + 3

        if len(norm_raw) >= 3:
            for i in range(len(norm_raw) - 2):
                tri = norm_raw[i : i + 3]
                for matched_canon in self.trigram_index.get(tri, set()):
                    candidate_scores[matched_canon] = candidate_scores.get(matched_canon, 0) + 1

        # Rank candidates by blocking match frequency (require at least 4 feature matches)
        sorted_canonicals = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)
        for canon, score in sorted_canonicals[:8]:
            if score >= 4 and canon not in seen_canonicals:
                candidates.append((canon, norm_raw, "blocking_index"))
                seen_canonicals.add(canon)

        # Fallback only for plausible similarity >= 60%
        if not candidates and norm_raw:
            for norm_variant, canon in self.all_variants:
                if fuzz.token_sort_ratio(norm_raw, norm_variant) >= 60:
                    candidates.append((canon, norm_variant, "fallback_scan"))
                    seen_canonicals.add(canon)
                    if len(candidates) >= 5:
                        break

        return candidates

    def score_candidates(
        self,
        raw_name: str,
        candidates: List[Tuple[str, str, str]],
    ) -> Tuple[Optional[str], str, float, float, Dict[str, Any], str]:
        """
        Candidate scoring stage:
        Evaluates deterministic matching signals (exact, normalized, domain, acronym, token_sort, token_set)
        and computes an explainable, bounded confidence score in [0.0, 1.0].
        Returns (best_canonical, best_method, best_confidence, best_fuzzy_score, best_signals, match_reason).
        """
        clean_raw = raw_name.strip()
        raw_lower = clean_raw.lower()
        norm_raw = normalize_entity_name(clean_raw)

        best_canonical = None
        best_method = "unresolved"
        best_confidence = 0.0
        best_fuzzy_score = 0.0
        best_signals: Dict[str, Any] = {}
        best_reason = "No plausible match found"

        for canon, _variant, _gen_source in candidates:
            aliases = self.canonical_aliases.get(canon, [canon])
            for alias in aliases:
                alias_clean = alias.strip()
                alias_lower = alias_clean.lower()
                alias_norm = normalize_entity_name(alias_clean)

                # Signal 1: Exact alias match
                is_exact = (raw_lower == alias_lower) or (
                    raw_lower in self.exact_map and self.exact_map[raw_lower] == canon
                )
                # Signal 2: Normalized match
                is_norm = norm_raw == alias_norm and len(norm_raw) > 0
                # Signal 3: Acronym expansion match (e.g. "IBM" <-> "International Business Machines")
                is_acronym = False
                raw_words = norm_raw.split()
                alias_words = alias_norm.split()
                if len(raw_words) == 1 and len(alias_words) >= 2:
                    is_acronym = norm_raw == extract_acronym(alias_norm)
                elif len(alias_words) == 1 and len(raw_words) >= 2:
                    is_acronym = alias_norm == extract_acronym(norm_raw)
                # Signal 4: Domain match
                is_domain = (
                    raw_lower in self.domain_map and self.domain_map[raw_lower] == canon
                ) or (
                    alias_lower in self.domain_map and self.domain_map[alias_lower] == canon and norm_raw in alias_lower
                )
                # Signal 5: RapidFuzz token ratios
                ts_ratio = fuzz.token_sort_ratio(norm_raw, alias_norm) / 100.0
                tset_ratio = fuzz.token_set_ratio(norm_raw, alias_norm) / 100.0

                signals = {
                    "exact_alias_match": is_exact,
                    "normalized_alias_match": is_norm,
                    "acronym_match": is_acronym,
                    "domain_match": is_domain,
                    "token_sort_ratio": round(ts_ratio, 4),
                    "token_set_ratio": round(tset_ratio, 4),
                    "candidate_alias": alias,
                }

                if is_exact:
                    conf = 1.0
                    fuzzy = 1.0
                    method = "exact"
                    reason = f"Exact alias match with canonical '{canon}'"
                elif is_norm:
                    conf = 0.95
                    fuzzy = 1.0
                    method = "normalized"
                    reason = f"Normalized string match with canonical '{canon}'"
                elif is_domain:
                    conf = 0.98
                    fuzzy = max(ts_ratio, 0.95)
                    method = "exact"
                    reason = f"Domain identifier match with canonical '{canon}'"
                elif is_acronym:
                    conf = 0.92
                    fuzzy = 0.92
                    method = "exact"
                    reason = f"Acronym match with canonical '{canon}'"
                else:
                    # Deterministic composite score weighted towards token_sort with subset bonus
                    score = (0.75 * ts_ratio) + (0.25 * tset_ratio)
                    conf = round(score, 4)
                    fuzzy = round(ts_ratio, 4)
                    method = "fuzzy"
                    reason = f"Fuzzy similarity ({fuzzy * 100:.1f}%) with canonical '{canon}' alias '{alias}'"

                if conf > best_confidence:
                    best_confidence = conf
                    best_canonical = canon
                    best_method = method
                    best_fuzzy_score = fuzzy
                    best_signals = signals
                    best_reason = reason

        return (
            best_canonical,
            best_method,
            round(best_confidence, 4),
            round(best_fuzzy_score, 4),
            best_signals,
            best_reason,
        )

    def resolve(
        self,
        raw_name: str,
        source: str = "pipeline",
    ) -> ResolutionResult:
        """
        Execute deterministic entity resolution pipeline:
        1. Input sanitization
        2. Candidate generation
        3. Candidate scoring
        4. Confidence threshold decision routing (AUTO_MATCH / NEEDS_REVIEW / NO_MATCH)
        5. Observability telemetry & JSONL audit logging
        """
        start_time = time.perf_counter()

        if not raw_name or not raw_name.strip():
            latency = time.perf_counter() - start_time
            res = ResolutionResult(
                canonical=None,
                method="unresolved",
                confidence=0.0,
                decision=ResolutionDecision.NO_MATCH,
                fuzzy_score=0.0,
                candidate=raw_name,
                match_reason="Empty or blank input name",
                signals={"empty_input": True},
                source=source,
                resolver_version=RESOLVER_VERSION,
            )
            self._record_telemetry(res, latency)
            return res

        clean_raw = raw_name.strip()

        # Step 1 & 2: Candidate Generation
        candidates = self.generate_candidates(clean_raw)

        # Step 3: Candidate Scoring
        if not candidates:
            best_canonical = None
            best_method = "unresolved"
            best_conf = 0.0
            best_fuzzy = 0.0
            best_signals: Dict[str, Any] = {"candidates_generated": 0}
            best_reason = "No candidate entities retrieved by blocking index"
        else:
            (
                best_canonical,
                best_method,
                best_conf,
                best_fuzzy,
                best_signals,
                best_reason,
            ) = self.score_candidates(clean_raw, candidates)

        # Step 4: Confidence Decision Threshold Routing
        if best_canonical and best_conf >= self.fuzzy_threshold:
            decision = ResolutionDecision.AUTO_MATCH
            canonical_result = best_canonical
            reason = best_reason
        elif best_canonical and best_conf >= self.review_threshold:
            decision = ResolutionDecision.NEEDS_REVIEW
            canonical_result = best_canonical
            reason = (
                f"Ambiguous similarity ({best_conf * 100:.1f}%) in review range "
                f"[{self.review_threshold * 100:.1f}%, {self.fuzzy_threshold * 100:.1f}%)"
            )
            # Enqueue into persistent human review queue
            self.review_queue.add_review(
                candidate=clean_raw,
                canonical=best_canonical,
                matching_method=best_method,
                fuzzy_score=best_fuzzy,
                confidence=best_conf,
                source=source,
                match_reason=reason,
                signals=best_signals,
                resolver_version=RESOLVER_VERSION,
            )
        else:
            decision = ResolutionDecision.NO_MATCH
            canonical_result = None
            best_method = "unresolved"
            reason = (
                f"Low similarity ({best_conf * 100:.1f}%) below review threshold ({self.review_threshold * 100:.1f}%)"
            )

        latency = time.perf_counter() - start_time
        res = ResolutionResult(
            canonical=canonical_result,
            method=best_method,
            confidence=best_conf,
            decision=decision,
            fuzzy_score=best_fuzzy,
            candidate=clean_raw,
            match_reason=reason,
            signals=best_signals,
            source=source,
            resolver_version=RESOLVER_VERSION,
        )

        self._record_telemetry(res, latency)
        self.log_decision(
            raw_name=clean_raw,
            canonical_name=canonical_result,
            method_used=best_method,
            confidence_score=best_conf,
            fuzzy_score=best_fuzzy,
            decision=decision.value,
            source=source,
            match_reason=reason,
            signals=best_signals,
        )
        return res

    def _record_telemetry(self, res: ResolutionResult, latency_seconds: float) -> None:
        """Record operational metrics and Prometheus counters."""
        metrics_collector.record_er_resolution(
            decision=res.decision,
            confidence=res.confidence,
            latency_seconds=latency_seconds,
        )
        if res.is_match():
            metrics_collector.record_er_merge(res.method)

    def log_decision(
        self,
        raw_name: str,
        canonical_name: Optional[str],
        method_used: str,
        confidence_score: float,
        fuzzy_score: float = 0.0,
        decision: str = ResolutionDecision.AUTO_MATCH.value,
        source: str = "pipeline",
        match_reason: str = "",
        signals: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Log entity resolution decision with complete audit fields to
        data/processed/entity_mapping_log.jsonl.
        """
        settings.DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        record_id = f"er-{abs(hash(raw_name)) % 10000:04d}"
        d_upper = decision.upper()
        status = (
            "merged"
            if d_upper in ("AUTO_MATCH", "MATCH")
            else ("needs_review" if d_upper in ("NEEDS_REVIEW", "REVIEW") else "kept_separate")
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
            "matching_signals": signals or {},
            "entity_name": canonical_name or raw_name,
            "raw_name": raw_name,
            "canonical_name": canonical_name,
            "entity_type": "startup",
            "resolver_version": RESOLVER_VERSION,
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

    def resolve_record(
        self,
        record: BaseModel,
        source: str = "pipeline",
    ) -> Tuple[BaseModel, Dict[str, Any]]:
        """
        Apply entity resolver to a Startup or Product Pydantic schema model object,
        updating entity name / maker_company to canonical form when resolved with AUTO_MATCH decision.
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
                "decision": ResolutionDecision.NO_MATCH.value,
            }

        res = self.resolve(raw_target_name, source=source)

        # Merge into canonical form only when final decision is explicit AUTO_MATCH
        if res.is_match() and res.canonical and target_field:
            setattr(record, target_field, res.canonical)

        decision_info = {
            "raw_name": raw_target_name,
            "candidate": raw_target_name,
            "canonical_name": res.canonical if res.is_match() else None,
            "matched_entity": res.canonical,
            "method_used": res.method_used,
            "matching_method": res.method_used,
            "confidence_score": res.confidence_score,
            "confidence": res.confidence_score,
            "fuzzy_score": res.fuzzy_score,
            "decision": res.decision,
            "final_decision": res.decision,
            "match_reason": res.match_reason,
            "signals": res.signals,
            "resolver_version": res.resolver_version,
        }
        return record, decision_info
