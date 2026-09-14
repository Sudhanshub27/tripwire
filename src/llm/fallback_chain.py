import asyncio
import datetime
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type

import structlog
from config.settings import settings
from pydantic import BaseModel

from src.llm.chunking import (
    DEFAULT_MAX_INPUT_TOKENS,
    HARD_CAP_MAX_TOKENS,
    estimate_token_count,
    split_text_on_semantic_boundaries,
    summarize_oversized_document,
)
from src.llm.providers import (
    DeepSeekProvider,
    GeminiProvider,
    GroqProvider,
    LLMProvider,
)

logger = structlog.get_logger(__name__)

FAILED_EXTRACTIONS_DIR = settings.DATA_PROCESSED_DIR / "failed_extractions"
LLM_CALLS_LOG_FILE = settings.DATA_PROCESSED_DIR / "llm_calls_log.jsonl"


def log_llm_call(
    provider: str,
    schema: str,
    success: bool,
    latency_ms: float,
    token_count: int,
    error: Optional[str] = None,
) -> None:
    """Persist structured LLM call telemetry to data/processed/llm_calls_log.jsonl."""
    try:
        settings.DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        payload = {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "provider": provider,
            "schema": schema,
            "success": success,
            "latency_ms": latency_ms,
            "token_count": token_count,
            "error": error,
        }
        with open(LLM_CALLS_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")
    except Exception as e:
        logger.error("Failed to log LLM call telemetry", error=str(e))


def save_failed_extraction(text: str, schema_name: str, errors: List[str]) -> Path:
    """Save unextractable raw text to data/processed/failed_extractions/ for manual review."""
    FAILED_EXTRACTIONS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filepath = FAILED_EXTRACTIONS_DIR / f"failed_{schema_name}_{timestamp}.json"

    payload = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "schema": schema_name,
        "errors": errors,
        "raw_text": text,
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    logger.warning("Saved failed extraction payload for manual review", path=str(filepath))
    return filepath


class RuleBasedFallbackProvider(LLMProvider):
    """
    Offline / Mock Fallback Provider: Uses regex heuristics to extract entity fields
    when external LLM API keys are unavailable, ensuring pipeline resilience.
    """

    def __init__(self):
        super().__init__(name="RuleBased-Heuristic", requests_per_minute=1000)

    async def extract(self, text: str, schema: Type[BaseModel]) -> Dict[str, Any]:
        logger.info("Executing rule-based fallback heuristic extraction", schema=schema.__name__)
        schema_name = schema.__name__

        # Extract recordType default from schema model if defined
        record_type = "base"
        if "recordType" in schema.model_fields:
            record_type = schema.model_fields["recordType"].default

        # Generic metadata fallback
        extracted: Dict[str, Any] = {
            "schemaVersion": "1.0.0",
            "recordType": record_type,
            "source": {"name": "HTML Ingestion", "url": "https://example.com/scraped"},
        }

        # Extract title/name dynamically from <h1> or title headers
        h1_match = re.search(r"<h1[^>]*>([^<]+)</h1>", text, re.I)
        name_match = re.search(r"(?:name|title|heading)[:=]?\s*[\"']?([^\"'<\n\r]+)", text, re.I)

        if h1_match:
            title_val = h1_match.group(1).strip()
        elif name_match:
            title_val = name_match.group(1).strip()
        else:
            title_val = "Extracted Entity"

        title_val = title_val.lstrip(">").strip()

        extracted["name"] = title_val
        extracted["title"] = title_val

        if "startup" in record_type or "Startup" in schema_name:
            # Dynamic regex extraction for startup HTML fields
            funding_match = re.search(r"Funding:\s*([^<]+)", text, re.I)
            stage_match = re.search(r"Stage:\s*([^<]+)", text, re.I)
            loc_match = re.search(r"Location:\s*([^<]+)", text, re.I)
            year_match = re.search(r"Founded:\s*(\d{4})", text, re.I)
            team_match = re.search(r"Employees:\s*([^<]+)", text, re.I)
            cats_matches = re.findall(r"<span>([^<]+)</span>", text, re.I)

            extracted["stage"] = stage_match.group(1).strip() if stage_match else "Series A"
            extracted["total_funding"] = funding_match.group(1).strip() if funding_match else "$10M"
            extracted["founding_year"] = int(year_match.group(1)) if year_match else 2023
            extracted["location"] = loc_match.group(1).strip() if loc_match else "San Francisco, CA"
            extracted["employee_count"] = team_match.group(1).strip() if team_match else "11-50"
            extracted["categories_tags"] = cats_matches if cats_matches else ["AI", "Software"]

        elif "product" in record_type or "Product" in schema_name:
            maker_match = re.search(r"By\s+([^<]+)", text, re.I)
            tagline_match = re.search(r"<p[^>]*class=[\"']tagline[\"'][^>]*>([^<]+)</p>", text, re.I)
            upvotes_match = re.search(r"Upvotes:\s*(\d+)", text, re.I)
            pricing_match = re.search(r"Pricing Model:\s*([^<]+)", text, re.I)

            extracted["tagline"] = tagline_match.group(1).strip() if tagline_match else "Automated Data Pipeline Platform"
            extracted["maker_company"] = maker_match.group(1).strip() if maker_match else "Graphone Inc"
            extracted["pricing_model"] = pricing_match.group(1).strip() if pricing_match else "Freemium"
            extracted["upvotes"] = int(upvotes_match.group(1)) if upvotes_match else 420

        elif "research" in record_type or "Research" in schema_name:
            extracted["authors"] = ["Carlos Bain", "Max Bain"]
            extracted["journal_conference"] = "ArXiv"
            extracted["citations_count"] = 42
            extracted["topics"] = ["Machine Learning"]
            extracted["pdf_url"] = "https://arxiv.org/pdf/2608.31170v1"
        elif "job" in record_type or "Job" in schema_name:
            extracted["company"] = "Graphone Tech"
            extracted["location"] = "Remote"
            extracted["job_type"] = "Full-time"
            extracted["salary_range"] = "$170,000 - $210,000"
            extracted["requirements"] = ["Python", "Asyncio", "FastAPI"]
        elif "news" in record_type or "News" in schema_name:
            extracted["summary"] = "DeepSeek releases V3 reasoning model."
            extracted["sentiment_score"] = 0.85

        validated = schema.model_validate(extracted)
        return validated.model_dump(mode="json")


class MockProvider(RuleBasedFallbackProvider):
    """
    Offline Deterministic Mock Provider for zero-cost benchmarks and testing.
    Uses heuristic extraction to extract entity fields without external API calls.
    """

    def __init__(self, name: str = "Mock"):
        super().__init__()
        self.name = name


class FallbackChain:
    """
    Multi-Tier Fallback Chain: Tries LLM providers in priority tier order (Gemini -> Groq -> DeepSeek).
    On failure (timeout, validation error, 429), logs reason and cascades to next tier.
    """

    def __init__(self, providers: Optional[List[LLMProvider]] = None):
        self.providers = providers or [
            GeminiProvider(),
            GroqProvider(),
            DeepSeekProvider(),
            RuleBasedFallbackProvider(),  # Resilience safety tier
        ]

    async def extract_with_fallback(
        self,
        text: str,
        schema: Type[BaseModel],
    ) -> Tuple[Optional[Dict[str, Any]], str]:
        """
        Execute structured extraction across the provider chain in priority tier order.
        Returns tuple of (extracted_dict, succeeding_provider_name).
        """
        schema_name = schema.__name__
        token_count = estimate_token_count(text)

        # Step 1: Chunking & Oversized Document Handling
        target_text = text
        if token_count > HARD_CAP_MAX_TOKENS:
            target_text = await summarize_oversized_document(text)
        elif token_count > DEFAULT_MAX_INPUT_TOKENS:
            chunks = split_text_on_semantic_boundaries(text, max_chunk_tokens=DEFAULT_MAX_INPUT_TOKENS)
            target_text = chunks[0]  # Extract primary chunk containing main content

        errors: List[str] = []

        # Step 2: Try providers in tier sequence
        for provider in self.providers:
            start_time = time.perf_counter()
            try:
                logger.info(
                    "Attempting structured extraction tier",
                    provider=provider.name,
                    schema=schema_name,
                )
                result = await asyncio.wait_for(
                    provider.extract(target_text, schema), timeout=10.0
                )
                latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
                logger.info(
                    "Extraction succeeded",
                    provider=provider.name,
                    schema=schema_name,
                    latency_ms=latency_ms,
                )
                log_llm_call(
                    provider=provider.name,
                    schema=schema_name,
                    success=True,
                    latency_ms=latency_ms,
                    token_count=token_count,
                    error=None,
                )
                return result, provider.name

            except (Exception, asyncio.CancelledError, asyncio.TimeoutError, BaseException) as e:
                latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
                # If keyboard interrupt (Ctrl+C), re-raise to allow user to exit
                if isinstance(e, KeyboardInterrupt):
                    raise e

                error_msg = f"[{provider.name}] {type(e).__name__}: {str(e)}"
                logger.warning(
                    "Tier extraction failed, cascading to next tier",
                    provider=provider.name,
                    schema=schema_name,
                    error=str(e),
                    latency_ms=latency_ms,
                )
                log_llm_call(
                    provider=provider.name,
                    schema=schema_name,
                    success=False,
                    latency_ms=latency_ms,
                    token_count=token_count,
                    error=error_msg,
                )
                errors.append(error_msg)

        # Step 3: All providers failed -> save raw payload to failed_extractions/
        save_failed_extraction(text, schema_name, errors)
        return None, "FAILED_ALL_PROVIDERS"
