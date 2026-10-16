"""
================================================================================
PHASE 6: CITATION-AWARE RAG TEST SUITE
================================================================================
Comprehensive test suite verifying:
1. Provenance survives retrieval and generation.
2. Citation references map to actual retrieved chunks.
3. Multiple citations resolve correctly.
4. Invalid citation references are rejected.
5. Fabricated source metadata is not accepted.
6. Unsupported claims trigger appropriate handling.
7. Insufficient evidence leads to abstention.
8. Missing page metadata is handled correctly.
9. Duplicate chunks do not corrupt citations.
10. Semantic validation distinguishes relevant from irrelevant evidence.
11. Validator failures are handled safely.
12. Conflicting evidence is represented appropriately.
13. Structured response schema validation.
14. Metrics update correctly.
15. Citation evaluation metrics use correct denominators.
16. API endpoints (/api/rag/query, /api/rag/validate, /rag/query, /rag/validate).
17. Integration with evaluation framework and benchmark dataset.
================================================================================
"""

import pytest
from evaluation.citation_evaluator import run_citation_benchmark
from evaluation.metrics import (
    calculate_abstention_accuracy,
    calculate_citation_faithfulness,
    calculate_citation_reference_validity,
    calculate_claim_citation_coverage,
    calculate_unsupported_claim_rate,
)
from fastapi.testclient import TestClient
from src.dashboard.main import app
from src.observability.metrics import metrics_collector
from src.retrieval.citation_validator import (
    CitationValidator,
)
from src.retrieval.models import (
    Citation,
    RAGQueryRequest,
    RAGQueryResponse,
    SourceChunkProvenance,
    ValidationStatus,
)
from src.retrieval.rag_service import RAGService


@pytest.fixture
def test_client():
    """FastAPI test client."""
    return TestClient(app)


@pytest.fixture
def sample_sources():
    """Sample provenance-tracked sources fixture."""
    return [
        SourceChunkProvenance(
            index=1,
            source_id="node-startup-anthropic",
            document_id="doc-startup-anthropic",
            chunk_id="doc-startup-anthropic:chunk-0",
            title="Anthropic, Inc.",
            source_url="https://techcrunch.com/anthropic",
            page_number=1,
            section="Company Profile",
            text="Anthropic is an AI safety research company based in San Francisco, CA that develops Claude foundation models with $7.3B in funding.",
            retrieval_method="VECTOR",
            retrieval_score=0.92,
            candidate_type="startup",
            metadata={"location": "San Francisco, CA", "total_funding": "$7.3B"},
        ),
        SourceChunkProvenance(
            index=2,
            source_id="node-product-claude",
            document_id="doc-product-claude",
            chunk_id="doc-product-claude:chunk-0",
            title="Claude 3.5 Sonnet",
            source_url="https://producthunt.com/products/claude-3-5",
            page_number=None,  # Missing page metadata explicitly None
            section="Release Notes",
            text="Claude 3.5 Sonnet is a frontier reasoning model developed by Anthropic with benchmark-leading coding abilities.",
            retrieval_method="GRAPH",
            retrieval_score=0.88,
            candidate_type="product",
            metadata={"maker_company": "Anthropic"},
        ),
    ]


def test_provenance_survives_retrieval_and_generation(sample_sources):
    """Test 1: Provenance metadata survives retrieval and serialization into response."""
    service = RAGService()
    rag_req = RAGQueryRequest(query="Where is Anthropic based and what is its funding?")
    response = service.answer_query(rag_req)

    assert isinstance(response, RAGQueryResponse)
    assert len(response.sources) > 0
    first_source = response.sources[0]
    assert first_source.document_id.startswith("doc-")
    assert first_source.chunk_id.startswith("doc-")
    assert first_source.title is not None
    assert first_source.retrieval_score >= 0.0
    assert first_source.retrieval_method in ("VECTOR", "GRAPH", "BOTH")


def test_citation_references_map_to_actual_retrieved_chunks(sample_sources):
    """Test 2: Each generated citation maps to a real retrieved chunk in sources."""
    validator = CitationValidator()
    answer = "Anthropic is based in San Francisco, CA with $7.3B in funding [1]."
    result = validator.validate(answer=answer, sources=sample_sources)

    assert result.status == ValidationStatus.VALID
    assert result.reference_valid is True
    assert result.semantic_valid is True
    assert result.valid_citations == 1
    assert result.invalid_citations == 0


def test_multiple_citations_resolve_correctly(sample_sources):
    """Test 3: Multiple citations [1][2] and [1, 2] resolve to their respective sources."""
    validator = CitationValidator()

    # Bracket concatenation format [1][2]
    ans1 = "Anthropic is an AI safety company [1] that develops Claude 3.5 Sonnet [2]."
    res1 = validator.validate(answer=ans1, sources=sample_sources)
    assert res1.status == ValidationStatus.VALID
    assert res1.valid_citations == 2
    assert res1.invalid_citations == 0

    # Comma-separated format [1, 2]
    ans2 = "Anthropic develops frontier foundation models such as Claude 3.5 Sonnet [1, 2]."
    res2 = validator.validate(answer=ans2, sources=sample_sources)
    assert res2.status == ValidationStatus.VALID
    assert res2.valid_citations == 2


def test_invalid_citation_references_are_rejected(sample_sources):
    """Test 4: Citation references to nonexistent sources (e.g. [99]) are flagged and rejected."""
    validator = CitationValidator()
    answer = "Anthropic is based in San Francisco [99]."
    result = validator.validate(answer=answer, sources=sample_sources)

    assert result.status == ValidationStatus.INVALID_REFERENCE
    assert result.reference_valid is False
    assert result.invalid_citations == 1
    assert any("[99]" in w for w in result.warnings)


def test_fabricated_source_metadata_is_not_accepted(sample_sources):
    """Test 5: Fabricated source ID or document ID in citations is rejected."""
    validator = CitationValidator()
    fake_citation = Citation(
        citation_id=1,
        source_id="node-startup-anthropic",
        document_id="doc-fabricated-fake-document-1234",
        claim_text="Anthropic is based in San Francisco [1].",
        reference_valid=True,
    )
    result = validator.validate(
        answer="Anthropic is based in San Francisco [1].",
        sources=sample_sources,
        citations=[fake_citation],
    )

    assert result.status == ValidationStatus.INVALID_REFERENCE
    assert result.reference_valid is False
    assert any("Fabricated document ID" in w for w in result.warnings)


def test_unsupported_claims_trigger_appropriate_handling(sample_sources):
    """Test 6: Valid citation ID attached to irrelevant text triggers UNSUPPORTED_CLAIM."""
    validator = CitationValidator()
    # Claim asserts robotic space travel which Anthropic source does not mention
    answer = "Anthropic manufactures humanoid robots for lunar exploration missions [1]."
    result = validator.validate(answer=answer, sources=sample_sources)

    assert result.status == ValidationStatus.UNSUPPORTED_CLAIM
    assert result.reference_valid is True  # Index 1 exists
    assert result.semantic_valid is False  # Evidence does not support claim
    assert result.unsupported_claims >= 1


def test_insufficient_evidence_leads_to_abstention():
    """Test 7: Empty or unrelated evidence triggers abstention policy."""
    service = RAGService()
    # Query with zero related entities
    req = RAGQueryRequest(query="What is the quantum teleportation efficiency of warp coils?")
    res = service.answer_query(req)

    assert res.abstained is True
    assert "insufficient" in res.answer.lower()
    assert res.validation_status in (ValidationStatus.INSUFFICIENT_EVIDENCE, ValidationStatus.VALID)
    assert len(res.citations) == 0


def test_missing_page_metadata_handled_correctly(sample_sources):
    """Test 8: Sources with page_number=None are supported without error or fabrication."""
    source_2 = sample_sources[1]
    assert source_2.page_number is None

    validator = CitationValidator()
    answer = "Claude 3.5 Sonnet has state of the art coding abilities [2]."
    result = validator.validate(answer=answer, sources=sample_sources)

    assert result.status == ValidationStatus.VALID
    assert result.reference_valid is True
    assert result.semantic_valid is True


def test_duplicate_chunks_do_not_corrupt_citations():
    """Test 9: Duplicate source chunks receive unique indices and resolve cleanly."""
    duplicate_sources = [
        SourceChunkProvenance(
            index=1,
            source_id="node-startup-anthropic",
            document_id="doc-startup-anthropic",
            chunk_id="doc-startup-anthropic:chunk-0",
            title="Anthropic",
            text="Anthropic raised $7.3B in Series E funding.",
            retrieval_method="VECTOR",
            retrieval_score=0.95,
            metadata={"total_funding": "$7.3B"},
        ),
        SourceChunkProvenance(
            index=2,
            source_id="node-startup-anthropic",
            document_id="doc-startup-anthropic",
            chunk_id="doc-startup-anthropic:chunk-1",
            title="Anthropic",
            text="Anthropic raised $7.3B in Series E funding.",
            retrieval_method="GRAPH",
            retrieval_score=0.95,
            metadata={"total_funding": "$7.3B"},
        ),
    ]

    validator = CitationValidator()
    answer = "Anthropic raised $7.3B in Series E funding [1][2]."
    result = validator.validate(answer=answer, sources=duplicate_sources)

    assert result.status == ValidationStatus.VALID
    assert result.valid_citations == 2
    assert result.invalid_citations == 0


def test_semantic_validation_distinguishes_relevant_from_irrelevant(sample_sources):
    """Test 10: Semantic validation verifies claim overlap and rejects irrelevant sources."""
    validator = CitationValidator()

    # Relevant claim matching source 2
    res_rel = validator.validate(
        answer="Claude 3.5 Sonnet is developed by Anthropic [2].",
        sources=sample_sources,
    )
    assert res_rel.status == ValidationStatus.VALID

    # Irrelevant claim citing source 2
    res_irrel = validator.validate(
        answer="Autonomous tractors were deployed for corn harvesting [2].",
        sources=sample_sources,
    )
    assert res_irrel.status == ValidationStatus.UNSUPPORTED_CLAIM


def test_validator_failures_handled_safely(sample_sources):
    """Test 11: Validator failure or exception is handled safely without crashing."""
    class FailingProvider:
        def extract_raw(self, prompt: str):
            raise TimeoutError("Simulated LLM validation timeout")

    validator = CitationValidator(llm_provider=FailingProvider())
    # Should safely fall back to heuristic validation without crashing
    res = validator.validate(
        answer="Anthropic is located in San Francisco [1].",
        sources=sample_sources,
        validator_type="llm",
    )
    assert res.status == ValidationStatus.VALID
    assert res.reference_valid is True


def test_conflicting_evidence_represented_appropriately():
    """Test 12: Conflicting facts across sources for the same entity trigger conflict warnings."""
    conflicting_sources = [
        SourceChunkProvenance(
            index=1,
            source_id="node-startup-anthropic-1",
            document_id="doc-anthropic-1",
            chunk_id="doc-anthropic-1:chunk-0",
            title="Anthropic",
            text="Anthropic was founded in 2021 in San Francisco.",
            retrieval_method="VECTOR",
            retrieval_score=0.90,
            metadata={},
        ),
        SourceChunkProvenance(
            index=2,
            source_id="node-startup-anthropic-2",
            document_id="doc-anthropic-2",
            chunk_id="doc-anthropic-2:chunk-0",
            title="Anthropic",
            text="Anthropic incorporated in 2023 as a Delaware corporation.",
            retrieval_method="GRAPH",
            retrieval_score=0.85,
            metadata={},
        ),
    ]

    validator = CitationValidator()
    answer = "Anthropic was founded in 2021 [1] and incorporated in 2023 [2]."
    res = validator.validate(answer=answer, sources=conflicting_sources)

    assert "conflicting_evidence" in res.details
    assert any("Conflicting date/year" in c for c in res.details["conflicting_evidence"])


def test_structured_response_schema_validation(sample_sources):
    """Test 13: Structured response models serialize and validate cleanly."""
    resp = RAGQueryResponse(
        query="Anthropic models",
        answer="Anthropic develops Claude 3.5 Sonnet [1].",
        citations=[
            Citation(
                citation_id=1,
                source_id="node-startup-anthropic",
                chunk_id="doc-startup-anthropic:chunk-0",
                document_id="doc-startup-anthropic",
                claim_text="Anthropic develops Claude 3.5 Sonnet",
                reference_valid=True,
                claim_supported=True,
            )
        ],
        sources=sample_sources,
        abstained=False,
        validation_status=ValidationStatus.VALID,
    )
    dumped = resp.model_dump(mode="json")
    assert dumped["answer"].startswith("Anthropic develops")
    assert len(dumped["citations"]) == 1
    assert dumped["validation_status"] == "VALID"


def test_metrics_update_correctly():
    """Test 14: System metrics record RAG queries and citations correctly."""
    metrics_collector.reset_rag_metrics()
    assert metrics_collector.rag_queries_total == 0

    metrics_collector.record_rag_query(
        latency_seconds=0.045,
        citations_count=3,
        invalid_citations_count=1,
        unsupported_claims_count=0,
        is_abstained=False,
        validation_status="VALID",
    )
    assert metrics_collector.rag_queries_total == 1
    assert metrics_collector.rag_citation_validations_total == 3
    assert metrics_collector.rag_invalid_citations_total == 1

    metrics = metrics_collector.get_rag_metrics()
    assert metrics["rag_queries_total"] == 1
    assert metrics["rag_query_avg_latency_ms"] > 0

    prom_text = metrics_collector.generate_prometheus_format()
    assert "graphone_rag_queries_total 1" in prom_text
    assert "graphone_rag_invalid_citations_total 1" in prom_text


def test_citation_evaluation_metrics_denominators():
    """Test 15: Evaluation metrics handle exact denominators and 0 edge cases."""
    # Reference validity: valid / total
    assert calculate_citation_reference_validity(4, 5) == 0.8
    assert calculate_citation_reference_validity(0, 0) == 1.0  # Abstention case

    # Claim coverage: cited / total
    assert calculate_claim_citation_coverage(3, 4) == 0.75
    assert calculate_claim_citation_coverage(0, 0) == 1.0

    # Faithfulness: supported / valid
    assert calculate_citation_faithfulness(4, 5) == 0.8
    assert calculate_citation_faithfulness(0, 0) == 1.0

    # Unsupported claim rate: unsupported / total
    assert calculate_unsupported_claim_rate(1, 4) == 0.25
    assert calculate_unsupported_claim_rate(0, 0) == 0.0

    # Abstention accuracy: correct / total
    assert calculate_abstention_accuracy(9, 10) == 0.9


def test_api_rag_query_endpoint(test_client):
    """Test 16: POST /api/rag/query and /rag/query return valid structured answers."""
    payload = {
        "query": "Anthropic AI funding and Claude models",
        "vector_limit": 5,
        "graph_depth": 1,
        "limit": 3,
        "semantic_validation": True,
    }
    for endpoint in ("/api/rag/query", "/rag/query"):
        res = test_client.post(endpoint, json=payload)
        assert res.status_code == 200, res.text
        data = res.json()
        assert "answer" in data
        assert "citations" in data
        assert "sources" in data
        assert "validation_status" in data


def test_api_rag_validate_endpoint(test_client, sample_sources):
    """Test 17: POST /api/rag/validate checks citations independently."""
    payload = {
        "answer": "Anthropic is based in San Francisco with $7.3B in funding [1].",
        "sources": [s.model_dump(mode="json") for s in sample_sources],
        "semantic_validation": True,
    }
    for endpoint in ("/api/rag/validate", "/rag/validate"):
        res = test_client.post(endpoint, json=payload)
        assert res.status_code == 200, res.text
        data = res.json()
        assert data["validation_status"] == "VALID"
        assert data["validation_result"]["reference_valid"] is True


def test_citation_benchmark_runner():
    """Test 18: run_citation_benchmark executes all 12 dataset scenarios."""
    report = run_citation_benchmark()
    assert report["total_test_cases"] == 12
    assert "citation_reference_validity" in report["metrics"]
    assert "claim_level_citation_coverage" in report["metrics"]
    assert "citation_faithfulness" in report["metrics"]
    assert "unsupported_claim_rate" in report["metrics"]
    assert "abstention_correctness" in report["metrics"]
    assert report["metrics"]["abstention_correctness"] >= 0.80
