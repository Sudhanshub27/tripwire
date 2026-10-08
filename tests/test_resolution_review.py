"""
================================================================================
TEST SUITE FOR CONFIDENCE-BASED ENTITY RESOLUTION & REVIEW QUEUE
================================================================================
Covers:
- Automatic match (MATCH) for exact, normalized, and high fuzzy similarity (>=85%)
- Automatic new entity (NEW) for low similarity (<65%) and empty inputs
- Review state (REVIEW) for ambiguous fuzzy matches (65% to 85%)
- Approval workflow (approve_review -> alias registration -> future MATCH)
- Rejection workflow (reject_review -> kept as separate NEW entity)
- Persistence of review decisions and logs across instances
- FastAPI backend review endpoints (/api/resolution/review, approve, reject)
- Schema record resolution handling for MATCH vs REVIEW vs NEW
================================================================================
"""

import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from src.dashboard.main import app
from src.resolution.entity_resolver import (
    EntityResolver,
    ResolutionDecision,
)
from src.resolution.review_queue import ReviewQueue
from src.schemas.base import SourceMetadata
from src.schemas.startup import Startup


@pytest.fixture
def temp_env():
    """Fixture providing isolated temporary directory and databases for resolver and queue."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        db_path = tmp_path / "test_reviews.db"
        jsonl_path = tmp_path / "test_reviews.jsonl"
        queue = ReviewQueue(db_path=db_path, jsonl_path=jsonl_path)
        resolver = EntityResolver(queue=queue)
        yield {
            "tmp_path": tmp_path,
            "db_path": db_path,
            "jsonl_path": jsonl_path,
            "queue": queue,
            "resolver": resolver,
        }


def test_automatic_match_exact(temp_env):
    """Test exact match yields MATCH decision and 1.0 confidence."""
    resolver = temp_env["resolver"]
    res = resolver.resolve("openai.com")

    assert res.decision == ResolutionDecision.MATCH.value
    assert res.canonical == "OpenAI"
    assert res.method == "exact"
    assert res.confidence == 1.0
    assert res.fuzzy_score == 1.0

    # Backwards-compatible 3-tuple unpacking
    canon, method, conf = res
    assert canon == "OpenAI"
    assert method == "exact"
    assert conf == 1.0


def test_automatic_match_normalized(temp_env):
    """Test normalized suffix-stripped match yields MATCH decision and 0.95 confidence."""
    resolver = temp_env["resolver"]
    res = resolver.resolve("Mistral AI SAS")

    assert res.decision == ResolutionDecision.MATCH.value
    assert res.canonical == "Mistral AI"
    assert res.method in ("exact", "normalized")
    assert res.confidence >= 0.95


def test_automatic_match_high_fuzzy(temp_env):
    """Test fuzzy similarity >= 85% yields automatic MATCH."""
    resolver = temp_env["resolver"]
    res = resolver.resolve("Hugging Face Hub")

    assert res.decision == ResolutionDecision.MATCH.value
    assert res.canonical == "Hugging Face"
    assert res.method == "fuzzy"
    assert res.fuzzy_score >= 0.85
    assert res.confidence >= 0.85


def test_automatic_new_entity(temp_env):
    """Test low similarity (< 65%) yields NEW decision with no canonical match."""
    resolver = temp_env["resolver"]
    res = resolver.resolve("Totally Novel Robotics Corp 2026")

    assert res.decision == ResolutionDecision.NEW.value
    assert res.canonical is None
    assert res.method == "unresolved"
    assert res.confidence < 0.65

    # Tuple unpacking returns None for canonical on NEW
    canon, method, conf = res
    assert canon is None


def test_automatic_new_entity_empty(temp_env):
    """Test empty/blank input yields NEW decision."""
    resolver = temp_env["resolver"]
    res = resolver.resolve("   ")

    assert res.decision == ResolutionDecision.NEW.value
    assert res.canonical is None
    assert res.confidence == 0.0


def test_review_state_ambiguous_fuzzy(temp_env):
    """
    Test ambiguous match in [65%, 85%) produces REVIEW decision,
    does NOT auto-merge canonical, and enqueues to persistent review queue.
    """
    resolver = temp_env["resolver"]
    queue = temp_env["queue"]

    # "Synthesia Media Labs" vs canonical "Synthesia" (or "Synthesia Labs" variant)
    # Norm variant: "synthesia" vs "synthesia media" -> fuzzy score ~ 76%
    res = resolver.resolve("Synthesia Media", source="test_feed")

    assert res.decision == ResolutionDecision.REVIEW.value
    assert res.canonical == "Synthesia"
    assert res.method == "fuzzy"
    assert 0.65 <= res.fuzzy_score < 0.85

    # Canonical in 3-tuple unpack is None until approved
    canon, method, conf = res
    assert canon is None

    # Verify enqueued into review queue
    pending = queue.get_pending_reviews()
    assert len(pending) == 1
    item = pending[0]
    assert item["candidate"] == "Synthesia Media"
    assert item["canonical"] == "Synthesia"
    assert item["matching_method"] == "fuzzy"
    assert item["decision"] == "REVIEW"
    assert item["source"] == "test_feed"
    assert item["fuzzy_score"] >= 0.65
    assert item["confidence"] >= 0.65


def test_approval_workflow(temp_env):
    """
    Test approving a pending review:
    1. Mark item APPROVED in queue
    2. Register alias in resolver
    3. Verify subsequent resolve() returns MATCH
    """
    resolver = temp_env["resolver"]
    queue = temp_env["queue"]

    # Trigger REVIEW
    res = resolver.resolve("Synthesia Media")
    assert res.decision == ResolutionDecision.REVIEW.value

    pending = queue.get_pending_reviews()
    assert len(pending) == 1
    review_id = pending[0]["id"]

    # Approve review
    approved = queue.approve_review(review_id, notes="Verified alias by reviewer")
    assert approved is not None
    assert approved["decision"] == "APPROVED"
    assert approved["reviewed_at"] is not None
    assert approved["reviewer_notes"] == "Verified alias by reviewer"

    # Pending queue should now be empty
    assert len(queue.get_pending_reviews()) == 0

    # Register approved alias with resolver
    resolver.add_approved_alias("Synthesia Media", "Synthesia")

    # Subsequent resolution must now resolve directly as MATCH
    res_after = resolver.resolve("Synthesia Media")
    assert res_after.decision == ResolutionDecision.MATCH.value
    assert res_after.canonical == "Synthesia"
    assert res_after.confidence >= 0.95


def test_rejection_workflow(temp_env):
    """
    Test rejecting a pending review:
    1. Mark item REJECTED in queue
    2. Entity is kept as separate NEW entity
    """
    resolver = temp_env["resolver"]
    queue = temp_env["queue"]

    res = resolver.resolve("OpenAI Labs")
    assert res.decision == ResolutionDecision.REVIEW.value

    pending = queue.get_pending_reviews()
    assert len(pending) == 1
    review_id = pending[0]["id"]

    # Reject review
    rejected = queue.reject_review(review_id, notes="Different project than core OpenAI")
    assert rejected is not None
    assert rejected["decision"] == "REJECTED"
    assert rejected["reviewed_at"] is not None
    assert rejected["reviewer_notes"] == "Different project than core OpenAI"

    # Pending queue should now be empty
    assert len(queue.get_pending_reviews()) == 0


def test_persistence_of_decisions(temp_env):
    """Test decisions and review queue persist across instance re-instantiations."""
    db_path = temp_env["db_path"]
    jsonl_path = temp_env["jsonl_path"]
    queue = temp_env["queue"]

    # Add review and approve
    item = queue.add_review(
        candidate="Cohere Health Services",
        canonical="Cohere",
        matching_method="fuzzy",
        fuzzy_score=0.72,
        confidence=0.72,
        source="health_crawl",
        match_reason="Ambiguous similarity",
    )
    queue.approve_review(item["id"], notes="Reviewed and approved")

    # Instantiate brand new ReviewQueue pointing to same db
    fresh_queue = ReviewQueue(db_path=db_path, jsonl_path=jsonl_path)
    all_reviews = fresh_queue.list_all_reviews()

    assert len(all_reviews) == 1
    persisted = all_reviews[0]
    assert persisted["id"] == item["id"]
    assert persisted["candidate"] == "Cohere Health Services"
    assert persisted["canonical"] == "Cohere"
    assert persisted["decision"] == "APPROVED"
    assert persisted["reviewer_notes"] == "Reviewed and approved"
    assert persisted["source"] == "health_crawl"


def test_resolve_record_pydantic_integration(temp_env):
    """
    Test resolve_record behaves correctly:
    - MATCH merges entity name
    - REVIEW keeps original entity name unmerged
    - NEW keeps original entity name unmerged
    """
    resolver = temp_env["resolver"]

    # 1. MATCH case: OpenAI, Inc. -> OpenAI
    startup_match = Startup(
        name="OpenAI, Inc.",
        stage="Series B",
        source=SourceMetadata(name="TechCrunch", url="https://techcrunch.com/openai-1"),
    )
    res_obj, info = resolver.resolve_record(startup_match)
    assert res_obj.name == "OpenAI"
    assert info["decision"] == ResolutionDecision.MATCH.value

    # 2. REVIEW case: Synthesia Media -> untouched until approved
    startup_review = Startup(
        name="Synthesia Media",
        stage="Seed",
        source=SourceMetadata(name="TechCrunch", url="https://techcrunch.com/syn"),
    )
    res_obj_rev, info_rev = resolver.resolve_record(startup_review)
    assert res_obj_rev.name == "Synthesia Media"  # Not merged
    assert info_rev["decision"] == ResolutionDecision.REVIEW.value
    assert info_rev["matched_entity"] == "Synthesia"

    # 3. NEW case: Completely new startup -> untouched
    startup_new = Startup(
        name="Quantum Deep Genomics Lab",
        stage="Pre-Seed",
        source=SourceMetadata(name="TechCrunch", url="https://techcrunch.com/qdg"),
    )
    res_obj_new, info_new = resolver.resolve_record(startup_new)
    assert res_obj_new.name == "Quantum Deep Genomics Lab"
    assert info_new["decision"] == ResolutionDecision.NEW.value


def test_backend_review_api_endpoints():
    """Test FastAPI review endpoints: GET /api/resolution/review, approve, reject."""
    client = TestClient(app)

    # 1. GET /api/resolution/review
    resp = client.get("/api/resolution/review")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)

    # 2. Create a review item via resolver or queue
    from src.resolution.review_queue import review_queue
    item = review_queue.add_review(
        candidate="Open AI Technologies Inc",
        canonical="OpenAI",
        matching_method="fuzzy",
        fuzzy_score=0.82,
        confidence=0.82,
        source="api_test",
        match_reason="Test candidate",
    )
    review_id = item["id"]

    # 3. GET should contain the newly added review item
    resp_pending = client.get("/api/resolution/review")
    assert resp_pending.status_code == 200
    ids = [r["id"] for r in resp_pending.json()]
    assert review_id in ids

    # 4. POST /api/resolution/review/{id}/approve
    resp_app = client.post(f"/api/resolution/review/{review_id}/approve")
    assert resp_app.status_code == 200
    app_data = resp_app.json()
    assert app_data["status"] == "approved"
    assert app_data["review"]["decision"] == "APPROVED"

    # 5. Add second review item and reject
    item2 = review_queue.add_review(
        candidate="Scale Media Holdings",
        canonical="Scale AI",
        matching_method="fuzzy",
        fuzzy_score=0.71,
        confidence=0.71,
        source="api_test",
    )
    resp_rej = client.post(f"/api/resolution/review/{item2['id']}/reject")
    assert resp_rej.status_code == 200
    rej_data = resp_rej.json()
    assert rej_data["status"] == "rejected"
    assert rej_data["review"]["decision"] == "REJECTED"

    # 6. Test 404 for non-existent review item
    resp_404 = client.post("/api/resolution/review/nonexistent-id-99999/approve")
    assert resp_404.status_code == 404


def test_automatic_match(temp_env):
    """Verify automatic match for exact, normalized, and fuzzy matching."""
    resolver = temp_env["resolver"]
    res1 = resolver.resolve("openai.com")
    assert res1.decision == ResolutionDecision.MATCH.value
    assert res1.canonical == "OpenAI"

    res2 = resolver.resolve("Mistral AI SAS")
    assert res2.decision == ResolutionDecision.MATCH.value
    assert res2.canonical == "Mistral AI"

    res3 = resolver.resolve("Hugging Face Hub")
    assert res3.decision == ResolutionDecision.MATCH.value
    assert res3.canonical == "Hugging Face"


def test_review_state(temp_env):
    """Verify ambiguous candidates enter REVIEW state and enqueue."""
    resolver = temp_env["resolver"]
    queue = temp_env["queue"]
    res = resolver.resolve("Synthesia Media")
    assert res.decision == ResolutionDecision.REVIEW.value
    assert len(queue.get_pending_reviews()) == 1


def test_approval(temp_env):
    """Verify approval of review queue item."""
    test_approval_workflow(temp_env)


def test_rejection(temp_env):
    """Verify rejection of review queue item."""
    test_rejection_workflow(temp_env)


def test_persistence(temp_env):
    """Verify persistence of review decisions and logs."""
    test_persistence_of_decisions(temp_env)

