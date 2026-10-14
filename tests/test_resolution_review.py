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


def test_invalid_repeated_review_transitions(temp_env):
    """Test state transition validations: approving already rejected or re-approving fails safely."""
    queue = temp_env["queue"]

    # 1. Enqueue item
    item = queue.add_review(
        candidate="Transition Test Co",
        canonical="Cohere",
        matching_method="fuzzy",
        fuzzy_score=0.74,
        confidence=0.74,
    )
    rev_id = item["id"]

    # 2. Reject first
    rejected = queue.reject_review(rev_id, notes="First rejection")
    assert rejected["status"] == "REJECTED"

    # 3. Attempting to approve an already rejected review must fail safely
    with pytest.raises(ValueError, match="Cannot approve review.*status 'REJECTED'"):
        queue.approve_review(rev_id, notes="Attempted approve after reject")

    # 4. Attempting to reject an already rejected review must fail safely
    with pytest.raises(ValueError, match="already rejected"):
        queue.reject_review(rev_id)

    # 5. Test another item for approve -> reject transition
    item2 = queue.add_review(
        candidate="Transition Test Co 2",
        canonical="Anthropic",
        matching_method="fuzzy",
        fuzzy_score=0.79,
        confidence=0.79,
    )
    rev_id2 = item2["id"]
    approved = queue.approve_review(rev_id2, notes="First approval")
    assert approved["status"] == "APPROVED"

    # 6. Attempting to reject an already approved review must fail safely
    with pytest.raises(ValueError, match="Cannot reject review.*status 'APPROVED'"):
        queue.reject_review(rev_id2, notes="Attempted reject after approve")

    # 7. Attempting to approve an already approved review must fail safely
    with pytest.raises(ValueError, match="already approved"):
        queue.approve_review(rev_id2)


def test_audit_trail_creation(temp_env):
    """Test immutable audit trail records enqueue, approve, and reject actions with reviewer and state."""
    queue = temp_env["queue"]

    item = queue.add_review(
        candidate="Audit Test Entity",
        canonical="OpenAI",
        matching_method="fuzzy",
        fuzzy_score=0.76,
        confidence=0.76,
        source="audit_crawl",
        match_reason="Ambiguous similarity test",
        signals={"test_signal": 123},
    )
    rev_id = item["id"]

    # Audit trail should contain initial ENQUEUE record
    trail_1 = queue.get_review_audit_trail(rev_id)
    assert len(trail_1) == 1
    enq = trail_1[0]
    assert enq["review_id"] == rev_id
    assert enq["reviewer_action"] == "ENQUEUE"
    assert enq["previous_state"] == "NONE"
    assert enq["new_state"] == "PENDING"
    assert enq["original_resolution"] == "NEEDS_REVIEW"
    assert enq["original_confidence"] == 0.76

    # Approve item
    queue.approve_review(rev_id, notes="Approved by Lead Reviewer", reviewer="reviewer_bob")

    # Audit trail should now contain 2 records
    trail_2 = queue.get_review_audit_trail(rev_id)
    assert len(trail_2) == 2
    appr = trail_2[1]
    assert appr["review_id"] == rev_id
    assert appr["reviewer_action"] == "APPROVE"
    assert appr["previous_state"] == "PENDING"
    assert appr["new_state"] == "APPROVED"
    assert appr["reviewer"] == "reviewer_bob"
    assert appr["notes"] == "Approved by Lead Reviewer"
    assert appr["timestamp"] is not None

    # Global audit trail query
    all_audit = queue.get_review_audit_trail()
    assert len(all_audit) >= 2


def test_candidate_ranking_and_generation(temp_env):
    """Test candidate generation separates blocking retrieval from scoring and ranks candidates."""
    resolver = temp_env["resolver"]

    # Candidate generation for exact match
    exact_cands = resolver.generate_candidates("OpenAI")
    assert len(exact_cands) >= 1
    assert any(c[0] == "OpenAI" for c in exact_cands)

    # Candidate generation for fuzzy name
    fuzzy_cands = resolver.generate_candidates("Synthesia Media")
    assert len(fuzzy_cands) >= 1
    assert any(c[0] == "Synthesia" for c in fuzzy_cands)

    # Completely disparate input generates no candidates
    none_cands = resolver.generate_candidates("Xy99ZzTotallyUnknownCorp999")
    assert len(none_cands) == 0

    # Candidate scoring ranks and scores candidates
    best_canon, method, conf, fuzzy, signals, reason = resolver.score_candidates(
        "Synthesia Media", fuzzy_cands
    )
    assert best_canon == "Synthesia"
    assert 0.65 <= conf < 0.85
    assert "token_sort_ratio" in signals


def test_deterministic_confidence_score(temp_env):
    """Test confidence scoring is deterministic, explainable, and bounded in [0.0, 1.0]."""
    resolver = temp_env["resolver"]

    res1 = resolver.resolve("Hugging Face Hub")
    res2 = resolver.resolve("Hugging Face Hub")
    assert res1.confidence == res2.confidence
    assert res1.fuzzy_score == res2.fuzzy_score
    assert res1.decision == res2.decision
    assert 0.0 <= res1.confidence <= 1.0
    assert 0.0 <= res1.fuzzy_score <= 1.0

    # Signals must explain the match
    assert "token_sort_ratio" in res1.signals
    assert res1.match_reason != ""


def test_metrics_updates_on_resolution_and_reviews(temp_env):
    """Test metrics collector tracks auto_matched, needs_review, no_match, approvals, and rejections."""
    from src.observability.metrics import metrics_collector

    metrics_collector.reset_er_metrics()
    resolver = temp_env["resolver"]
    queue = temp_env["queue"]

    # 1. AUTO_MATCH
    resolver.resolve("openai.com")
    stats = metrics_collector.get_er_metrics()
    assert stats["entity_resolution_auto_matched"] == 1
    assert stats["entity_resolution_total"] == 1

    # 2. NEEDS_REVIEW
    resolver.resolve("Synthesia Media")
    stats = metrics_collector.get_er_metrics()
    assert stats["entity_resolution_needs_review"] == 1
    assert stats["entity_resolution_total"] == 2

    # 3. NO_MATCH
    resolver.resolve("Completely Random Nonexistent Entity 12345")
    stats = metrics_collector.get_er_metrics()
    assert stats["entity_resolution_no_match"] == 1
    assert stats["entity_resolution_total"] == 3

    # 4. Review approval
    item = queue.add_review(
        candidate="Metric Test Candidate",
        canonical="OpenAI",
        matching_method="fuzzy",
        fuzzy_score=0.75,
        confidence=0.75,
    )
    queue.approve_review(item["id"])
    stats = metrics_collector.get_er_metrics()
    assert stats["entity_resolution_review_approved"] == 1

    # 5. Review rejection
    item2 = queue.add_review(
        candidate="Metric Test Candidate 2",
        canonical="Anthropic",
        matching_method="fuzzy",
        fuzzy_score=0.71,
        confidence=0.71,
    )
    queue.reject_review(item2["id"])
    stats = metrics_collector.get_er_metrics()
    assert stats["entity_resolution_review_rejected"] == 1

    # Prometheus export check
    prom_text = metrics_collector.generate_prometheus_format()
    assert "graphone_entity_resolution_total" in prom_text
    assert "graphone_entity_resolution_auto_matched" in prom_text
    assert "graphone_entity_resolution_needs_review" in prom_text
    assert "graphone_entity_resolution_no_match" in prom_text
    assert "graphone_entity_resolution_review_approved" in prom_text
    assert "graphone_entity_resolution_review_rejected" in prom_text


def test_review_api_comprehensive():
    """Test full Review API suite: GET /reviews, filtering, get by ID, audit, and 409 transition errors."""
    client = TestClient(app)
    from src.resolution.review_queue import review_queue

    # 1. Add review item
    item = review_queue.add_review(
        candidate="API Review Test Corp",
        canonical="Cohere",
        matching_method="fuzzy",
        fuzzy_score=0.77,
        confidence=0.77,
        source="api_suite",
    )
    rev_id = item["id"]

    # 2. GET /reviews (defaults to pending)
    res_list = client.get("/reviews")
    assert res_list.status_code == 200
    ids = [r["id"] for r in res_list.json()]
    assert rev_id in ids

    # 3. GET /reviews?status=pending
    res_pending = client.get("/reviews?status=pending")
    assert res_pending.status_code == 200

    # 4. GET /reviews/{review_id}
    res_get = client.get(f"/reviews/{rev_id}")
    assert res_get.status_code == 200
    data = res_get.json()
    assert data["id"] == rev_id
    assert "audit_trail" in data

    # 5. GET /reviews/{review_id}/audit
    res_audit = client.get(f"/reviews/{rev_id}/audit")
    assert res_audit.status_code == 200
    assert len(res_audit.json()) >= 1

    # 6. GET /reviews/stats
    res_stats = client.get("/reviews/stats")
    assert res_stats.status_code == 200
    assert "pending" in res_stats.json()
    assert "total" in res_stats.json()

    # 7. POST /reviews/{review_id}/approve with payload
    res_app = client.post(
        f"/reviews/{rev_id}/approve",
        json={"reviewer": "reviewer_alice", "notes": "Approved for merging"},
    )
    assert res_app.status_code == 200
    assert res_app.json()["status"] == "approved"
    assert res_app.json()["review"]["status"] == "APPROVED"

    # 8. POST /reviews/{review_id}/reject after approve -> 409 Conflict
    res_rej_conflict = client.post(
        f"/reviews/{rev_id}/reject",
        json={"reviewer": "reviewer_alice"},
    )
    assert res_rej_conflict.status_code == 409
    assert "Cannot reject review" in res_rej_conflict.json()["detail"]

    # 9. POST /reviews/{review_id}/approve repeated -> 409 Conflict
    res_app_repeated = client.post(f"/reviews/{rev_id}/approve")
    assert res_app_repeated.status_code == 409


def test_concurrent_review_safety(temp_env):
    """Test concurrent review approve/reject attempts execute safely without corrupting SQLite state."""
    import concurrent.futures
    queue = temp_env["queue"]

    item = queue.add_review(
        candidate="Concurrent Candidate",
        canonical="OpenAI",
        matching_method="fuzzy",
        fuzzy_score=0.76,
        confidence=0.76,
    )
    rev_id = item["id"]

    success_count = 0
    conflict_count = 0

    def attempt_action(action: str):
        if action == "approve":
            try:
                return queue.approve_review(rev_id, reviewer="worker_a")
            except ValueError:
                return "conflict"
        else:
            try:
                return queue.reject_review(rev_id, reviewer="worker_b")
            except ValueError:
                return "conflict"

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(attempt_action, "approve"),
            executor.submit(attempt_action, "approve"),
            executor.submit(attempt_action, "reject"),
            executor.submit(attempt_action, "reject"),
        ]
        results = [f.result() for f in futures]

    for res in results:
        if isinstance(res, dict):
            success_count += 1
        elif res == "conflict":
            conflict_count += 1

    # Exactly 1 thread must succeed; all others must safely conflict
    assert success_count == 1
    assert conflict_count == 3
    final_item = queue.get_review_by_id(rev_id)
    assert final_item["status"] in ("APPROVED", "REJECTED")
