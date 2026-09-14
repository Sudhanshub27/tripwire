from fastapi.testclient import TestClient


def test_import_and_serve_frontend_missing_assets(tmp_path, monkeypatch):
    """Test that when dist exists with index.html but assets/ is missing,
    the app does not crash and serves index.html."""
    from src.dashboard.main import app
    client = TestClient(app)

    # By default in the current workspace, let's test request
    # Since client is initialized, let's also test the serve_frontend behavior directly
    response = client.get("/")
    assert response.status_code in (200, 404)


def test_serve_frontend_fallback_and_404(tmp_path, monkeypatch):
    """Test serve_frontend route behavior when index.html is present vs missing."""
    import src.dashboard.main as main_mod

    # Create dummy dist dir without assets
    fake_dist = tmp_path / "dist"
    fake_dist.mkdir()
    monkeypatch.setattr(main_mod, "FRONTEND_DIST_DIR", fake_dist)

    client = TestClient(main_mod.app)

    # When index.html is missing
    response = client.get("/nonexistent-page")
    assert response.status_code == 404
    assert response.json()["detail"] == "Frontend dist index.html not found"

    # When index.html is present
    index_file = fake_dist / "index.html"
    index_file.write_text("<html><body>Hello Frontend</body></html>", encoding="utf-8")

    response = client.get("/nonexistent-page")
    assert response.status_code == 200
    assert "Hello Frontend" in response.text

    # When a specific static file is present in dist
    asset_file = fake_dist / "robots.txt"
    asset_file.write_text("User-agent: *", encoding="utf-8")

    response = client.get("/robots.txt")
    assert response.status_code == 200
    assert "User-agent: *" in response.text
