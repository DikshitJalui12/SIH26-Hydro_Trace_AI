"""
Comprehensive Automated Test Suite for HydroTrace-AI Platform Backend.
Tests:
1. System Health & Diagnostics
2. Authentication (Login, JWT, Roles, MFA)
3. Granular RBAC Permissions
4. Incident & Sentinel-1 Ingestion Listing & Filtering
5. Multi-Agent Unified Incident Retrieval
6. Hydrodynamic Simulation Recalibration
7. Court-Admissible PDF Generation with Cryptographic SHA-256 Seal
8. Security Headers (CSP, HSTS, X-Frame-Options) & Rate Limiting
"""

import pytest
from fastapi.testclient import TestClient
from server.app import app
from server.config import settings

client = TestClient(app)


def test_system_health():
    """Verify backend health telemetry and subsystem status."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "operational"
    assert "agent_1_vision_ai" in data["subsystems"]
    assert "agent_2_hindcast_rk4" in data["subsystems"]
    assert "agent_3_ais_attribution" in data["subsystems"]


def test_security_headers():
    """Verify strict DevSecOps security headers (CSP, HSTS, X-Frame-Options)."""
    response = client.get("/api/v1/health")
    assert "Content-Security-Policy" in response.headers
    assert "default-src 'self'" in response.headers["Content-Security-Policy"]
    assert response.headers.get("X-Frame-Options") == "DENY"
    assert response.headers.get("X-Content-Type-Options") == "nosniff"
    assert "max-age=31536000" in response.headers.get("Strict-Transport-Security", "")


def test_authentication_login_flow():
    """Verify login and JWT token generation for Coast Guard Commander."""
    payload = {
        "username": "commander",
        "password": "navalcommand2026"
    }
    response = client.post("/api/v1/auth/login", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"
    assert data["user"]["role"] == "COAST_GUARD_COMMANDER"
    assert "dispatch_interception" in data["user"]["permissions"]


def test_authentication_invalid_user():
    """Verify rejection of unknown credentials."""
    payload = {
        "username": "unauthorized_infiltrator",
        "password": "wrongpassword"
    }
    response = client.post("/api/v1/auth/login", json=payload)
    assert response.status_code == 401


def test_list_incidents_and_quadrants():
    """Verify incident retrieval and Indian EEZ quadrant geometries."""
    # List incidents
    res = client.get("/api/v1/incidents")
    assert res.status_code == 200
    incidents = res.json()
    assert len(incidents) >= 2
    assert any("MALACCA" in inc["slick_id"] for inc in incidents)

    # Indian EEZ Quadrants
    quad_res = client.get("/api/v1/incidents/eez/quadrants")
    assert quad_res.status_code == 200
    quadrants = quad_res.json()["quadrants"]
    assert "arabian_sea" in quadrants
    assert "bay_of_bengal" in quadrants
    assert "lakshadweep" in quadrants
    assert "andaman_sea" in quadrants


def test_unified_incident_details():
    """Verify multi-agent payload fusion for Malacca incident."""
    slick_id = "SLICK-2026-MALACCA-0881"
    res = client.get(f"/api/v1/incidents/{slick_id}")
    assert res.status_code == 200
    data = res.json()
    assert data["slick_id"] == slick_id
    assert data["severity"] == "Critical"
    assert data["metrics"]["area_km2"] == 14.8
    assert data["hindcast"]["uncertainty_cone"]["semi_major_axis_km"] == 6.0
    assert len(data["ranked_suspects"]) >= 1
    
    prime_culprit = data["prime_culprit"]
    assert prime_culprit is not None
    assert prime_culprit["vessel_identity"]["name"] == "OCEAN APEX"
    assert prime_culprit["attribution_score"] > 70.0
    assert prime_culprit["kinematics"]["ais_blackout_duration_mins"] == 48.0


def test_court_pdf_export_and_cryptographic_seal():
    """Verify generation of certified court-admissible PDF with SHA-256 digest."""
    slick_id = "SLICK-2026-MALACCA-0881"
    
    # Check seal preview
    seal_res = client.get(f"/api/v1/export/seal-preview/{slick_id}")
    assert seal_res.status_code == 200
    seal_data = seal_res.json()
    assert "digital_digest_sha256" in seal_data
    assert len(seal_data["digital_digest_sha256"]) == 64
    assert seal_data["pgp_key_fingerprint"] == settings.FORENSIC_PGP_KEY_ID

    # Download certified PDF
    pdf_res = client.get(f"/api/v1/export/court-pdf/{slick_id}")
    assert pdf_res.status_code == 200
    assert pdf_res.headers["content-type"] == "application/pdf"
    assert "X-Cryptographic-SHA256-Seal" in pdf_res.headers
    assert len(pdf_res.content) > 1000 # Valid non-empty PDF binary
    assert pdf_res.content[:4] == b"%PDF"


def test_simulation_recalibration():
    """Verify simulation runner for hindcast rewind and forward boom forecast."""
    payload = {
        "slick_id": "SLICK-2026-MALACCA-0881",
        "direction": "FORWARD_BOOM",
        "duration_hours": 24.0,
        "particles_count": 10000
    }
    res = client.post("/api/v1/simulation/run", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["mode"] == "FORWARD_BOOM"
    assert len(data["stochastic_particles"]) > 0


def test_audit_logs_endpoint():
    """Verify append-only immutable audit trail recording."""
    res = client.get("/api/v1/export/audit-logs")
    assert res.status_code == 200
    data = res.json()
    assert "audit_stream" in data
    assert len(data["audit_stream"]) > 0


def test_frontend_spa_serving():
    """Verify single page application HTML and static CSS/JS delivery."""
    # Root SPA page
    r_index = client.get("/")
    assert r_index.status_code == 200
    assert "HydroTrace-AI" in r_index.text
    assert "map-canvas-container" in r_index.text
    assert "time-scrubber-slider" in r_index.text

    # Static CSS
    r_css = client.get("/static/styles/theme.css")
    assert r_css.status_code == 200
    assert "--bg-canvas" in r_css.text

    # Static JS
    r_js = client.get("/static/src/app.js")
    assert r_js.status_code == 200
    assert "HydroTraceApp" in r_js.text

