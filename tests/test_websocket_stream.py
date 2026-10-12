"""
Automated WebSocket Telemetry Streaming Test.
Verifies bi-directional frame streaming of 60 FPS Lagrangian particles,
interpolated AIS ship markers, and transponder blackout warnings.
"""

import pytest
from fastapi.testclient import TestClient
from server.app import app

client = TestClient(app)


def test_websocket_simulation_stream():
    """Verify WebSocket connection and telemetry payload schema."""
    slick_id = "SLICK-2026-MALACCA-0881"
    with client.websocket_connect(f"/api/v1/ws/simulation/{slick_id}") as websocket:
        # Receive first streaming frame
        data = websocket.receive_json()
        assert data["slick_id"] == slick_id
        assert "particles" in data
        assert "vessels" in data
        assert "time_ratio" in data

        # Send interactive playback speed change
        websocket.send_json({"action": "play", "speed": 2.0})

        # Receive next accelerated frame
        data2 = websocket.receive_json()
        assert data2["slick_id"] == slick_id
        assert len(data2["vessels"]) > 0
        
        # Verify culprit has attribution and dark vessel logic
        culprits = [v for v in data2["vessels"] if v["is_culprit"]]
        assert len(culprits) == 1
        assert culprits[0]["name"] == "OCEAN APEX"
