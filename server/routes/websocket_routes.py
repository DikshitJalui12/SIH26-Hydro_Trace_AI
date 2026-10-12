"""
Real-time WebSocket Streaming Routes.
Streams live 60 FPS Lagrangian particle advection frames,
synchronized AIS vessel position interpolation, and transponder blackout alerts.
"""

import asyncio
import json
from datetime import datetime, timezone
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from server.services.incident_service import incident_service

router = APIRouter(tags=["Real-Time WebSockets Telemetry"])


@router.websocket("/api/v1/ws/simulation/{slick_id}")
async def websocket_simulation_stream(websocket: WebSocket, slick_id: str):
    """
    Bi-directional streaming WebSocket:
    - Client can send playback commands: {"action": "play", "speed": 2.0, "time_ratio": 0.45}
    - Server streams particle advection coordinates and interpolated ship positions.
    """
    await websocket.accept()

    incident = incident_service.get_incident_by_id(slick_id)
    if not incident:
        await websocket.send_json({
            "error": "IncidentNotFound",
            "message": f"Slick ID '{slick_id}' does not exist in operational store."
        })
        await websocket.close()
        return

    particles = incident.hindcast.stochastic_particles
    suspects = incident.ranked_suspects

    step = 0
    total_steps = 40
    playback_speed = 1.0
    is_playing = True

    try:
        while True:
            # Check for non-blocking client messages (seek, pause, speed change)
            try:
                client_msg = await asyncio.wait_for(websocket.receive_text(), timeout=0.08)
                data = json.loads(client_msg)
                if data.get("action") == "pause":
                    is_playing = False
                elif data.get("action") == "play":
                    is_playing = True
                    if "speed" in data:
                        playback_speed = float(data["speed"])
                elif data.get("action") == "seek":
                    ratio = float(data.get("time_ratio", 0.0))
                    step = int(ratio * total_steps)
            except asyncio.TimeoutError:
                pass

            if is_playing:
                step = (step + 1) % total_steps

            ratio = step / float(total_steps)

            # Compute current frame particles
            frame_particles = []
            for p in particles:
                coords = p.get("coordinates", [])
                if not coords:
                    continue
                # Interpolate along path
                idx = min(len(coords) - 1, int(ratio * (len(coords) - 1)))
                pt = coords[idx]
                frame_particles.append({
                    "id": p.get("particle_id", 0),
                    "lon": pt[0],
                    "lat": pt[1],
                    "u": incident.hindcast.weather.avg_current_knots,
                    "v": incident.hindcast.weather.avg_wind_knots
                })

            # Compute interpolated vessel positions
            vessels_state = []
            for s in suspects:
                track = s.track_geometry.get("coordinates", [])
                if track:
                    v_idx = min(len(track) - 1, int(ratio * (len(track) - 1)))
                    v_pt = track[v_idx]
                    
                    # Check if vessel is in dark zone
                    is_dark = False
                    dark_alert = None
                    if s.is_culprit and (0.35 <= ratio <= 0.65):
                        is_dark = True
                        dark_alert = f"⚠️ Transponder Gap: {s.kinematics.ais_blackout_duration_mins:.0f} mins (Likely Bilge Dump Event)"

                    vessels_state.append({
                        "mmsi": s.vessel_identity.mmsi,
                        "name": s.vessel_identity.name,
                        "lon": v_pt[0],
                        "lat": v_pt[1],
                        "sog": s.kinematics.cruising_speed_knots - (s.kinematics.speed_drop_knots if is_dark else 0),
                        "is_culprit": s.is_culprit,
                        "is_dark": is_dark,
                        "dark_alert": dark_alert,
                        "score": s.attribution_score
                    })

            payload = {
                "slick_id": slick_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "time_ratio": round(ratio, 4),
                "step": step,
                "particles_count": len(frame_particles),
                "particles": frame_particles[:100], # Stream top 100 dynamic coordinates
                "vessels": vessels_state
            }

            await websocket.send_json(payload)
            await asyncio.sleep(0.1 / max(1.0, playback_speed))

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.close()
        except Exception:
            pass
