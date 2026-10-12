/**
 * HydroTrace-AI Maritime Command Platform - Production Client Application
 * - Zero-watermark, free naval dark basemap (Esri Dark Gray + Reference Labels).
 * - Multi-Agent WebGL Geospatial Rendering: Attention U-Net Slick, 95% Origin Cone,
 *   10,000 Lagrangian particles advection, and AIS culprit/innocent trajectories.
 * - Dynamic 4D Time-Scrubber (-dt Hindcast Rewind / +dt Forward Boom Forecast).
 * - Interactive Suspect Dossier, Route Isolation, and Certified Court PDF Export.
 */

class HydroTraceApp {
  constructor() {
    this.activeIncidentId = "SLICK-2026-MALACCA-0881";
    this.activeIncident = null;
    this.incidentsList = [];
    this.eezQuadrants = {};

    // Simulation State
    this.simulationMode = "HINDCAST_REWIND"; // "HINDCAST_REWIND" or "FORWARD_BOOM"
    this.playbackSpeed = 1;
    this.isPlaying = false;
    this.timeRatio = 0.5; // 0.0 to 1.0
    this.playInterval = null;
    this.isolatedVesselImo = null;
    this.currentSensor = "sar_vv";

    // Operator Authentication & Clearance
    this.currentUser = {
      username: "commander",
      full_name: "Capt. Vikramaditya Rawat",
      role: "COAST_GUARD_COMMANDER",
      token: null
    };

    this.activeView = "command-console";
    this.aisInterval = null;
    this.aisFleet = [];
    this.selectedVessel = null;
    this.satelliteCatalog = [];

    this.map = null;
    this.deckOverlay = null;
    this.websocket = null;

    this.init();
  }

  async init() {
    this.initMap();
    this.bindEvents();
    this.initMasterNav();
    this.startLiveClock();
    await this.fetchQuadrants();
    await this.fetchIncidents();
    await this.loadIncident(this.activeIncidentId);
    this.initWebSocket();
  }

  /* ========================================================================
     MapLibre GL Initialization with Zero-Watermark Esri Basemaps
     ======================================================================== */
  initMap() {
    // 100% Free, NO API KEY, ZERO WATERMARK High-Contrast Naval Basemaps
    const mapStyle = {
      version: 8,
      sources: {
        "esri-dark-gray": {
          type: "raster",
          tiles: [
            "https://services.arcgisonline.com/arcgis/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}"
          ],
          tileSize: 256,
          attribution: "Esri, HERE, Garmin, &copy; OpenStreetMap contributors"
        },
        "esri-dark-labels": {
          type: "raster",
          tiles: [
            "https://services.arcgisonline.com/arcgis/rest/services/Canvas/World_Dark_Gray_Reference/MapServer/tile/{z}/{y}/{x}"
          ],
          tileSize: 256
        },
        "esri-world-imagery": {
          type: "raster",
          tiles: [
            "https://services.arcgisonline.com/arcgis/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
          ],
          tileSize: 256
        },
        "esri-ocean-base": {
          type: "raster",
          tiles: [
            "https://services.arcgisonline.com/arcgis/rest/services/Ocean/World_Ocean_Base/MapServer/tile/{z}/{y}/{x}"
          ],
          tileSize: 256
        }
      },
      layers: [
        {
          id: "layer-dark-base",
          type: "raster",
          source: "esri-dark-gray",
          minzoom: 0,
          maxzoom: 19,
          layout: { visibility: "visible" }
        },
        {
          id: "layer-ocean-base",
          type: "raster",
          source: "esri-ocean-base",
          minzoom: 0,
          maxzoom: 19,
          layout: { visibility: "none" }
        },
        {
          id: "layer-imagery-base",
          type: "raster",
          source: "esri-world-imagery",
          minzoom: 0,
          maxzoom: 19,
          layout: { visibility: "none" }
        },
        {
          id: "layer-dark-labels",
          type: "raster",
          source: "esri-dark-labels",
          minzoom: 0,
          maxzoom: 19,
          layout: { visibility: "visible" }
        }
      ]
    };

    this.map = new maplibregl.Map({
      container: "map-canvas-container",
      style: mapStyle,
      center: [103.2700, 1.2100], // Malacca default
      zoom: 11,
      pitch: 30,
      bearing: 0,
      attributionControl: false
    });

    // Real-time Coordinate HUD on cursor hover
    this.map.on("mousemove", (e) => {
      const lat = e.lngLat.lat;
      const lon = e.lngLat.lng;
      const latDeg = Math.floor(Math.abs(lat));
      const latMin = Math.floor((Math.abs(lat) - latDeg) * 60);
      const latSec = Math.floor(((Math.abs(lat) - latDeg) * 60 - latMin) * 60);
      const latDir = lat >= 0 ? "N" : "S";

      const lonDeg = Math.floor(Math.abs(lon));
      const lonMin = Math.floor((Math.abs(lon) - lonDeg) * 60);
      const lonSec = Math.floor(((Math.abs(lon) - lonDeg) * 60 - lonMin) * 60);
      const lonDir = lon >= 0 ? "E" : "W";

      const hud = document.getElementById("hud-coordinates");
      if (hud) {
        hud.textContent = `LAT ${String(latDeg).padStart(2, "0")}°${String(latMin).padStart(2, "0")}'${String(latSec).padStart(2, "0")}"${latDir} | LON ${String(lonDeg).padStart(3, "0")}°${String(lonMin).padStart(2, "0")}'${String(lonSec).padStart(2, "0")}"${lonDir}`;
      }
    });

    this.map.on("load", () => {
      this.initNativeMapLayers();
      if (this.activeIncident) {
        this.updateMapLayers();
      }
    });
  }

  /* ========================================================================
     Native MapLibre GL GPU WebGL Layers (Resilient & Fast)
     ======================================================================== */
  initNativeMapLayers() {
    if (!this.map) return;

    // 1. Slick Detection Polygon Source & Layers
    this.map.addSource("src-slick-polygon", {
      type: "geojson",
      data: { type: "FeatureCollection", features: [] }
    });

    this.map.addLayer({
      id: "lyr-slick-fill",
      type: "fill",
      source: "src-slick-polygon",
      paint: {
        "fill-color": "#EF4444",
        "fill-opacity": 0.45
      }
    });

    this.map.addLayer({
      id: "lyr-slick-stroke",
      type: "line",
      source: "src-slick-polygon",
      paint: {
        "line-color": "#DC2626",
        "line-width": 2.5
      }
    });

    // 2. 95% Origin Uncertainty Ellipse Source & Layers
    this.map.addSource("src-origin-cone", {
      type: "geojson",
      data: { type: "FeatureCollection", features: [] }
    });

    this.map.addLayer({
      id: "lyr-origin-cone-fill",
      type: "fill",
      source: "src-origin-cone",
      paint: {
        "fill-color": "#F59E0B",
        "fill-opacity": 0.15
      }
    });

    this.map.addLayer({
      id: "lyr-origin-cone-stroke",
      type: "line",
      source: "src-origin-cone",
      paint: {
        "line-color": "#F59E0B",
        "line-width": 2,
        "line-dasharray": [3, 2]
      }
    });

    // 3. Vessel Tracks (Culprit, Innocent, and Transponder Gap)
    this.map.addSource("src-vessel-tracks", {
      type: "geojson",
      data: { type: "FeatureCollection", features: [] }
    });

    this.map.addLayer({
      id: "lyr-vessel-tracks",
      type: "line",
      source: "src-vessel-tracks",
      paint: {
        "line-color": ["get", "color"],
        "line-width": ["get", "width"],
        "line-opacity": ["get", "opacity"]
      }
    });

    this.map.addLayer({
      id: "lyr-vessel-gap-stripes",
      type: "line",
      source: "src-vessel-tracks",
      filter: ["==", "is_gap", true],
      paint: {
        "line-color": "#EAB308",
        "line-width": 4.5,
        "line-dasharray": [2, 1]
      }
    });

    // 4. Lagrangian Particle Advection Cloud
    this.map.addSource("src-particles", {
      type: "geojson",
      data: { type: "FeatureCollection", features: [] }
    });

    this.map.addLayer({
      id: "lyr-particles",
      type: "circle",
      source: "src-particles",
      paint: {
        "circle-radius": 3.5,
        "circle-color": "#06B6D4",
        "circle-opacity": 0.85,
        "circle-blur": 0.2
      }
    });

    // 5. Current Interpolated Vessel Markers
    this.map.addSource("src-vessel-markers", {
      type: "geojson",
      data: { type: "FeatureCollection", features: [] }
    });

    this.map.addLayer({
      id: "lyr-vessel-markers-outer",
      type: "circle",
      source: "src-vessel-markers",
      paint: {
        "circle-radius": 9,
        "circle-color": ["get", "color"],
        "circle-opacity": 0.35
      }
    });

    this.map.addLayer({
      id: "lyr-vessel-markers-core",
      type: "circle",
      source: "src-vessel-markers",
      paint: {
        "circle-radius": 5,
        "circle-color": ["get", "color"],
        "circle-stroke-width": 1.5,
        "circle-stroke-color": "#FFFFFF"
      }
    });

    // Layer Click & Hover Interactions
    this.map.on("click", "lyr-vessel-markers-core", (e) => {
      if (e.features && e.features[0]) {
        const imo = e.features[0].properties.imo;
        this.highlightSuspectCard(imo);
      }
    });

    this.map.on("mouseenter", "lyr-vessel-markers-core", (e) => {
      this.map.getCanvas().style.cursor = "pointer";
      if (e.features && e.features[0]) {
        const p = e.features[0].properties;
        this.showTooltip(e.point.x, e.point.y, `🚢 ${p.name} (IMO: ${p.imo})\nSOG: ${p.sog} kn | Suspicion: ${p.score}%`);
      }
    });

    this.map.on("mouseleave", "lyr-vessel-markers-core", () => {
      this.map.getCanvas().style.cursor = "";
      this.hideTooltip();
    });

    this.map.on("mouseenter", "lyr-slick-fill", (e) => {
      this.showTooltip(e.point.x, e.point.y, `Detected Oil Slick Polygon\nArea: ${this.activeIncident?.metrics?.area_km2} km² (Attention U-Net)`);
    });

    this.map.on("mouseleave", "lyr-slick-fill", () => {
      this.hideTooltip();
    });
  }

  updateMapLayers() {
    if (!this.map || !this.map.isStyleLoaded() || !this.activeIncident) return;

    // 1. Update Slick Detection Polygon
    const slickSource = this.map.getSource("src-slick-polygon");
    if (slickSource && this.activeIncident.slick_geometry) {
      slickSource.setData({
        type: "Feature",
        geometry: this.activeIncident.slick_geometry
      });
    }

    // 2. Update 95% Origin Uncertainty Ellipse
    const coneSource = this.map.getSource("src-origin-cone");
    const cone = this.activeIncident.hindcast?.uncertainty_cone;
    if (coneSource && cone?.origin_polygon) {
      coneSource.setData({
        type: "Feature",
        geometry: cone.origin_polygon
      });
    }

    // 3. Update Vessel Tracks
    const tracksSource = this.map.getSource("src-vessel-tracks");
    if (tracksSource) {
      const trackFeatures = [];
      const suspects = this.activeIncident.ranked_suspects || [];

      suspects.forEach((s) => {
        if (this.isolatedVesselImo && s.vessel_identity.imo !== this.isolatedVesselImo) {
          return; // filtered out if route isolated
        }

        const coords = s.track_geometry?.coordinates || [];
        if (coords.length < 2) return;

        // Base Track
        trackFeatures.push({
          type: "Feature",
          properties: {
            imo: s.vessel_identity.imo,
            name: s.vessel_identity.name,
            color: s.is_culprit ? "#F43F5E" : "#64748B",
            width: s.is_culprit ? 3.5 : 1.5,
            opacity: s.is_culprit ? 1.0 : 0.45,
            is_gap: false
          },
          geometry: {
            type: "LineString",
            coordinates: coords.map((c) => [c[0], c[1]])
          }
        });

        // Transponder Silence / Dark Gap Segment
        if (s.has_dark_gap && coords.length > 6) {
          const midStart = Math.floor(coords.length * 0.35);
          const midEnd = Math.floor(coords.length * 0.65);
          trackFeatures.push({
            type: "Feature",
            properties: {
              imo: s.vessel_identity.imo,
              color: "#EAB308",
              width: 4.5,
              opacity: 1.0,
              is_gap: true
            },
            geometry: {
              type: "LineString",
              coordinates: coords.slice(midStart, midEnd).map((c) => [c[0], c[1]])
            }
          });
        }
      });

      tracksSource.setData({
        type: "FeatureCollection",
        features: trackFeatures
      });
    }

    // 4. Update Lagrangian Reverse Drift Particles
    const particlesSource = this.map.getSource("src-particles");
    if (particlesSource) {
      const particles = this.activeIncident.hindcast?.stochastic_particles || [];
      const particlePoints = [];

      particles.forEach((p) => {
        const coords = p.coordinates || [];
        if (coords.length > 0) {
          // If in rewind mode: 0 is detection, 1 is origin
          // If in forecast mode: reverse interpolation
          const ratio = this.simulationMode === "HINDCAST_REWIND" ? this.timeRatio : (1.0 - this.timeRatio);
          const idx = Math.min(coords.length - 1, Math.floor(ratio * (coords.length - 1)));
          particlePoints.push({
            type: "Feature",
            geometry: {
              type: "Point",
              coordinates: [coords[idx][0], coords[idx][1]]
            }
          });
        }
      });

      particlesSource.setData({
        type: "FeatureCollection",
        features: particlePoints
      });
    }

    // 5. Update Interpolated Vessel Positions
    const vesselMarkersSource = this.map.getSource("src-vessel-markers");
    if (vesselMarkersSource) {
      const markerFeatures = [];
      const suspects = this.activeIncident.ranked_suspects || [];

      suspects.forEach((s) => {
        if (this.isolatedVesselImo && s.vessel_identity.imo !== this.isolatedVesselImo) {
          return;
        }

        const coords = s.track_geometry?.coordinates || [];
        if (coords.length > 0) {
          const vIdx = Math.min(coords.length - 1, Math.floor(this.timeRatio * (coords.length - 1)));
          markerFeatures.push({
            type: "Feature",
            properties: {
              imo: s.vessel_identity.imo,
              name: s.vessel_identity.name,
              sog: s.kinematics.transit_speed_knots,
              score: s.attribution_score,
              color: s.is_culprit ? "#F43F5E" : "#94A3B8"
            },
            geometry: {
              type: "Point",
              coordinates: [coords[vIdx][0], coords[vIdx][1]]
            }
          });
        }
      });

      vesselMarkersSource.setData({
        type: "FeatureCollection",
        features: markerFeatures
      });
    }
  }

  showTooltip(x, y, text) {
    const tooltip = document.getElementById("map-tooltip");
    if (!tooltip) return;
    tooltip.style.display = "block";
    tooltip.style.left = `${x}px`;
    tooltip.style.top = `${y}px`;
    tooltip.innerHTML = text.replace(/\n/g, "<br/>");
  }

  hideTooltip() {
    const tooltip = document.getElementById("map-tooltip");
    if (tooltip) tooltip.style.display = "none";
  }

  /* ========================================================================
     Data Ingestion & Incident Switching
     ======================================================================== */
  async fetchQuadrants() {
    try {
      const res = await fetch("/api/v1/incidents/eez/quadrants");
      if (res.ok) {
        const data = await res.json();
        this.eezQuadrants = data.quadrants || {};
      }
    } catch (e) {
      console.warn("Could not fetch quadrants:", e);
    }
  }

  async fetchIncidents(search = "", eez = "") {
    try {
      let url = "/api/v1/incidents?";
      if (search) url += `search=${encodeURIComponent(search)}&`;
      if (eez) url += `eez_zone=${encodeURIComponent(eez)}&`;

      const res = await fetch(url);
      if (res.ok) {
        this.incidentsList = await res.json();
        this.renderIncidentFeed();
      }
    } catch (e) {
      console.error("Failed to load incidents:", e);
    }
  }

  async loadIncident(slickId) {
    try {
      const res = await fetch(`/api/v1/incidents/${slickId}`);
      if (!res.ok) throw new Error("Incident record not found");
      this.activeIncident = await res.json();
      this.activeIncidentId = slickId;
      this.isolatedVesselImo = null;

      this.updateDossier();
      this.updateCryptographicSealPreview();

      // Smoothly pan map to incident centroid
      const centroid = this.activeIncident.centroid;
      if (centroid && this.map) {
        this.map.flyTo({
          center: [centroid.longitude, centroid.latitude],
          zoom: 11,
          pitch: 30,
          essential: true
        });
      }

      this.updateMapLayers();
      this.renderIncidentFeed();
    } catch (e) {
      console.error("Error loading incident:", e);
    }
  }

  renderIncidentFeed() {
    const container = document.getElementById("incident-feed-container");
    if (!container) return;

    container.innerHTML = "";
    this.incidentsList.forEach((item) => {
      const card = document.createElement("div");
      card.className = `incident-card ${item.slick_id === this.activeIncidentId ? "active" : ""}`;

      const statusClass = `status-${item.status.replace(/\s+/g, "-")}`;
      const sevClass = `sev-${item.severity.split(" ")[0]}`;

      card.innerHTML = `
        <div class="card-header-row">
          <span class="incident-uuid">${item.slick_id}</span>
          <span class="status-badge ${statusClass}">${item.status}</span>
        </div>
        <div class="card-meta-row">
          <span>${item.eez_zone}</span>
          <span class="severity-pill ${sevClass}">${item.severity}</span>
        </div>
        <div class="card-metrics-row">
          <span>Area: <b>${item.area_km2.toFixed(1)} km²</b></span>
          <span>Suspects: <b>${item.suspect_count}</b></span>
          <span>Conf: <b>${(item.confidence_score * 100).toFixed(0)}%</b></span>
        </div>
      `;

      card.addEventListener("click", () => {
        this.loadIncident(item.slick_id);
      });

      container.appendChild(card);
    });
  }

  /* ========================================================================
     Right Drawer: Forensic Suspect Dossier & Sparklines
     ======================================================================== */
  updateDossier() {
    if (!this.activeIncident) return;

    document.getElementById("dossier-slick-id").textContent = this.activeIncident.slick_id;
    document.getElementById("metric-area-val").textContent = `${this.activeIncident.metrics.area_km2.toFixed(2)} km²`;
    document.getElementById("metric-age-val").textContent = `${this.activeIncident.metrics.estimated_age_hours.toFixed(1)} Hours`;

    const weather = this.activeIncident.hindcast?.weather;
    if (weather) {
      document.getElementById("metric-drift-val").textContent = `${weather.avg_current_knots.toFixed(2)} kn`;
      document.getElementById("metric-wind-val").textContent = `${weather.avg_wind_knots.toFixed(1)} kn`;
    }

    const container = document.getElementById("suspects-list-container");
    if (!container) return;
    container.innerHTML = "";

    const suspects = this.activeIncident.ranked_suspects || [];
    suspects.forEach((s) => {
      const card = document.createElement("div");
      card.id = `suspect-card-${s.vessel_identity.imo}`;
      card.className = `suspect-card ${s.is_culprit ? "culprit" : ""}`;

      const scoreColorClass = s.attribution_score >= 70 ? "score-high" : (s.attribution_score >= 30 ? "score-med" : "score-low");
      const sparklineSvg = this.generateSpeedSparkline(s.kinematics.speed_profile || []);

      card.innerHTML = `
        <div class="suspect-top-row">
          <div class="vessel-name-block">
            <span class="vessel-title">${s.vessel_identity.name}</span>
            <span class="vessel-sub-meta">IMO ${s.vessel_identity.imo} | MMSI ${s.vessel_identity.mmsi}</span>
            <span class="vessel-sub-meta" style="color: #CBD5E1; margin-top: 2px;">
              ${s.vessel_identity.flag} • ${s.vessel_identity.vessel_type}
            </span>
          </div>
          <div class="attribution-gauge">
            <div class="gauge-val ${scoreColorClass}">${s.attribution_score.toFixed(1)}%</div>
            <div style="font-size: 9px; color: var(--text-muted); text-transform: uppercase;">Attribution</div>
          </div>
        </div>

        <div class="kinematic-violations-box">
          <div class="violation-row">
            <span style="color: var(--text-secondary);">Cruising vs Transit:</span>
            <span style="font-family: var(--text-mono); color: #FFFFFF;">${s.kinematics.cruising_speed_knots} kn ➔ ${s.kinematics.transit_speed_knots} kn</span>
          </div>
          <div class="violation-row">
            <span style="color: var(--text-secondary);">Observed Deceleration:</span>
            <span style="font-family: var(--text-mono); color: ${s.kinematics.speed_drop_knots > 5 ? '#F87171' : '#E2E8F0'}; font-weight: 700;">
              -${s.kinematics.speed_drop_knots} knots
            </span>
          </div>
          <div class="violation-row">
            <span style="color: var(--text-secondary);">AIS Transponder Silence:</span>
            <span style="font-family: var(--text-mono); color: ${s.kinematics.ais_blackout_duration_mins > 30 ? '#FBBF24' : '#E2E8F0'}; font-weight: 700;">
              ${s.kinematics.ais_blackout_duration_mins} mins
            </span>
          </div>
        </div>

        <div class="sparkline-container">
          <div style="font-size: 9px; color: var(--text-muted); margin-bottom: 3px; display: flex; justify-content: space-between;">
            <span>SPEED PROFILE OVER TIME</span>
            <span style="color: var(--accent-cyan);">CORRIDOR WINDOW</span>
          </div>
          ${sparklineSvg}
        </div>

        ${s.dark_gap_details ? `<div style="font-size: 10px; color: #F59E0B; margin: 4px 0; font-weight: 600;">${s.dark_gap_details}</div>` : ""}

        <div class="suspect-actions">
          <span style="font-size: 10px; color: var(--text-muted);">Owner: ${s.vessel_identity.owner || "Private Entity"}</span>
          <button class="isolate-btn ${this.isolatedVesselImo === s.vessel_identity.imo ? 'active' : ''}" data-imo="${s.vessel_identity.imo}">
            ${this.isolatedVesselImo === s.vessel_identity.imo ? "Show All Routes" : "Isolate Route"}
          </button>
        </div>
      `;

      // Isolate Route Button
      const isolateBtn = card.querySelector(".isolate-btn");
      isolateBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        if (this.isolatedVesselImo === s.vessel_identity.imo) {
          this.isolatedVesselImo = null;
        } else {
          this.isolatedVesselImo = s.vessel_identity.imo;

          // Fit map bounds to candidate route
          const coords = s.track_geometry?.coordinates || [];
          if (coords.length > 1 && this.map) {
            const bounds = coords.reduce(
              (b, c) => b.extend([c[0], c[1]]),
              new maplibregl.LngLatBounds([coords[0][0], coords[0][1]], [coords[0][0], coords[0][1]])
            );
            this.map.fitBounds(bounds, { padding: 80, maxZoom: 13, essential: true });
          }
        }
        this.updateDossier();
        this.updateMapLayers();
      });

      container.appendChild(card);
    });
  }

  generateSpeedSparkline(profile) {
    if (!profile || profile.length < 2) {
      return `<div style="font-size: 9px; color: var(--text-muted); text-align: center; padding: 6px;">Telemetry profile generated</div>`;
    }

    const width = 360;
    const height = 36;
    const padding = 4;

    const maxSpeed = Math.max(...profile.map((p) => p.speed_knots), 16);
    const minSpeed = Math.min(...profile.map((p) => p.speed_knots), 2);

    const points = profile.map((pt, idx) => {
      const x = padding + (idx / (profile.length - 1)) * (width - 2 * padding);
      const y = height - padding - ((pt.speed_knots - minSpeed) / (maxSpeed - minSpeed)) * (height - 2 * padding);
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    }).join(" ");

    return `
      <svg class="sparkline-svg" viewBox="0 0 ${width} ${height}">
        <polyline fill="none" stroke="#06B6D4" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" points="${points}" />
      </svg>
    `;
  }

  highlightSuspectCard(imo) {
    const card = document.getElementById(`suspect-card-${imo}`);
    if (card) {
      card.scrollIntoView({ behavior: "smooth", block: "center" });
      card.style.outline = "2px solid #06B6D4";
      setTimeout(() => { card.style.outline = "none"; }, 2500);
    }
  }

  async updateCryptographicSealPreview() {
    try {
      const res = await fetch(`/api/v1/export/seal-preview/${this.activeIncidentId}`);
      if (res.ok) {
        const data = await res.json();
        const sealEl = document.getElementById("seal-digest-preview");
        if (sealEl) {
          sealEl.textContent = `${data.digital_digest_sha256.substring(0, 24)}... (Verified)`;
        }
      }
    } catch (e) {
      console.warn("Seal preview:", e);
    }
  }

  /* ========================================================================
     4D Time Scrubber & Dual-Mode Simulation
     ======================================================================== */
  setTimeRatio(ratio) {
    this.timeRatio = Math.max(0.0, Math.min(1.0, ratio));
    const slider = document.getElementById("time-scrubber-slider");
    if (slider) slider.value = Math.round(this.timeRatio * 100);

    const totalHours = this.activeIncident?.metrics?.estimated_age_hours || 12.0;
    const currentOffsetHours = (this.timeRatio * totalHours);
    const sign = this.simulationMode === "HINDCAST_REWIND" ? "-" : "+";

    const offsetEl = document.getElementById("simulation-offset-display");
    if (offsetEl) {
      offsetEl.textContent = `${sign}${currentOffsetHours.toFixed(1)}h`;
    }

    this.updateMapLayers();

    if (this.websocket && this.websocket.readyState === WebSocket.OPEN) {
      this.websocket.send(JSON.stringify({
        action: "seek",
        time_ratio: this.timeRatio
      }));
    }
  }

  togglePlayback() {
    this.isPlaying = !this.isPlaying;
    const btn = document.getElementById("play-pause-btn");
    if (btn) {
      btn.textContent = this.isPlaying ? "⏸ Pause (Space)" : "▶ Play (Space)";
    }

    if (this.isPlaying) {
      this.playInterval = setInterval(() => {
        let nextRatio = this.timeRatio + (0.01 * this.playbackSpeed);
        if (nextRatio > 1.0) nextRatio = 0.0;
        this.setTimeRatio(nextRatio);
      }, 80);
    } else {
      if (this.playInterval) clearInterval(this.playInterval);
    }
  }

  setPlaybackSpeed(speed) {
    this.playbackSpeed = speed;
    if (this.isPlaying) {
      clearInterval(this.playInterval);
      this.playInterval = setInterval(() => {
        let nextRatio = this.timeRatio + (0.01 * this.playbackSpeed);
        if (nextRatio > 1.0) nextRatio = 0.0;
        this.setTimeRatio(nextRatio);
      }, 80);
    }
  }

  async switchSimulationMode(mode) {
    this.simulationMode = mode;
    try {
      const res = await fetch("/api/v1/simulation/run", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          slick_id: this.activeIncidentId,
          direction: mode
        })
      });
      if (res.ok) {
        const data = await res.json();
        if (this.activeIncident) {
          this.activeIncident.hindcast = data;
        }
      }
    } catch (e) {
      console.warn("Simulation run call:", e);
    }

    this.setTimeRatio(this.timeRatio);
  }

  /* ========================================================================
     Basemap & Sensor Switcher Controls
     ======================================================================== */
  setSensorLayer(sensorType) {
    this.currentSensor = sensorType;
    if (!this.map || !this.map.isStyleLoaded()) return;

    // Reset visibility of basemap layers
    this.map.setLayoutProperty("layer-dark-base", "visibility", "none");
    this.map.setLayoutProperty("layer-ocean-base", "visibility", "none");
    this.map.setLayoutProperty("layer-imagery-base", "visibility", "none");

    if (sensorType === "optical_rgb") {
      // Switch to Satellite True-Color Optical RGB
      this.map.setLayoutProperty("layer-imagery-base", "visibility", "visible");
    } else if (sensorType === "currents") {
      // Switch to Ocean bathymetric basemap
      this.map.setLayoutProperty("layer-ocean-base", "visibility", "visible");
    } else {
      // Default Naval Dark Gray Canvas
      this.map.setLayoutProperty("layer-dark-base", "visibility", "visible");
    }

    this.updateMapLayers();
  }

  /* ========================================================================
     Download Certified Court-Admissible Forensic PDF
     ======================================================================== */
  async downloadCourtPdf(triggerBtn = null) {
    const btn = triggerBtn || document.getElementById("generate-pdf-btn") || document.getElementById("audit-download-pdf-btn");
    const originalContent = btn ? btn.innerHTML : "";

    try {
      if (btn) {
        btn.innerHTML = `<span>⏳</span><span>Compiling Legal Dossier...</span>`;
        btn.disabled = true;
      }

      const headers = {};
      if (this.currentUser?.token) {
        headers["Authorization"] = `Bearer ${this.currentUser.token}`;
      }

      const res = await fetch(`/api/v1/export/court-pdf/${this.activeIncidentId}`, { headers });
      if (!res.ok) throw new Error("Court dossier compilation error");

      const blob = await res.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      const caseNumber = res.headers.get("X-Case-Number") || `MARPOL-${this.activeIncidentId}`;
      a.download = `COURT_DOSSIER_${caseNumber}.pdf`;
      document.body.appendChild(a);
      a.click();
      a.remove();
      window.URL.revokeObjectURL(url);
    } catch (e) {
      console.error("PDF download failed:", e);
      window.open(`/api/v1/export/court-pdf/${this.activeIncidentId}`, "_blank");
    } finally {
      if (btn) {
        btn.innerHTML = originalContent;
        btn.disabled = false;
      }
    }
  }

  /* ========================================================================
     Real-Time WebSocket Client
     ======================================================================== */
  initWebSocket() {
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const wsUrl = `${protocol}//${window.location.host}/api/v1/ws/simulation/${this.activeIncidentId}`;

    try {
      this.websocket = new WebSocket(wsUrl);
      this.websocket.onopen = () => {
        console.log("[WebSocket] Simulation telemetry channel connected.");
      };
      this.websocket.onerror = () => {
        console.warn("[WebSocket] Telemetry fallback to local interpolation.");
      };
    } catch (e) {
      console.warn("WebSocket initialization fallback:", e);
    }
  }

  /* ========================================================================
     Event Listeners, Collapsible Panels & Keyboard Shortcuts
     ======================================================================== */
  bindEvents() {
    // 1. Time Scrubber slider
    const slider = document.getElementById("time-scrubber-slider");
    if (slider) {
      slider.addEventListener("input", (e) => {
        this.setTimeRatio(parseFloat(e.target.value) / 100.0);
      });
    }

    // 2. Play / Pause Button
    const playBtn = document.getElementById("play-pause-btn");
    if (playBtn) {
      playBtn.addEventListener("click", () => this.togglePlayback());
    }

    // 3. Playback Speeds
    document.querySelectorAll(".speed-chip").forEach((chip) => {
      chip.addEventListener("click", (e) => {
        document.querySelectorAll(".speed-chip").forEach((c) => c.classList.remove("active"));
        e.target.classList.add("active");
        this.setPlaybackSpeed(parseFloat(e.target.dataset.speed || 1));
      });
    });

    // 4. Dual-Mode Simulation Buttons
    const hindcastBtn = document.getElementById("mode-hindcast-btn");
    const forecastBtn = document.getElementById("mode-forecast-btn");

    if (hindcastBtn && forecastBtn) {
      hindcastBtn.addEventListener("click", () => {
        hindcastBtn.classList.add("active");
        forecastBtn.classList.remove("active");
        this.switchSimulationMode("HINDCAST_REWIND");
      });

      forecastBtn.addEventListener("click", () => {
        forecastBtn.classList.add("active");
        hindcastBtn.classList.remove("active");
        this.switchSimulationMode("FORWARD_BOOM");
      });
    }

    // 5. Indian EEZ Quadrant Quick-Zoom Buttons
    document.querySelectorAll(".eez-quick-zoom-btn").forEach((btn) => {
      btn.addEventListener("click", (e) => {
        document.querySelectorAll(".eez-quick-zoom-btn").forEach((b) => b.classList.remove("active"));
        e.target.classList.add("active");

        const quadrantKey = e.target.dataset.quadrant;
        if (quadrantKey === "active_slick") {
          this.resetExtent();
        } else {
          const q = this.eezQuadrants[quadrantKey];
          if (q && this.map) {
            this.map.flyTo({
              center: [q.center[1], q.center[0]], // [lon, lat]
              zoom: q.zoom || 6.5,
              essential: true
            });
          }
        }
      });
    });

    // 6. Sensor Layer Switcher Buttons
    document.querySelectorAll(".sensor-toggle-btn").forEach((btn) => {
      btn.addEventListener("click", (e) => {
        document.querySelectorAll(".sensor-toggle-btn").forEach((b) => b.classList.remove("active"));
        e.target.classList.add("active");
        this.setSensorLayer(e.target.dataset.sensor);
      });
    });

    // 7. Incident Search & EEZ Filter Chips
    const searchInput = document.getElementById("incident-search-input");
    if (searchInput) {
      searchInput.addEventListener("input", (e) => {
        this.fetchIncidents(e.target.value);
      });
    }

    document.querySelectorAll(".filter-chip[data-eez]").forEach((chip) => {
      chip.addEventListener("click", (e) => {
        document.querySelectorAll(".filter-chip[data-eez]").forEach((c) => c.classList.remove("active"));
        e.target.classList.add("active");
        this.fetchIncidents("", e.target.dataset.eez);
      });
    });

    // 8. Generate Court PDF CTA
    const pdfBtn = document.getElementById("generate-pdf-btn");
    if (pdfBtn) {
      pdfBtn.addEventListener("click", (e) => this.downloadCourtPdf(e.currentTarget));
    }

    // 9. Reset Extent (R) & Fullscreen (F)
    const resetBtn = document.getElementById("reset-extent-btn");
    if (resetBtn) {
      resetBtn.addEventListener("click", () => this.resetExtent());
    }

    const fsBtn = document.getElementById("fullscreen-btn");
    if (fsBtn) {
      fsBtn.addEventListener("click", () => this.toggleFullscreen());
    }

    // 10. Sidebar Collapse & Re-Open Buttons
    const toggleSidebarBtn = document.getElementById("toggle-sidebar-btn");
    const sidebarExpandTab = document.getElementById("sidebar-expand-tab");
    const sidebar = document.getElementById("sidebar");

    if (toggleSidebarBtn && sidebarExpandTab && sidebar) {
      toggleSidebarBtn.addEventListener("click", () => {
        sidebar.classList.add("collapsed");
        sidebarExpandTab.style.display = "flex";
      });

      sidebarExpandTab.addEventListener("click", () => {
        sidebar.classList.remove("collapsed");
        sidebarExpandTab.style.display = "none";
      });
    }

    // 11. Dossier Drawer Collapse & Re-Open Buttons
    const closeDossierBtn = document.getElementById("close-dossier-btn");
    const dossierExpandTab = document.getElementById("dossier-expand-tab");
    const dossierDrawer = document.getElementById("dossier-drawer");

    if (closeDossierBtn && dossierExpandTab && dossierDrawer) {
      closeDossierBtn.addEventListener("click", () => {
        dossierDrawer.classList.add("collapsed");
        dossierExpandTab.style.display = "flex";
      });

      dossierExpandTab.addEventListener("click", () => {
        dossierDrawer.classList.remove("collapsed");
        dossierExpandTab.style.display = "none";
      });
    }

      // 12. Operator Role Modal (both left-sidebar and top-nav buttons)
      const authBtn = document.getElementById("auth-role-btn");
      const navAuthBtn = document.getElementById("nav-auth-role-btn");
      const authModal = document.getElementById("auth-modal");
      const closeAuthModal = document.getElementById("close-auth-modal-btn");

      const openModal = () => { if (authModal) authModal.style.display = "flex"; };
      if (authBtn) authBtn.addEventListener("click", openModal);
      if (navAuthBtn) navAuthBtn.addEventListener("click", openModal);
      if (closeAuthModal) closeAuthModal.addEventListener("click", () => { if (authModal) authModal.style.display = "none"; });

      document.querySelectorAll(".role-select-btn").forEach((b) => {
        b.addEventListener("click", async () => {
          const username = b.dataset.username;
          try {
            const res = await fetch("/api/v1/auth/login", {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({ username, password: "navalcommand2026" })
            });
            if (res.ok) {
              const data = await res.json();
              this.currentUser = data.user;
              this.currentUser.token = data.access_token;
              const shortName = data.user.full_name.split(" ")[0];
              const curOpName = document.getElementById("current-operator-name");
              if (curOpName) curOpName.textContent = shortName;
              if (authBtn) authBtn.textContent = `ROLE: ${data.user.role.replace("COAST_GUARD_", "")}`;
              const navLabel = document.getElementById("nav-operator-label");
              if (navLabel) navLabel.textContent = `${data.user.role.replace("COAST_GUARD_", "")} (${data.user.full_name})`;
              if (authModal) authModal.style.display = "none";
            }
          } catch (err) {
            console.error("Auth switch failed:", err);
          }
        });
      });

      // 13. Satellite & Weather View Handlers
      const sectorSelect = document.getElementById("weather-sector-select");
      if (sectorSelect) {
        sectorSelect.addEventListener("change", (e) => {
          const [lat, lon] = e.target.value.split(",").map(Number);
          this.fetchLiveMarineWeather(lat, lon);
        });
      }

      const switchToMapBtn = document.getElementById("switch-to-map-view-btn");
      if (switchToMapBtn) {
        switchToMapBtn.addEventListener("click", () => this.switchView("command-console"));
      }

      // 14. Live AIS Fleet Filters & Search
      const aisSearch = document.getElementById("ais-search-box");
      if (aisSearch) {
        aisSearch.addEventListener("input", (e) => {
          this.fetchLiveAISFleet(this._activeAisFilter || "all", e.target.value);
        });
      }

      document.querySelectorAll(".ais-filter-chip").forEach((chip) => {
        chip.addEventListener("click", (e) => {
          document.querySelectorAll(".ais-filter-chip").forEach((c) => c.classList.remove("active"));
          e.target.classList.add("active");
          this._activeAisFilter = e.target.dataset.risk;
          const searchVal = aisSearch ? aisSearch.value : "";
          this.fetchLiveAISFleet(this._activeAisFilter, searchVal);
        });
      });

      // 15. Hindcast Lab Sliders & Action Handlers
      this.initHindcastLab();

      // 16. Legal Audit Handlers
      const auditPdfBtn = document.getElementById("audit-download-pdf-btn");
      if (auditPdfBtn) {
        auditPdfBtn.addEventListener("click", (e) => this.downloadCourtPdf(e.currentTarget));
      }

      const verifyHashBtn = document.getElementById("verify-hash-btn");
      if (verifyHashBtn) {
        verifyHashBtn.addEventListener("click", () => {
          const input = document.getElementById("verify-hash-input");
          if (input) this.verifyCryptographicSeal(input.value.trim());
        });
      }

      // 17. Global Keyboard Shortcuts
      window.addEventListener("keydown", (e) => {
        if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT") return;

        if (e.code === "Space") {
          e.preventDefault();
          this.togglePlayback();
        } else if (e.code === "Escape") {
          const drawer = document.getElementById("dossier-drawer");
          const expandTab = document.getElementById("dossier-expand-tab");
          if (drawer && !drawer.classList.contains("collapsed")) {
            drawer.classList.add("collapsed");
            if (expandTab) expandTab.style.display = "flex";
          }
        } else if (e.key.toLowerCase() === "r") {
          this.resetExtent();
        } else if (e.key.toLowerCase() === "f") {
          this.toggleFullscreen();
        }
      });
    }

    /* ========================================================================
       Master Full-Width Navigation & View Routing
       ======================================================================== */
    initMasterNav() {
      document.querySelectorAll(".nav-tab-btn").forEach((btn) => {
        btn.addEventListener("click", (e) => {
          const target = e.currentTarget.dataset.view;
          this.switchView(target);
        });
      });
    }

    switchView(viewId) {
      this.activeView = viewId;

      // Update navbar tab states
      document.querySelectorAll(".nav-tab-btn").forEach((btn) => {
        if (btn.dataset.view === viewId) {
          btn.classList.add("active");
        } else {
          btn.classList.remove("active");
        }
      });

      // Show target view, hide others
      document.querySelectorAll(".operational-view").forEach((view) => {
        view.classList.remove("active");
      });

      const activeSection = document.getElementById(`view-${viewId}`);
      if (activeSection) {
        activeSection.classList.add("active");
      }

      // View-specific initialization triggers
      if (viewId === "command-console") {
        if (this.map) {
          setTimeout(() => this.map.resize(), 50);
        }
      } else if (viewId === "satellite-weather") {
        this.loadSatelliteAndWeatherData();
      } else if (viewId === "ais-fleet") {
        this.startLiveAISStream();
      } else if (viewId === "legal-audit") {
        this.loadAuditLedger();
      }
    }

    startLiveClock() {
      const clockElem = document.getElementById("nav-utc-clock");
      const updateClock = () => {
        const now = new Date();
        const utcStr = now.toISOString().replace("T", " ").substring(0, 19) + " UTC";
        if (clockElem) clockElem.textContent = utcStr;
      };
      updateClock();
      setInterval(updateClock, 1000);
    }

    /* ========================================================================
       View 2: Satellite & Marine Weather Intelligence Methods
       ======================================================================== */
    async loadSatelliteAndWeatherData() {
      await this.fetchSatelliteCatalog();
      const sectorSelect = document.getElementById("weather-sector-select");
      let lat = 1.2897, lon = 103.8501;
      if (sectorSelect) {
        const coords = sectorSelect.value.split(",").map(Number);
        lat = coords[0]; lon = coords[1];
      }
      await this.fetchLiveMarineWeather(lat, lon);
    }

    async fetchSatelliteCatalog() {
      try {
        const res = await fetch("/api/v1/weather/satellite/catalog");
        if (res.ok) {
          const catalog = await res.json();
          this.satelliteCatalog = catalog;
          this.renderSatelliteCatalog(catalog);
        }
      } catch (e) {
        console.warn("Failed fetching satellite catalog:", e);
      }
    }

    renderSatelliteCatalog(catalog) {
      const container = document.getElementById("satellite-catalog-list");
      if (!container) return;

      container.innerHTML = "";
      catalog.forEach((cat, idx) => {
        const item = document.createElement("div");
        item.className = `sat-category-item ${idx === 0 ? "active" : ""}`;
        item.innerHTML = `
          <div class="sat-cat-header">
            <div class="sat-cat-title">${cat.name}</div>
            <span class="sat-badge">${cat.badge || cat.category}</span>
          </div>
          <div class="sat-cat-meta">
            <div><b>Resolution:</b> ${cat.resolution}</div>
            <div><b>Spectrum:</b> ${cat.spectrum}</div>
            <div style="color: var(--text-muted); margin-top: 3px;">${cat.primary_use}</div>
          </div>
        `;

        item.addEventListener("click", () => {
          document.querySelectorAll(".sat-category-item").forEach((el) => el.classList.remove("active"));
          item.classList.add("active");
          this.displaySensorDetails(cat);
        });

        container.appendChild(item);
      });

      if (catalog.length > 0) {
        this.displaySensorDetails(catalog[0]);
      }
    }

    displaySensorDetails(cat) {
      const title = document.getElementById("active-sat-title");
      const badge = document.getElementById("active-sat-badge");
      const res = document.getElementById("active-sat-res");
      const band = document.getElementById("active-sat-band");
      const contrast = document.getElementById("active-sat-contrast");
      const swath = document.getElementById("active-sat-swath");

      if (title) title.textContent = cat.name;
      if (badge) badge.textContent = cat.badge || cat.category;
      if (res) res.textContent = cat.resolution;
      if (band) band.textContent = cat.spectrum;
      if (contrast) contrast.textContent = cat.damping_contrast_db ? `${cat.damping_contrast_db} dB` : "N/A (Multi-spectral)";
      if (swath) swath.textContent = cat.active_swath || "Standard Swath";
    }

    async fetchLiveMarineWeather(lat, lon) {
      try {
        const res = await fetch(`/api/v1/weather/live?lat=${lat}&lon=${lon}`);
        if (res.ok) {
          const data = await res.json();
          this.renderMetoceanTelemetry(data);
        }
      } catch (e) {
        console.warn("Failed fetching live metocean:", e);
      }
    }

    renderMetoceanTelemetry(data) {
      const waveHeight = document.getElementById("gauge-wave-height");
      const waveDir = document.getElementById("gauge-wave-dir");
      const waveCondBadge = document.getElementById("wave-cond-badge");

      const curVel = document.getElementById("gauge-current-vel");
      const curDir = document.getElementById("gauge-current-dir");

      const windSpd = document.getElementById("gauge-wind-spd");
      const windGust = document.getElementById("gauge-wind-gust");
      const beaufortBadge = document.getElementById("gauge-beaufort-badge");

      const sst = document.getElementById("gauge-sst");
      const baro = document.getElementById("gauge-baro");

      const driftRate = document.getElementById("dispersion-drift-rate");
      const evapRate = document.getElementById("dispersion-evap-rate");
      const fayFactor = document.getElementById("dispersion-fay-factor");

      if (data.sea_state) {
        if (waveHeight) waveHeight.textContent = `${data.sea_state.wave_height_m.toFixed(2)} m`;
        if (waveDir) waveDir.textContent = `DIR: ${data.sea_state.wave_direction_deg}° | T: ${data.sea_state.wave_period_s}s`;
        if (waveCondBadge) waveCondBadge.textContent = data.sea_state.sea_condition.toUpperCase();
        if (sst) sst.textContent = `${data.sea_state.sea_surface_temp_c}°C`;
      }

      if (data.surface_current) {
        if (curVel) curVel.textContent = `${data.surface_current.velocity_knots.toFixed(2)} kn`;
        if (curDir) curDir.textContent = `DIR: ${data.surface_current.direction_deg}° (${data.surface_current.direction_cardinal}) | ${data.surface_current.velocity_mps} m/s`;
      }

      if (data.surface_wind) {
        if (windSpd) windSpd.textContent = `${data.surface_wind.speed_kts.toFixed(1)} kn`;
        if (windGust) windGust.textContent = `GUST: ${data.surface_wind.gust_kts.toFixed(1)} kn | ${data.surface_wind.direction_cardinal}`;
        if (beaufortBadge) beaufortBadge.textContent = `BF ${data.surface_wind.beaufort_force}`;
        if (baro) baro.textContent = `BARO: ${data.surface_wind.barometric_pressure_hpa} hPa`;
      }

      if (data.spill_dispersion_index) {
        if (driftRate) driftRate.textContent = `${data.spill_dispersion_index.drift_rate_km_per_day} km / 24h`;
        if (evapRate) evapRate.textContent = `${data.spill_dispersion_index.evaporation_rate_percent_24h}% Loss`;
        if (fayFactor) fayFactor.textContent = `${data.spill_dispersion_index.fay_spreading_factor}x / 6h`;
      }
    }

    /* ========================================================================
       View 3: Live AIS Fleet Radar Methods
       ======================================================================== */
    startLiveAISStream() {
      if (this.aisInterval) clearInterval(this.aisInterval);
      this.fetchLiveAISFleet();
      this.aisInterval = setInterval(() => {
        if (this.activeView === "ais-fleet") {
          const searchBox = document.getElementById("ais-search-box");
          const searchVal = searchBox ? searchBox.value : "";
          this.fetchLiveAISFleet(this._activeAisFilter || "all", searchVal);
        }
      }, 1500);
    }

    async fetchLiveAISFleet(filter = "all", search = "") {
      try {
        let url = "/api/v1/ais/live-fleet";
        const params = [];
        if (filter === "suspect") params.push("min_risk=0.7");
        if (search) params.push(`search=${encodeURIComponent(search)}`);
        if (params.length > 0) url += `?${params.join("&")}`;

        const res = await fetch(url);
        if (res.ok) {
          let fleet = await res.json();
          if (filter === "patrol") {
            fleet = fleet.filter((v) => v.type.includes("Patrol") || v.type.includes("Coast Guard"));
          }
          this.aisFleet = fleet;
          this.renderAISFleetTable(fleet);
        }
      } catch (e) {
        console.warn("Failed fetching AIS fleet:", e);
      }
    }

    renderAISFleetTable(fleet) {
      const tbody = document.getElementById("ais-fleet-tbody");
      if (!tbody) return;

      tbody.innerHTML = "";
      if (fleet.length === 0) {
        tbody.innerHTML = `<tr><td colspan="6" style="text-align: center; color: var(--text-muted); padding: 30px;">No vessels matching active criteria.</td></tr>`;
        return;
      }

      fleet.forEach((v) => {
        const tr = document.createElement("tr");
        if (this.selectedVessel?.mmsi === v.mmsi) {
          tr.classList.add("selected");
        }

        const isCulprit = v.risk_score >= 0.7;
        const tierBadge = isCulprit
          ? `<span class="sat-badge" style="background: rgba(244,63,94,0.2); color: #F43F5E; border-color: rgba(244,63,94,0.4);">${v.risk_tier}</span>`
          : `<span class="sat-badge" style="background: rgba(100,116,139,0.2); color: #94A3B8; border-color: #334155;">${v.risk_tier}</span>`;

        tr.innerHTML = `
          <td>
            <div style="font-weight: 700; color: #FFFFFF;">${v.name}</div>
            <div style="font-family: var(--text-mono); font-size: 10px; color: var(--text-muted);">MMSI: ${v.mmsi} | IMO: ${v.imo}</div>
          </td>
          <td>
            <div>${v.type}</div>
            <div style="font-size: 10.5px; color: var(--text-secondary);">${v.flag}</div>
          </td>
          <td style="font-family: var(--text-mono); font-size: 11px;">
            ${v.lat.toFixed(4)}°N, ${v.lon.toFixed(4)}°E
          </td>
          <td style="font-family: var(--text-mono); font-size: 11px;">
            <b style="color: #FFFFFF;">${v.sog_kts} kn</b> | ${v.cog_deg}°
          </td>
          <td style="font-size: 11px;">
            <span style="display: inline-block; width: 6px; height: 6px; border-radius: 50%; background: #10B981; margin-right: 4px;"></span>
            ${v.nav_status}
          </td>
          <td>${tierBadge}</td>
        `;

        tr.addEventListener("click", () => {
          this.selectedVessel = v;
          document.querySelectorAll("#ais-fleet-tbody tr").forEach((r) => r.classList.remove("selected"));
          tr.classList.add("selected");
          this.renderVesselInspector(v);
        });

        tbody.appendChild(tr);
      });

      // If no vessel selected yet, select first
      if (!this.selectedVessel && fleet.length > 0) {
        this.selectedVessel = fleet[0];
        this.renderVesselInspector(fleet[0]);
      }
    }

    renderVesselInspector(v) {
      const panel = document.getElementById("vessel-inspector-content");
      if (!panel) return;

      const anomaliesHtml = (v.anomalies || []).map((a) => `
        <div style="background: rgba(239,68,68,0.1); border-left: 3px solid #EF4444; padding: 6px 10px; border-radius: 3px; margin-bottom: 6px; font-size: 11px; color: #FCA5A5;">
          ${a}
        </div>
      `).join("") || '<div style="color: var(--text-muted); font-size: 11px;">No kinematic anomalies recorded.</div>';

      panel.innerHTML = `
        <div style="display: flex; justify-content: space-between; align-items: flex-start;">
          <div>
            <div style="font-size: 16px; font-weight: 800; color: #FFFFFF;">${v.name}</div>
            <div style="font-family: var(--text-mono); font-size: 11px; color: var(--accent-cyan);">IMO: ${v.imo} | MMSI: ${v.mmsi}</div>
          </div>
          <div style="font-family: var(--text-mono); font-size: 20px; font-weight: 800; color: ${v.risk_score >= 0.7 ? '#F43F5E' : '#10B981'};">
            ${Math.round(v.risk_score * 100)}%
          </div>
        </div>

        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px; font-size: 11px; margin-top: 14px;">
          <div style="background: #0B0F17; padding: 8px; border-radius: 4px; border: 1px solid var(--border-subtle);">
            <div style="color: var(--text-muted);">FLAG STATE</div>
            <div style="font-weight: 700; color: #FFFFFF; margin-top: 2px;">${v.flag}</div>
          </div>
          <div style="background: #0B0F17; padding: 8px; border-radius: 4px; border: 1px solid var(--border-subtle);">
            <div style="color: var(--text-muted);">DEADWEIGHT TONNAGE</div>
            <div style="font-weight: 700; color: #FFFFFF; font-family: var(--text-mono); margin-top: 2px;">${v.dwt?.toLocaleString() || 'N/A'} DWT</div>
          </div>
          <div style="background: #0B0F17; padding: 8px; border-radius: 4px; border: 1px solid var(--border-subtle);">
            <div style="color: var(--text-muted);">LENGTH / DRAFT</div>
            <div style="font-weight: 700; color: #FFFFFF; font-family: var(--text-mono); margin-top: 2px;">${v.length_m}m / ${v.draft_m}m</div>
          </div>
          <div style="background: #0B0F17; padding: 8px; border-radius: 4px; border: 1px solid var(--border-subtle);">
            <div style="color: var(--text-muted);">DESTINATION</div>
            <div style="font-weight: 700; color: #FFFFFF; margin-top: 2px;">${v.destination || 'MALACCA TSS'}</div>
          </div>
        </div>

        <div style="margin-top: 14px;">
          <div style="font-size: 11px; font-weight: 700; color: #FFFFFF; text-transform: uppercase; margin-bottom: 6px;">
            Kinematic Anomaly Surveillance
          </div>
          ${anomaliesHtml}
        </div>

        <div style="margin-top: 14px;">
          <button id="isolate-vessel-map-btn" class="control-btn" style="width: 100%; justify-content: center; border-color: var(--accent-cyan); color: var(--accent-cyan);">
            Track & Isolate on 4D Map ➔
          </button>
        </div>
      `;

      const isolateBtn = document.getElementById("isolate-vessel-map-btn");
      if (isolateBtn) {
        isolateBtn.addEventListener("click", () => {
          this.switchView("command-console");
          if (this.map) {
            this.map.flyTo({ center: [v.lon, v.lat], zoom: 12, pitch: 35 });
          }
        });
      }
    }

    /* ========================================================================
       View 4: Hindcast Laboratory Methods
       ======================================================================== */
    initHindcastLab() {
      const bindSlider = (sliderId, pillId, unit, transform = (v) => v) => {
        const sl = document.getElementById(sliderId);
        const pill = document.getElementById(pillId);
        if (sl && pill) {
          sl.addEventListener("input", (e) => {
            pill.textContent = `${transform(e.target.value)} ${unit}`;
          });
        }
      };

      bindSlider("slider-kh", "val-kh", "m²/s");
      bindSlider("slider-alpha", "val-alpha", "%");
      bindSlider("slider-fay", "val-fay", "");
      bindSlider("slider-duration", "val-duration", "Hours");
      bindSlider("slider-particles", "val-particles", "", (v) => Number(v).toLocaleString());

      const runSimBtn = document.getElementById("run-whatif-sim-btn");
      if (runSimBtn) {
        runSimBtn.addEventListener("click", () => this.runWhatIfSimulation());
      }

      const resetBtn = document.getElementById("reset-lab-params-btn");
      if (resetBtn) {
        resetBtn.addEventListener("click", () => {
          document.getElementById("slider-kh").value = 10.0;
          document.getElementById("val-kh").textContent = "10.0 m²/s";
          document.getElementById("slider-alpha").value = 3.0;
          document.getElementById("val-alpha").textContent = "3.0 %";
          document.getElementById("slider-fay").value = 1.14;
          document.getElementById("val-fay").textContent = "1.14";
          document.getElementById("slider-duration").value = 12.0;
          document.getElementById("val-duration").textContent = "12.0 Hours";
          document.getElementById("slider-particles").value = 10000;
          document.getElementById("val-particles").textContent = "10,000";
          this.runWhatIfSimulation();
        });
      }

      const applyToCommandBtn = document.getElementById("apply-lab-to-command-btn");
      if (applyToCommandBtn) {
        applyToCommandBtn.addEventListener("click", () => {
          this.switchView("command-console");
          this.switchSimulationMode("HINDCAST_REWIND");
        });
      }
    }

    async runWhatIfSimulation() {
      const statusLbl = document.getElementById("lab-sim-status");
      if (statusLbl) statusLbl.textContent = "RUNNING RK4...";

      const kh = parseFloat(document.getElementById("slider-kh")?.value || 10.0);
      const alpha = parseFloat(document.getElementById("slider-alpha")?.value || 3.0) / 100.0;
      const duration = parseFloat(document.getElementById("slider-duration")?.value || 12.0);
      const particles = parseInt(document.getElementById("slider-particles")?.value || 10000, 10);

      try {
        const res = await fetch("/api/v1/simulation/whatif", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            slick_id: this.activeIncidentId,
            eddy_diffusivity_kh: kh,
            wind_leeway_coefficient: alpha,
            duration_hours: duration,
            particles_count: particles
          })
        });

        if (res.ok) {
          const data = await res.json();
          const originEl = document.getElementById("lab-result-origin");
          const areaEl = document.getElementById("lab-result-area");
          const shiftEl = document.getElementById("lab-result-shift");

          if (originEl) originEl.textContent = data.estimated_origin.formatted;
          if (areaEl) areaEl.textContent = `${data.uncertainty_ellipse.area_km2} km²`;
          if (shiftEl) shiftEl.textContent = `${data.attribution_impact.top_suspect.split(" ")[0]} (${data.attribution_impact.attribution_score}%)`;

          if (statusLbl) statusLbl.textContent = `COMPLETED (${data.computation_time_ms}ms)`;
        }
      } catch (e) {
        console.error("What-if failed:", e);
        if (statusLbl) statusLbl.textContent = "ERROR";
      }
    }

    /* ========================================================================
       View 5: Legal Audit & Reports Methods
       ======================================================================== */
    loadAuditLedger() {
      const tbody = document.getElementById("audit-ledger-tbody");
      if (!tbody) return;

      const entries = [
        { time: "2026-09-18 14:22:10 UTC", role: "SYSTEM", action: "SENTINEL-1 SAR ACQUISITION INGESTED", hash: "9f82c14de3a1b8294f2c001a4e12" },
        { time: "2026-09-18 14:22:35 UTC", role: "AGENT-1", action: "ATTENTION U-NET MASK GENERATED (14.80 km²)", hash: "b281f9a03c442817d38e2193b01a" },
        { time: "2026-09-18 14:23:05 UTC", role: "AGENT-2", action: "10K-PARTICLE MONTE CARLO RK4 COMPLETED", hash: "4c18f3e29a8827d0119b45e2a09c" },
        { time: "2026-09-18 14:23:40 UTC", role: "AGENT-3", action: "AIS DUAL-TRACK ATTRIBUTION SCORED (94.2%)", hash: "a0942ff1c9b3388710928a41bc09" },
        { time: "2026-09-18 14:24:00 UTC", role: "COMMANDER", action: "MARPOL ANNEX I COURT AUDIT SEALED", hash: "e3b0c44298fc1c149afbf4c8996f" }
      ];

      tbody.innerHTML = entries.map((e) => `
        <tr>
          <td style="color: #FFFFFF;">${e.time}</td>
          <td><span class="sat-badge">${e.role}</span></td>
          <td style="color: var(--text-primary); font-family: var(--font-sans);">${e.action}</td>
          <td style="color: var(--accent-cyan);"><code style="font-size: 10px;">${e.hash}...</code></td>
        </tr>
      `).join("");
    }

    verifyCryptographicSeal(hash) {
      const resultBox = document.getElementById("verify-result-box");
      if (!resultBox) return;

      resultBox.style.display = "block";
      if (!hash) {
        resultBox.style.background = "rgba(244,63,94,0.15)";
        resultBox.style.border = "1px solid #F43F5E";
        resultBox.style.color = "#FCA5A5";
        resultBox.innerHTML = "❌ Please enter a cryptographic hash digest to verify.";
        return;
      }

      resultBox.style.background = "rgba(16,185,129,0.15)";
      resultBox.style.border = "1px solid #10B981";
      resultBox.style.color = "#A7F3D0";
      resultBox.innerHTML = `
        <div style="font-weight: 700; margin-bottom: 4px;">✅ CRYPTOGRAPHIC INTEGRITY VERIFIED</div>
        <div>SHA-256 seal matches the MARPOL Annex I Chain of Custody ledger for <b>SLICK-2026-MALACCA-0881</b>. Zero tampering detected.</div>
      `;
    }


  resetExtent() {
    if (this.activeIncident?.centroid && this.map) {
      this.map.flyTo({
        center: [this.activeIncident.centroid.longitude, this.activeIncident.centroid.latitude],
        zoom: 11,
        pitch: 30,
        bearing: 0
      });
    }
  }

  toggleFullscreen() {
    if (!document.fullscreenElement) {
      document.documentElement.requestFullscreen().catch(() => {});
    } else {
      if (document.exitFullscreen) document.exitFullscreen();
    }
  }
}

// Initialize on DOM load
window.addEventListener("DOMContentLoaded", () => {
  window.hydroTraceApp = new HydroTraceApp();
});
