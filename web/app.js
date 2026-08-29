/* OpenFloodHub — DC flood map. Reads preds.json (per-gauge local CNN forecast
 * + return-period thresholds + NOAA overlays) and renders a Flood-Hub-style
 * map + gauge detail panel. No backend, no build step. */

const RISK = {
  normal:  getCSS('--normal'),
  warning: getCSS('--warning'),
  danger:  getCSS('--danger'),
  extreme: getCSS('--extreme'),
  nodata:  getCSS('--nodata'),
};
const FLOW = getCSS('--flow');

function getCSS(v) { return getComputedStyle(document.documentElement).getPropertyValue(v).trim(); }

const map = L.map('map', { zoomControl: true }).setView([38.9072, -77.0369], 11);
// CARTO's keyless basemap now watermarks every tile with "API KEY REQUIRED";
// Esri's Light Gray Canvas is the same muted style and needs no key.
// Native tiles stop at z16 — Leaflet upscales beyond that.
L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}', {
  attribution: '&copy; Esri, HERE, Garmin &copy; OpenStreetMap contributors',
  maxNativeZoom: 16, maxZoom: 19,
}).addTo(map);
L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Reference/MapServer/tile/{z}/{y}/{x}', {
  maxNativeZoom: 16, maxZoom: 19,
}).addTo(map);

let SELECTED = null;
let SELECTED_P = null;   // the selected prediction record (drives the drawer)
const MARKERS = {};
let BASINS = {};        // usgs id -> {gauge_id, up_area_km2, geometry}
let BASIN_LAYER = null; // contributing-area polygon for the selected gauge

main();

async function main() {
  const [data, basins] = await Promise.all([
    fetch('preds.json', { cache: 'no-store' }).then(r => r.json()),
    fetch('basins.json').then(r => r.ok ? r.json() : {}).catch(() => ({})),
  ]);
  BASINS = basins;
  document.getElementById('updated').textContent =
    'Updated ' + fmtAge(data.updated);

  for (const p of data.predictions) {
    const r = riskOf(p);
    const m = L.circleMarker([p.lat, p.lon], markerStyle(r, false))
      .addTo(map)
      .on('click', () => select(p));
    m.bindTooltip(`${p.name}: ${r.label}`, { direction: 'top', offset: [0, -6] });
    MARKERS[p.id] = { marker: m, risk: r };
  }

  document.getElementById('panelClose').onclick = deselect;
  document.getElementById('compareClose').onclick = closeCompare;
  window.addEventListener('resize', () => {
    if (SELECTED_P && !document.getElementById('compare').hidden) renderCompare(SELECTED_P);
  });

  // Auto-open the marquee gauge so the panel isn't empty on load.
  const potomac = data.predictions.find(p => p.id === '01646500') || data.predictions[0];
  if (potomac) select(potomac);
}

/* ---- risk classification ------------------------------------------------ */

function currentFlow(p) {
  const obs = p.series.filter(e => 'o' in e);
  return obs.length ? obs[obs.length - 1].o : null;
}
function forecastPeak(p) {
  const fc = p.series.filter(e => 'p' in e).map(e => e.p);
  return fc.length ? Math.max(...fc) : null;
}
function riskOf(p) {
  if (!p.series.length) return { key: 'offline', label: 'Gauge offline', color: RISK.nodata };
  const th = p.thresholds;
  if (!th) return { key: 'nodata', label: 'No threshold', color: RISK.nodata };
  const peak = Math.max(currentFlow(p) ?? 0, forecastPeak(p) ?? 0);
  if (peak >= th.extreme) return { key: 'extreme', label: 'Extreme', color: RISK.extreme };
  if (peak >= th.danger)  return { key: 'danger',  label: 'Danger',  color: RISK.danger };
  if (peak >= th.warning) return { key: 'warning', label: 'Warning', color: RISK.warning };
  return { key: 'normal', label: 'Normal', color: RISK.normal };
}
function markerStyle(r, selected) {
  return {
    radius: selected ? 10 : 7, color: '#fff', weight: 2,
    fillColor: r.color, fillOpacity: 1,
  };
}

/* ---- selection ---------------------------------------------------------- */

function select(p) {
  if (SELECTED && MARKERS[SELECTED]) {
    MARKERS[SELECTED].marker.setStyle(markerStyle(MARKERS[SELECTED].risk, false));
  }
  SELECTED = p.id;
  SELECTED_P = p;
  MARKERS[p.id].marker.setStyle(markerStyle(MARKERS[p.id].risk, true));
  showBasin(p);
  renderPanel(p);
  document.getElementById('panel').classList.add('open');
  // Drawer follows the selection while it's open.
  if (!document.getElementById('compare').hidden) renderCompare(p);
}
function deselect() {
  if (SELECTED && MARKERS[SELECTED]) {
    MARKERS[SELECTED].marker.setStyle(markerStyle(MARKERS[SELECTED].risk, false));
  }
  SELECTED = null;
  SELECTED_P = null;
  hideBasin();
  closeCompare();
  document.getElementById('panel').classList.remove('open');
}

/* ---- HYBAS contributing area -------------------------------------------- */

function showBasin(p) {
  hideBasin();
  const b = BASINS[p.id];
  if (!b) return;
  BASIN_LAYER = L.geoJSON(b.geometry, {
    style: { color: '#00897b', weight: 1.5, fillColor: '#00897b',
             fillOpacity: 0.08, dashArray: '4 3' },
    interactive: false,
  }).addTo(map);
}
function hideBasin() {
  if (BASIN_LAYER) { map.removeLayer(BASIN_LAYER); BASIN_LAYER = null; }
}

/* ---- detail panel ------------------------------------------------------- */

function renderPanel(p) {
  const r = riskOf(p);
  const th = p.thresholds;
  const cur = currentFlow(p), peak = forecastPeak(p);
  const nse = p.metrics && p.metrics.nse_overall;

  const body = document.getElementById('panelBody');
  body.innerHTML = `
    <h2>${p.name}</h2>
    <div class="sub">${p.short}, USGS ${p.id}
      <span class="status-pill" style="background:${r.color};margin-left:6px">${r.label}</span></div>

    <div style="font-size:12px;color:var(--muted);margin-bottom:6px">Discharge (m³/s)</div>
    ${chartSVG(p)}
    <div class="chart-note">Solid line is observed flow, dashed is the CNN forecast, dotted is NOAA's NWM${p.google_flood && p.google_flood.forecast ? ', dash-dot is Google Flood Hub' : ''}.</div>

    ${th ? `
    <div class="legend-row">
      <div class="item"><span class="dot" style="background:var(--warning)"></span><span class="lbl">Warning</span><div class="val">${fmt(th.warning)}</div></div>
      <div class="item"><span class="dot" style="background:var(--danger)"></span><span class="lbl">Danger</span><div class="val">${fmt(th.danger)}</div></div>
      <div class="item"><span class="dot" style="background:var(--extreme)"></span><span class="lbl">Extreme</span><div class="val">${fmt(th.extreme)}</div></div>
    </div>` : `<div class="chart-note">No flood thresholds for this gauge.</div>`}

    <div class="section-title compare-toggle" id="cmpToggle">
      Compare model predictions — last 7 days <span style="float:right">▸</span></div>

    <div class="section-title">Now and forecast</div>
    <div class="info-grid">
      <div><div class="k">Current flow</div><div class="v">${cur != null ? fmt(cur) + ' m³/s' : '—'}</div></div>
      <div><div class="k">12h forecast peak</div><div class="v">${peak != null ? fmt(peak) + ' m³/s' : '—'}</div></div>
      <div><div class="k">Issued</div><div class="v">${p.issue_time ? fmtTime(p.issue_time) : '—'}</div></div>
      <div><div class="k">Forecast horizon</div><div class="v">12 hours</div></div>
    </div>

    <div class="section-title">Gauge and model</div>
    <div class="info-grid">
      <div><div class="k">Model</div><div class="v">1D CNN (per gauge)</div></div>
      <div><div class="k">Skill (overall NSE)</div><div class="v">${nse != null ? nse.toFixed(3) : '—'}</div></div>
      <div><div class="k">River gauge ID</div><div class="v">${p.id}</div></div>
      <div><div class="k">Source</div><div class="v">USGS NWIS</div></div>
      <div><div class="k">Lat / Long</div><div class="v">${p.lat.toFixed(4)}, ${p.lon.toFixed(4)}</div></div>
      <div><div class="k">Drainage</div><div class="v">${p.drainage_sqmi.toLocaleString()} mi²</div></div>
      <div><div class="k">Catchment</div><div class="v" style="text-transform:capitalize">${p.kind}</div></div>
      <div><div class="k">Record</div><div class="v">${th ? th.record_years + ' yr' : '—'}</div></div>
      ${googleRow(p)}
      ${hybasRow(p)}
    </div>

    ${sourcesBox()}
  `;
  document.getElementById('cmpToggle').onclick = () => openCompare(p);
}

function sourcesBox() {
  return `<div class="sources-box">
    <div class="title">Data sources &amp; credits</div>
    <div class="row">Streamflow — <a href="https://waterservices.usgs.gov/" target="_blank" rel="noopener">USGS NWIS</a></div>
    <div class="row">Weather forcings — <a href="https://open-meteo.com/" target="_blank" rel="noopener">Open-Meteo</a> (ERA5-Land)</div>
    <div class="row">NWM streamflow &amp; QPF — <a href="https://api.water.noaa.gov/nwps/v1/docs/" target="_blank" rel="noopener">NOAA/NWS</a> · MRMS via <a href="https://mesonet.agron.iastate.edu/" target="_blank" rel="noopener">IEM</a></div>
    <div class="row">Flood forecasts — <a href="https://developers.google.com/flood-forecasting" target="_blank" rel="noopener">Google Flood Forecasting API</a> (CC BY 4.0)</div>
    <div class="row">Basins — <a href="https://www.hydrosheds.org/products/hydrobasins" target="_blank" rel="noopener">HydroBASINS</a> © HydroSHEDS</div>
    <div class="row">Basemap — Esri World Light Gray Canvas</div>
    <div class="row">12h forecast — OpenFloodHub per-gauge CNN (<a href="https://github.com/abhiramm7/openfloodhub" target="_blank" rel="noopener">source</a>)</div>
  </div>`;
}

function hybasRow(p) {
  const b = BASINS[p.id];
  if (!b) return '';
  const sqmi = b.up_area_km2 / 2.58999;
  return `<div><div class="k">HYBAS basin (shown on map)</div>
    <div class="v">${sqmi >= 100 ? Math.round(sqmi).toLocaleString() : sqmi.toFixed(1)} mi²</div></div>`;
}

function googleRow(p) {
  const g = p.google_flood;
  if (!g || !g.severity) return '';
  const SEV = {
    NO_FLOODING:  ['No flooding', 'var(--normal)'],
    ABOVE_NORMAL: ['Above normal', 'var(--warning)'],
    SEVERE:       ['Severe', 'var(--danger)'],
    EXTREME:      ['Extreme', 'var(--extreme)'],
  };
  const [label, color] = SEV[g.severity] || ['Unknown', 'var(--nodata)'];
  const trend = { RISE: ' ↑', FALL: ' ↓', NO_CHANGE: ' →' }[g.trend] || '';
  return `<div><div class="k">Google Flood Hub</div>
    <div class="v" style="color:${color}">${label}${trend}</div></div>`;
}

/* ---- model comparison drawer (last 7 days, full width) ------------------ */

function compareData(p) {
  return {
    obs: (p.backtest || []).slice(-168)
      .filter(e => e.o != null).map(e => ({ t: +new Date(e.t), v: e.o })),
    cnn: (p.backtest || []).slice(-168)
      .filter(e => e.p12 != null).map(e => ({ t: +new Date(e.t), v: e.p12 })),
    goog: ((p.google_flood && p.google_flood.backtest) || [])
      .map(e => ({ t: +new Date(e.t), v: e.v })),
  };
}

function openCompare(p) {
  document.getElementById('compare').hidden = false;
  renderCompare(p);
}
function closeCompare() {
  document.getElementById('compare').hidden = true;
}
function renderCompare(p) {
  const { obs, cnn, goog } = compareData(p);
  document.getElementById('compareTitle').textContent =
    `${p.name} — model predictions vs observed, last 7 days`;
  const body = document.getElementById('compareBody');
  if (!obs.length && !goog.length) {
    body.innerHTML = `<div class="chart-note">No stored predictions for this gauge yet.</div>`;
    return;
  }
  // Daily observed rainfall (MRMS) drawn as bars hanging from the chart top.
  const rain = (p.noaa_mrms_precip || []).map(e => ({
    t0: +new Date(e.date + 'T00:00:00Z'),
    t1: +new Date(e.date + 'T00:00:00Z') + 86400e3,
    mm: e.precip_mm,
  }));
  const W = Math.max(360, body.clientWidth || (window.innerWidth - 40));
  body.innerHTML = `
    <div class="chart-note" style="margin:2px 0 8px">Each model's past predictions
      replayed against what the river actually did: the CNN's 12h-ahead forecast
      (dashed) and Google Flood Hub's next-day forecast (dash-dot points) vs observed
      flow (solid), m³/s. Bars from the top are observed daily rainfall (MRMS, mm).</div>
    ${compareSVG(obs, cnn, goog, rain, W, 220)}
    ${compareStats(obs, cnn, goog)}`;
}

function compareSVG(obs, cnn, goog, rain, W, H) {
  const M = { t: 12, r: 14, b: 20, l: 44 };
  const pw = W - M.l - M.r, ph = H - M.t - M.b;
  const all = [...obs, ...cnn, ...goog];
  const tMin = Math.min(...all.map(d => d.t)), tMax = Math.max(...all.map(d => d.t));
  const vals = all.map(d => d.v);
  let hi = Math.max(...vals), lo = Math.min(...vals);
  const span = (hi - lo) || hi || 1;
  const top = hi + span * 0.15, bot = Math.max(0, lo - span * 0.15);

  const x = t => M.l + ((t - tMin) / (tMax - tMin || 1)) * pw;
  const y = v => M.t + ph - ((v - bot) / (top - bot || 1)) * ph;
  const line = pts => pts.map((d, i) => (i ? 'L' : 'M') + x(d.t).toFixed(1) + ' ' + y(d.v).toFixed(1)).join(' ');

  let svg = `<svg width="${W}" height="${H}" viewBox="0 0 ${W} ${H}">`;
  for (let i = 0; i <= 4; i++) {
    const v = bot + ((top - bot) / 4) * i, yy = y(v);
    svg += `<line x1="${M.l}" y1="${yy}" x2="${W - M.r}" y2="${yy}" stroke="var(--line)" stroke-width="1"/>`;
    svg += `<text class="axis-label" x="${M.l - 6}" y="${yy + 3}" text-anchor="end">${fmtAxis(v)}</text>`;
  }
  // Rainfall hyetograph: daily bars hanging from the top, on their own scale
  // (max bar = 28% of plot height), clipped to the flow series' time window.
  const bars = (rain || [])
    .map(b => ({ ...b, t0: Math.max(b.t0, tMin), t1: Math.min(b.t1, tMax) }))
    .filter(b => b.t1 > b.t0 && b.mm > 0);
  const rainMax = bars.length ? Math.max(...bars.map(b => b.mm)) : 0;
  if (rainMax > 0) {
    for (const b of bars) {
      const h = (b.mm / rainMax) * ph * 0.28;
      svg += `<rect x="${x(b.t0).toFixed(1)}" y="${M.t}" width="${(x(b.t1) - x(b.t0)).toFixed(1)}"`
           + ` height="${h.toFixed(1)}" fill="#7db4e8" opacity="0.5"/>`;
    }
    svg += `<text class="axis-label" x="${W - M.r}" y="${M.t + 9}" text-anchor="end"`
         + ` fill="#5f83a6">rain, wettest day ${rainMax.toFixed(1)} mm</text>`;
  }

  if (obs.length > 1) svg += `<path d="${line(obs)}" fill="none" stroke="${FLOW}" stroke-width="2"/>`;
  if (cnn.length > 1) svg += `<path d="${line(cnn)}" fill="none" stroke="${FLOW}" stroke-width="1.4" stroke-dasharray="4 3" opacity="0.85"/>`;
  if (goog.length > 1) svg += `<path d="${line(goog)}" fill="none" stroke="#00897b" stroke-width="1.4" stroke-dasharray="6 3 1.5 3" opacity="0.9"/>`;
  for (const d of goog) svg += `<circle cx="${x(d.t)}" cy="${y(d.v)}" r="2.6" fill="#00897b"/>`;
  for (let i = 0; i <= 4; i++) {
    const t = tMin + ((tMax - tMin) / 4) * i;
    const anchor = i === 0 ? 'start' : i === 4 ? 'end' : 'middle';
    const xx = i === 0 ? M.l : i === 4 ? W - M.r : x(t);
    svg += `<text class="axis-label" x="${xx}" y="${H - 5}" text-anchor="${anchor}">${fmtTick(t)}</text>`;
  }
  return svg + '</svg>';
}

function compareStats(obs, cnn, goog) {
  if (!obs.length) return `<div class="chart-note">No USGS observations to score against.</div>`;
  const obsAt = new Map(obs.map(d => [d.t, d.v]));
  // CNN 12h-ahead: hourly, scored against the observation at the same hour.
  const cnnErr = cnn.filter(d => obsAt.has(d.t)).map(d => Math.abs(d.v - obsAt.get(d.t)));
  // Google next-day: daily value, scored against that day's observed mean.
  const dayMean = new Map();
  for (const d of obs) {
    const day = new Date(d.t).toISOString().slice(0, 10);
    if (!dayMean.has(day)) dayMean.set(day, []);
    dayMean.get(day).push(d.v);
  }
  const googErr = goog.map(d => {
    const day = new Date(d.t).toISOString().slice(0, 10);
    const m = dayMean.get(day);
    return m ? Math.abs(d.v - m.reduce((a, b) => a + b) / m.length) : null;
  }).filter(e => e != null);
  const mae = a => a.length ? (a.reduce((x, y) => x + y) / a.length) : null;
  const f = v => v == null ? '—' : (v >= 100 ? Math.round(v).toLocaleString() : v.toFixed(2));
  return `<div class="legend-row">
    <div class="item"><span class="lbl">CNN 12h MAE</span><div class="val">${f(mae(cnnErr))} m³/s</div></div>
    <div class="item"><span class="lbl">Google 1d MAE</span><div class="val">${f(mae(googErr))} m³/s</div></div>
    <div class="item"><span class="lbl">Hours scored</span><div class="val">${cnnErr.length}</div></div>
  </div>`;
}

/* ---- SVG chart ---------------------------------------------------------- */

function chartSVG(p) {
  const W = 344, H = 196, M = { t: 14, r: 10, b: 22, l: 38 };
  const pw = W - M.l - M.r, ph = H - M.t - M.b;

  const obs = p.series.filter(e => 'o' in e).map(e => ({ t: +new Date(e.d), v: e.o }));
  const fc  = p.series.filter(e => 'p' in e).map(e => ({ t: +new Date(e.d), v: e.p }));

  // NWM short-range overlay clipped to the chart's time window. When the
  // gauge itself is offline this is the only series — still worth drawing.
  const nwm = (p.noaa_nwm || []).map(e => ({ t: +new Date(e.t), v: e.flow_m3s }));
  if (!obs.length && !fc.length && nwm.length < 2) return `<svg width="${W}" height="${H}"></svg>`;

  const base = [...obs, ...fc, ...nwm];
  const tMin = Math.min(...base.map(d => d.t));
  const tMax = Math.max(...base.map(d => d.t));

  // Google Flood Hub forecast — only when it speaks discharge (a stage-only
  // model can't share the m³/s axis), clipped to the chart window since
  // Google forecasts run out ~7 days.
  const gf = p.google_flood;
  const goog = (gf && gf.unit === 'CUBIC_METERS_PER_SECOND' ? gf.forecast || [] : [])
    .map(e => ({ t: +new Date(e.t), v: e.v }))
    .filter(d => d.t >= tMin && d.t <= tMax);

  const all = [...base, ...goog];
  const th = p.thresholds;

  // Fit the y-axis to the data so the observed→forecast line is actually
  // legible. Thresholds within reach are drawn as reference lines; thresholds
  // far above the current flow (calm baseflow) become compact chips instead of
  // crushing the series flat against the axis.
  const vals = all.map(d => d.v);
  let dataMax = Math.max(...vals), dataMin = Math.min(...vals);
  if (!isFinite(dataMax)) { dataMax = 1; dataMin = 0; }
  const span = (dataMax - dataMin) || dataMax || 1;
  let top = dataMax + span * 0.25;
  const bot = Math.max(0, dataMin - span * 0.25);
  const NEAR = dataMax + span * 1.5;   // threshold "within reach" cutoff
  const thrLines = [], thrChips = [];
  if (th) for (const [name, color] of [['warning', RISK.warning], ['danger', RISK.danger], ['extreme', RISK.extreme]]) {
    const v = th[name]; if (v == null) continue;
    if (v <= NEAR) { thrLines.push([name, color, v]); top = Math.max(top, v + span * 0.18); }
    else thrChips.push([name, color, v]);
  }

  const x = t => M.l + ((t - tMin) / (tMax - tMin || 1)) * pw;
  const y = v => M.t + ph - ((v - bot) / (top - bot || 1)) * ph;
  const nowX = obs.length ? x(obs[obs.length - 1].t) : x(tMin);

  const line = pts => pts.map((d, i) => (i ? 'L' : 'M') + x(d.t).toFixed(1) + ' ' + y(d.v).toFixed(1)).join(' ');
  const join = obs.length && fc.length ? [obs[obs.length - 1], ...fc] : fc;

  let svg = `<svg width="${W}" height="${H}" viewBox="0 0 ${W} ${H}">`;

  // y gridlines + labels
  for (let i = 0; i <= 4; i++) {
    const v = bot + ((top - bot) / 4) * i, yy = y(v);
    svg += `<line x1="${M.l}" y1="${yy}" x2="${W - M.r}" y2="${yy}" stroke="var(--line)" stroke-width="1"/>`;
    svg += `<text class="axis-label" x="${M.l - 6}" y="${yy + 3}" text-anchor="end">${fmtAxis(v)}</text>`;
  }

  // threshold reference lines (those within reach of the current flow)
  for (const [name, color, v] of thrLines) {
    const yy = y(v);
    svg += `<line x1="${M.l}" y1="${yy}" x2="${W - M.r}" y2="${yy}" stroke="${color}" stroke-width="1.4" stroke-dasharray="5 3"/>`;
    svg += `<text class="thr-label" x="${W - M.r}" y="${yy - 3}" text-anchor="end" fill="${color}">${cap(name)} ${fmtAxis(v)}</text>`;
  }
  // off-scale thresholds (gauge well below flood stage) — compact chips
  thrChips.forEach(([name, color, v], i) => {
    svg += `<text class="thr-label" x="${M.l + 3}" y="${M.t + 9 + i * 13}" text-anchor="start" fill="${color}">▲ ${cap(name)} ${fmtAxis(v)}</text>`;
  });

  // NWM overlay (dotted)
  if (nwm.length > 1) svg += `<path d="${line(nwm)}" fill="none" stroke="#7b61ff" stroke-width="1.5" stroke-dasharray="1.5 2.5" opacity="0.9"/>`;

  // Google Flood Hub overlay (dash-dot)
  if (goog.length > 1) svg += `<path d="${line(goog)}" fill="none" stroke="#00897b" stroke-width="1.5" stroke-dasharray="6 3 1.5 3" opacity="0.9"/>`;

  // observed (solid) + forecast (dashed)
  if (obs.length > 1) svg += `<path d="${line(obs)}" fill="none" stroke="${FLOW}" stroke-width="2.2"/>`;
  if (join.length > 1) svg += `<path d="${line(join)}" fill="none" stroke="${FLOW}" stroke-width="2.2" stroke-dasharray="4 3"/>`;

  // "Now" marker
  svg += `<line x1="${nowX}" y1="${M.t}" x2="${nowX}" y2="${M.t + ph}" stroke="#9aa0a6" stroke-width="1" stroke-dasharray="3 3"/>`;
  svg += `<text class="axis-label" x="${nowX + 3}" y="${M.t + 9}" fill="#5f6368">Now</text>`;

  // x ticks (start, now, end)
  svg += `<text class="axis-label" x="${M.l}" y="${H - 6}" text-anchor="start">${fmtTick(tMin)}</text>`;
  svg += `<text class="axis-label" x="${W - M.r}" y="${H - 6}" text-anchor="end">${fmtTick(tMax)}</text>`;

  svg += `</svg>`;
  return svg;
}

/* ---- formatting --------------------------------------------------------- */

function cap(s) { return s[0].toUpperCase() + s.slice(1); }
function fmt(v) { return v >= 100 ? Math.round(v).toLocaleString() : v.toFixed(2); }
function fmtAxis(v) { return v >= 1000 ? (v / 1000).toFixed(1) + 'k' : Math.round(v); }
function fmtTick(t) { return new Date(t).toLocaleString([], { month: 'numeric', day: 'numeric', hour: 'numeric' }); }
function fmtTime(iso) { return new Date(iso).toLocaleString([], { month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit' }); }
function fmtAge(iso) {
  const mins = Math.round((Date.now() - new Date(iso)) / 60000);
  if (mins < 60) return mins + ' min ago';
  const h = Math.round(mins / 60);
  return h < 24 ? h + ' h ago' : new Date(iso).toLocaleDateString();
}
