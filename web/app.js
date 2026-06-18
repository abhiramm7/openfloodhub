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
L.tileLayer('https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png', {
  attribution: '&copy; OpenStreetMap &copy; CARTO',
  subdomains: 'abcd', maxZoom: 19,
}).addTo(map);

let SELECTED = null;
const MARKERS = {};

main();

async function main() {
  const data = await (await fetch('preds.json', { cache: 'no-store' })).json();
  document.getElementById('updated').textContent =
    'Updated ' + fmtAge(data.updated);

  for (const p of data.predictions) {
    const r = riskOf(p);
    const m = L.circleMarker([p.lat, p.lon], markerStyle(r, false))
      .addTo(map)
      .on('click', () => select(p));
    m.bindTooltip(`${p.name} — ${r.label}`, { direction: 'top', offset: [0, -6] });
    MARKERS[p.id] = { marker: m, risk: r };
  }

  document.getElementById('panelClose').onclick = deselect;

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
  MARKERS[p.id].marker.setStyle(markerStyle(MARKERS[p.id].risk, true));
  renderPanel(p);
  document.getElementById('panel').classList.add('open');
}
function deselect() {
  if (SELECTED && MARKERS[SELECTED]) {
    MARKERS[SELECTED].marker.setStyle(markerStyle(MARKERS[SELECTED].risk, false));
  }
  SELECTED = null;
  document.getElementById('panel').classList.remove('open');
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
    <div class="sub">${p.short} · USGS ${p.id}
      <span class="status-pill" style="background:${r.color};margin-left:6px">${r.label}</span></div>

    <div style="font-size:12px;color:var(--muted);margin-bottom:6px">Discharge (m³/s) — observed &amp; 12-hour forecast</div>
    ${chartSVG(p)}
    <div class="chart-note">Solid = USGS observed · dashed = local CNN forecast · dotted = NOAA NWM</div>

    ${th ? `
    <div class="legend-row">
      <div class="item"><span class="dot" style="background:var(--warning)"></span><span class="lbl">Warning</span><div class="val">${fmt(th.warning)}</div></div>
      <div class="item"><span class="dot" style="background:var(--danger)"></span><span class="lbl">Danger</span><div class="val">${fmt(th.danger)}</div></div>
      <div class="item"><span class="dot" style="background:var(--extreme)"></span><span class="lbl">Extreme</span><div class="val">${fmt(th.extreme)}</div></div>
    </div>` : `<div class="chart-note">No return-period thresholds for this gauge.</div>`}

    <div class="section-title">Now &amp; forecast</div>
    <div class="info-grid">
      <div><div class="k">Current flow</div><div class="v">${cur != null ? fmt(cur) + ' m³/s' : '—'}</div></div>
      <div><div class="k">12h forecast peak</div><div class="v">${peak != null ? fmt(peak) + ' m³/s' : '—'}</div></div>
      <div><div class="k">Issued</div><div class="v">${fmtTime(p.issue_time)}</div></div>
      <div><div class="k">Forecast horizon</div><div class="v">12 hours</div></div>
    </div>

    <div class="section-title">Gauge &amp; model</div>
    <span class="badge">Local per-site model</span>
    <div class="info-grid">
      <div><div class="k">Model</div><div class="v">1D CNN (per gauge)</div></div>
      <div><div class="k">Skill (overall NSE)</div><div class="v">${nse != null ? nse.toFixed(3) : '—'}</div></div>
      <div><div class="k">River gauge ID</div><div class="v">${p.id}</div></div>
      <div><div class="k">Source</div><div class="v">USGS NWIS</div></div>
      <div><div class="k">Lat / Long</div><div class="v">${p.lat.toFixed(4)}, ${p.lon.toFixed(4)}</div></div>
      <div><div class="k">Drainage</div><div class="v">${p.drainage_sqmi.toLocaleString()} mi²</div></div>
      <div><div class="k">Catchment</div><div class="v" style="text-transform:capitalize">${p.kind}</div></div>
      <div><div class="k">Record</div><div class="v">${th ? th.record_years + ' yr' : '—'}</div></div>
    </div>
  `;
}

/* ---- SVG chart ---------------------------------------------------------- */

function chartSVG(p) {
  const W = 344, H = 196, M = { t: 14, r: 10, b: 22, l: 38 };
  const pw = W - M.l - M.r, ph = H - M.t - M.b;

  const obs = p.series.filter(e => 'o' in e).map(e => ({ t: +new Date(e.d), v: e.o }));
  const fc  = p.series.filter(e => 'p' in e).map(e => ({ t: +new Date(e.d), v: e.p }));
  if (!obs.length && !fc.length) return `<svg width="${W}" height="${H}"></svg>`;

  // NWM short-range overlay clipped to the chart's time window.
  const nwm = (p.noaa_nwm || []).map(e => ({ t: +new Date(e.t), v: e.flow_m3s }));

  const all = [...obs, ...fc, ...nwm];
  const tMin = Math.min(...all.map(d => d.t));
  const tMax = Math.max(...all.map(d => d.t));
  const th = p.thresholds;
  const dataMax = Math.max(...all.map(d => d.v));
  // Include threshold bands in the y-range so the gauge's distance-to-flood
  // reads the way Flood Hub shows it (bands above, calm flow low).
  const top = (th ? Math.max(th.extreme, dataMax) : dataMax) * 1.08 || 1;

  const x = t => M.l + ((t - tMin) / (tMax - tMin || 1)) * pw;
  const y = v => M.t + ph - (v / top) * ph;
  const nowX = obs.length ? x(obs[obs.length - 1].t) : x(tMin);

  const line = pts => pts.map((d, i) => (i ? 'L' : 'M') + x(d.t).toFixed(1) + ' ' + y(d.v).toFixed(1)).join(' ');
  const join = obs.length && fc.length ? [obs[obs.length - 1], ...fc] : fc;

  let svg = `<svg width="${W}" height="${H}" viewBox="0 0 ${W} ${H}">`;

  // y gridlines + labels
  for (let i = 0; i <= 4; i++) {
    const v = (top / 4) * i, yy = y(v);
    svg += `<line x1="${M.l}" y1="${yy}" x2="${W - M.r}" y2="${yy}" stroke="var(--line)" stroke-width="1"/>`;
    svg += `<text class="axis-label" x="${M.l - 6}" y="${yy + 3}" text-anchor="end">${fmtAxis(v)}</text>`;
  }

  // threshold lines
  if (th) {
    for (const [name, color] of [['warning', RISK.warning], ['danger', RISK.danger], ['extreme', RISK.extreme]]) {
      const v = th[name]; if (v == null || v > top) continue;
      const yy = y(v);
      svg += `<line x1="${M.l}" y1="${yy}" x2="${W - M.r}" y2="${yy}" stroke="${color}" stroke-width="1.4"/>`;
      svg += `<text class="thr-label" x="${W - M.r}" y="${yy - 3}" text-anchor="end" fill="${color}">${name[0].toUpperCase() + name.slice(1)}</text>`;
    }
  }

  // NWM overlay (dotted)
  if (nwm.length > 1) svg += `<path d="${line(nwm)}" fill="none" stroke="#7b61ff" stroke-width="1.5" stroke-dasharray="1.5 2.5" opacity="0.9"/>`;

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
