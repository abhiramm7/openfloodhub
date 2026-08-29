"""Record the demo GIF for the README: open the map, zoom out to the full
Potomac catchment, click the Potomac gauge, open the model-performance
comparison.

Re-run whenever the UI changes:

    .venv/bin/python scripts/make_demo_gif.py     # writes assets/demo.gif

Dev-only dependencies (not in requirements.txt):
    uv pip install --python .venv/bin/python playwright pillow
    .venv/bin/playwright install chromium

Serves web/ on a local port itself, so nothing needs to be running first.
"""
from __future__ import annotations

import http.server
import io
import socket
import threading
from functools import partial
from pathlib import Path

from PIL import Image
from playwright.sync_api import sync_playwright

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / 'assets' / 'demo.gif'
VIEWPORT = {'width': 1440, 'height': 900}
DEVICE_SCALE = 2        # hi-res capture
GIF_WIDTH = 1440
FRAME_MS = 100          # base frame duration

POTOMAC = (38.9498, -77.1276)

CURSOR_JS = """
() => {
  const c = document.createElement('div');
  c.id = 'fakecursor';
  c.style.cssText = `position:fixed;width:16px;height:16px;border-radius:50%;
    border:2.5px solid #000;background:rgba(255,255,255,.75);z-index:99999;
    pointer-events:none;transform:translate(-50%,-50%);left:640px;top:400px;
    box-shadow:0 1px 4px rgba(0,0,0,.4)`;
  document.body.appendChild(c);
}
"""


def serve_web(port: int):
    handler = partial(http.server.SimpleHTTPRequestHandler,
                      directory=str(REPO / 'web'))
    srv = http.server.ThreadingHTTPServer(('127.0.0.1', port), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def free_port() -> int:
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


def main():
    port = free_port()
    serve_web(port)

    frames: list[Image.Image] = []
    durations: list[int] = []

    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page(viewport=VIEWPORT,
                                device_scale_factor=DEVICE_SCALE)

        def cursor_to(x, y):
            page.evaluate(
                "([x, y]) => { const c = document.getElementById('fakecursor');"
                " c.style.left = x + 'px'; c.style.top = y + 'px'; }", [x, y])

        def cursor_flash(on):
            page.evaluate(
                "on => { document.getElementById('fakecursor').style.borderColor"
                " = on ? '#ff3300' : '#000'; }", on)

        def shot(hold_ms=FRAME_MS):
            img = Image.open(io.BytesIO(page.screenshot()))
            scale = GIF_WIDTH / img.width
            img = img.resize((GIF_WIDTH, round(img.height * scale)),
                             Image.LANCZOS).convert('RGB')
            frames.append(img)
            durations.append(hold_ms)

        def glide(x0, y0, x1, y1, steps=7):
            for i in range(1, steps + 1):
                f = i / steps
                # ease in-out
                e = f * f * (3 - 2 * f)
                cursor_to(x0 + (x1 - x0) * e, y0 + (y1 - y0) * e)
                shot()

        def click(x, y, settle_frames=6):
            cursor_flash(True)
            shot(120)
            page.mouse.click(x, y)
            for _ in range(settle_frames):   # catch the CSS transitions
                shot(80)
            cursor_flash(False)

        # 1. Open the page. The app auto-selects the Potomac; start the story
        #    from the bare map instead, as a first-time visitor would see it.
        page.goto(f'http://127.0.0.1:{port}/')
        page.wait_for_load_state('networkidle')
        page.evaluate('document.fonts.ready')
        page.evaluate('deselect()')
        page.wait_for_timeout(400)
        page.evaluate(CURSOR_JS)
        cx, cy = VIEWPORT['width'] / 2, VIEWPORT['height'] / 2
        cursor_to(cx, cy)
        shot(1200)

        # 2. Zoom out until the Potomac's entire HYBAS catchment fits, so the
        #    basin outline lands fully in frame when the gauge is selected.
        page.evaluate(
            "() => { const b = L.geoJSON(BASINS['01646500'].geometry).getBounds();"
            " map.flyToBounds(b.pad(0.06), {duration: 1.4}); }")
        for _ in range(12):     # frames during the flight
            shot()
        page.wait_for_timeout(1500)   # let tiles at the new zoom land
        shot(1000)

        # 3. Click the Potomac gauge.
        pt = page.evaluate(
            "([lat, lon]) => { const p = map.latLngToContainerPoint([lat, lon]);"
            " const r = document.getElementById('map').getBoundingClientRect();"
            " return [r.left + p.x, r.top + p.y]; }", list(POTOMAC))
        glide(cx, cy, pt[0], pt[1])
        click(pt[0], pt[1])
        shot(1600)

        # 3. Click "Compare model predictions" to open the performance drawer.
        box = page.locator('#cmpToggle').bounding_box()
        tx, ty = box['x'] + box['width'] / 2, box['y'] + box['height'] / 2
        glide(pt[0], pt[1], tx, ty)
        click(tx, ty)
        shot(1000)
        shot(2400)

        browser.close()

    OUT.parent.mkdir(exist_ok=True)
    frames[0].save(
        OUT, save_all=True, append_images=frames[1:], loop=0,
        duration=durations, optimize=True)
    print(f'wrote {OUT} ({OUT.stat().st_size / 1e6:.2f} MB, {len(frames)} frames)')


if __name__ == '__main__':
    main()
