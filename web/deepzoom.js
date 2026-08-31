/* Tiled deep-zoom viewer.

   Magnification is quantised onto a log2 ladder: level L means the model is
   rendered at 2**L tiles across, and the viewer always draws the level whose
   tiles land nearest 1:1 on screen. Tiles are fetched on demand, rendered
   server-side at that level's native resolution and cached, so the picture is
   never scaled by more than about 2x and never dissolves into a blur however
   far in you go.

   The earlier version asked for one image of the exact current window. Between
   a render completing and the next arriving the browser was stretching a stale
   bitmap, which past a few hundred times magnification looked exactly like
   zooming into a photograph.

   Coarser levels are painted first as a progressive backdrop, so there is
   always something sharp-ish on screen while finer tiles load.

   Panning is strictly drag-to-pan: it starts on a primary-button press and
   ends on pointerup/cancel/leave/blur, including at window level, so the view
   can never be left following the cursor with no button held. */
class DeepZoom {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.cache = new Map();
    this.inflight = 0;
    this.manifest = null;
    this.tileUrl = null;
    this.drag = null;
    this.onstatus = null;
    this.maxLevel = 40;
    this.dpr = window.devicePixelRatio || 1;

    this._onDown = e => this.onDown(e);
    this._onMove = e => this.onMove(e);
    this._onUp = () => this.endDrag();
    this._onWheel = e => this.onWheel(e);
    canvas.addEventListener('wheel', this._onWheel, { passive: false });
    canvas.addEventListener('pointerdown', this._onDown);
    canvas.addEventListener('pointermove', this._onMove);
    canvas.addEventListener('pointerup', this._onUp);
    canvas.addEventListener('pointercancel', this._onUp);
    canvas.addEventListener('pointerleave', this._onUp);
    window.addEventListener('pointerup', this._onUp);
    window.addEventListener('blur', this._onUp);
  }

  load(base, manifest, tileUrl) {
    this.base = (base || '').replace(/\/$/, '');
    this.manifest = manifest || { tile: 256, levels: 0, cx: 0, cy: 0, span: 2 };
    this.tileUrl = tileUrl || null;
    this.cache.clear();
    this.resize();
    this.home();
  }

  resize() {
    const c = this.canvas, r = c.getBoundingClientRect();
    this.dpr = window.devicePixelRatio || 1;
    c.width = Math.max(1, Math.round(r.width * this.dpr));
    c.height = Math.max(1, Math.round(r.height * this.dpr));
    if (this.manifest) this.draw();
  }

  home() {
    const m = this.manifest; if (!m) return;
    this.cx = m.cx; this.cy = m.cy;
    this.scale = Math.min(this.canvas.width, this.canvas.height) / m.span;
    this.minScale = this.scale;
    this.draw();
  }

  get zoomFactor() { return this.scale / this.minScale; }

  toWorld(px, py) {
    const c = this.canvas;
    return {
      x: this.cx + (px * this.dpr - c.width / 2) / this.scale,
      y: this.cy - (py * this.dpr - c.height / 2) / this.scale,
    };
  }

  // Level whose tiles are closest to 1:1 on screen. Rounding (not flooring)
  // keeps the worst-case scaling to sqrt(2) either way.
  targetLevel() {
    const m = this.manifest;
    const want = Math.log2(m.span * this.scale / m.tile);
    return Math.max(0, Math.min(this.maxLevel, Math.round(want)));
  }

  zoomAt(factor, px, py) {
    if (!this.manifest) return;
    const before = this.toWorld(px, py);
    const maxScale = this.minScale * Math.pow(2, this.maxLevel);
    this.scale = Math.max(this.minScale, Math.min(maxScale, this.scale * factor));
    const after = this.toWorld(px, py);
    this.cx += before.x - after.x;
    this.cy += before.y - after.y;
    this.draw();
  }

  onWheel(e) {
    if (!this.manifest) return;
    e.preventDefault(); e.stopPropagation();
    const r = this.canvas.getBoundingClientRect();
    this.zoomAt(e.deltaY < 0 ? 1.35 : 1 / 1.35, e.clientX - r.left, e.clientY - r.top);
  }

  onDown(e) {
    if (!this.manifest || e.button !== 0) return;
    e.preventDefault(); e.stopPropagation();
    this.drag = { x: e.clientX, y: e.clientY, id: e.pointerId };
    try { this.canvas.setPointerCapture(e.pointerId); } catch (_) {}
    this.canvas.classList.add('isPanning');
  }

  onMove(e) {
    if (!this.drag) return;
    if (e.buttons === 0) { this.endDrag(); return; }
    e.stopPropagation();
    this.cx -= (e.clientX - this.drag.x) * this.dpr / this.scale;
    this.cy += (e.clientY - this.drag.y) * this.dpr / this.scale;
    this.drag.x = e.clientX; this.drag.y = e.clientY;
    this.draw();
  }

  endDrag() {
    if (!this.drag) return;
    const id = this.drag.id;
    this.drag = null;
    this.canvas.classList.remove('isPanning');
    try { if (id !== undefined) this.canvas.releasePointerCapture(id); } catch (_) {}
  }

  destroy() {
    window.removeEventListener('pointerup', this._onUp);
    window.removeEventListener('blur', this._onUp);
    this.drag = null;
  }

  tile(level, col, row) {
    const key = `${level}/${col}/${row}`;
    const hit = this.cache.get(key);
    if (hit) return hit.ok ? hit.img : null;
    // Cap concurrency: each miss is a real render on the server, and firing a
    // whole screen's worth at once just delays every one of them.
    if (this.inflight >= 6) return null;
    const img = new Image();
    const entry = { img, ok: false };
    this.cache.set(key, entry);
    this.inflight++;
    img.onload = () => { entry.ok = true; this.inflight--; this.draw(); };
    img.onerror = () => { entry.failed = true; this.inflight--; };
    // Baked levels are static files; anything deeper is rendered on demand.
    img.src = (this.tileUrl && level > (this.manifest.levels || 0))
      ? `${this.tileUrl}?level=${level}&col=${col}&row=${row}&tile=${this.manifest.tile}`
      : `${this.base}/L${level}/${col}_${row}.jpg`;
    return null;
  }

  drawLevel(level, left, right, top, bottom) {
    const m = this.manifest;
    const n = Math.pow(2, level), sub = m.span / n;
    const bx = m.cx - m.span / 2, by = m.cy + m.span / 2;
    const c0 = Math.max(0, Math.floor((left - bx) / sub));
    const c1 = Math.min(n - 1, Math.ceil((right - bx) / sub));
    const r0 = Math.max(0, Math.floor((by - top) / sub));
    const r1 = Math.min(n - 1, Math.ceil((by - bottom) / sub));
    // A pathological view would ask for thousands of tiles; clamp so a bad
    // frame degrades instead of hanging the browser.
    if ((c1 - c0 + 1) * (r1 - r0 + 1) > 240) return 0;
    let drawn = 0;
    for (let col = c0; col <= c1; col++) {
      for (let row = r0; row <= r1; row++) {
        const img = this.tile(level, col, row);
        if (!img) continue;
        const sx = (bx + sub * col - left) * this.scale;
        const sy = (top - (by - sub * row)) * this.scale;
        const size = sub * this.scale;
        this.ctx.drawImage(img, sx, sy, size + 1, size + 1);
        drawn++;
      }
    }
    return drawn;
  }

  draw() {
    const m = this.manifest; if (!m) return;
    const c = this.canvas, ctx = this.ctx;
    ctx.fillStyle = '#03060f';
    ctx.fillRect(0, 0, c.width, c.height);
    ctx.imageSmoothingEnabled = true;

    const left = this.cx - c.width / 2 / this.scale;
    const top = this.cy + c.height / 2 / this.scale;
    const right = left + c.width / this.scale;
    const bottom = top - c.height / this.scale;
    const target = this.targetLevel();

    // Progressive backdrop: a few coarser levels, then the target on top.
    // Only a handful, otherwise every draw walks 40 levels of tile lookups.
    for (let l = Math.max(0, target - 3); l < target; l++) {
      this.drawLevel(l, left, right, top, bottom);
    }
    this.drawLevel(target, left, right, top, bottom);

    if (this.onstatus) this.onstatus(this.zoomFactor, target, this.maxLevel);
  }
}
window.DeepZoom = DeepZoom;
