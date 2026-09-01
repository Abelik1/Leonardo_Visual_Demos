/* Responsive coherent deep zoom.

   A finite recursive growth budget means independent image tiles can choose
   different branch frontiers at their borders. This viewer uses one cached
   source image per quantised viewport instead. Every wheel/pan event projects
   the last good source immediately; one background request then refines it. */
class DeepZoom {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d', { alpha: false });
    this.cache = new Map(); this.pending = new Set();
    this.maxCachedViews = 64;
    this.manifest = null; this.viewUrl = null; this.drag = null;
    this.onstatus = null; this.maxLevel = 24;
    this.dpr = window.devicePixelRatio || 1;
    this.fetchTimer = null; this.generation = 0;
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

  load(base, manifest, viewUrl) {
    this.base = (base || '').replace(/\/$/, '');
    this.manifest = manifest || { cx: 0, cy: 0, span: 2 };
    this.viewUrl = viewUrl || null;
    const detailBase = Number(this.manifest.detail_base ?? 7);
    const detailMax = Number(this.manifest.detail_max ?? (detailBase + 7));
    // Older runs did not record stable-grammar metadata; their safe fallback
    // is seven refinement transitions, not their obsolete image-pyramid cap.
    this.maxLevel = Math.max(0, Math.min(12, Math.floor(detailMax - detailBase)));
    this.cache.clear(); this.pending.clear(); this.generation++;
    this.resize(); this.home();
  }

  resize() {
    const r = this.canvas.getBoundingClientRect();
    this.dpr = window.devicePixelRatio || 1;
    this.canvas.width = Math.max(1, Math.round(r.width * this.dpr));
    this.canvas.height = Math.max(1, Math.round(r.height * this.dpr));
    if (this.manifest) this.draw(true);
  }

  home() {
    if (!this.manifest) return;
    this.cx = this.manifest.cx; this.cy = this.manifest.cy;
    this.scale = Math.min(this.canvas.width, this.canvas.height) / this.manifest.span;
    this.minScale = this.scale;
    this.draw(true);
  }

  get zoomFactor() { return this.scale / this.minScale; }

  toWorld(px, py) {
    return { x: this.cx + (px * this.dpr - this.canvas.width / 2) / this.scale,
             y: this.cy - (py * this.dpr - this.canvas.height / 2) / this.scale };
  }

  targetLevel() {
    return Math.max(0, Math.min(this.maxLevel, Math.floor(Math.log2(this.zoomFactor))));
  }

  detailBlend() {
    const level = this.targetLevel();
    if (level >= this.maxLevel) return 0;
    const fractional = Math.max(0, Math.min(1, Math.log2(this.zoomFactor) - level));
    // Cubic easing fades only the next-depth residual; it never swaps the
    // current crystal for an unrelated one.
    return fractional * fractional * (3 - 2 * fractional);
  }

  worldBounds() {
    const c = this.canvas, span = c.width / this.scale, spanY = c.height / this.scale;
    return { left: this.cx - span / 2, right: this.cx + span / 2,
             top: this.cy + spanY / 2, bottom: this.cy - spanY / 2, span, spanY };
  }

  zoomAt(factor, px, py) {
    if (!this.manifest) return;
    const before = this.toWorld(px, py);
    const maxScale = this.minScale * Math.pow(2, this.maxLevel);
    this.scale = Math.max(this.minScale, Math.min(maxScale, this.scale * factor));
    const after = this.toWorld(px, py);
    this.cx += before.x - after.x; this.cy += before.y - after.y;
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
    this.cx -= (e.clientX - this.drag.x) * this.dpr / this.scale;
    this.cy += (e.clientY - this.drag.y) * this.dpr / this.scale;
    this.drag.x = e.clientX; this.drag.y = e.clientY;
    this.draw();
  }

  endDrag() {
    if (!this.drag) return;
    const id = this.drag.id; this.drag = null;
    this.canvas.classList.remove('isPanning');
    try { this.canvas.releasePointerCapture(id); } catch (_) {}
  }

  destroy() {
    clearTimeout(this.fetchTimer);
    window.removeEventListener('pointerup', this._onUp);
    window.removeEventListener('blur', this._onUp);
    this.drag = null;
  }

  requestSpec(level) {
    const b = this.worldBounds(), step = b.span * .25;
    const cx = Math.round(this.cx / step) * step, cy = Math.round(this.cy / step) * step;
    const span = b.span * 1.5;
    const width = Math.max(768, Math.min(1600, Math.round(this.canvas.width)));
    const height = Math.max(432, Math.min(1000, Math.round(width * this.canvas.height / this.canvas.width)));
    const key = `${level}/${cx.toPrecision(14)}/${cy.toPrecision(14)}/${span.toPrecision(14)}/${width}x${height}`;
    return { key, level, cx, cy, span, width, height };
  }

  scheduleFetch(immediate = false) {
    clearTimeout(this.fetchTimer);
    this.fetchTimer = setTimeout(() => {
      const level = this.targetLevel();
      this.fetchView(level);
      // Fetch the next cumulative depth alongside the current one.  It is
      // composited as a colour residual while the user crosses this octave.
      if (level < this.maxLevel) this.fetchView(level + 1);
    }, immediate ? 0 : 100);
  }

  fetchView(level) {
    if (!this.viewUrl || !this.manifest) return;
    const spec = this.requestSpec(level);
    if (this.cache.has(spec.key) || this.pending.has(spec.key)) return;
    this.pending.add(spec.key);
    const generation = this.generation, image = new Image();
    image.decoding = 'async';
    image.onload = () => {
      this.pending.delete(spec.key);
      if (generation !== this.generation) return;
      this.cache.set(spec.key, { ...spec, image });
      while (this.cache.size > this.maxCachedViews) this.cache.delete(this.cache.keys().next().value);
      this.draw(false);
    };
    image.onerror = () => { this.pending.delete(spec.key); };
    image.src = `${this.viewUrl}?cx=${encodeURIComponent(spec.cx)}&cy=${encodeURIComponent(spec.cy)}` +
      `&span=${encodeURIComponent(spec.span)}&w=${spec.width}&h=${spec.height}&level=${spec.level}`;
  }

  bestView(bounds, level) {
    let best = null, bestScore = Infinity;
    for (const view of this.cache.values()) {
      if (view.level !== level) continue;
      const aspect = view.height / view.width, halfX = view.span / 2, halfY = view.span * aspect / 2;
      const overlapX = Math.max(0, Math.min(bounds.right, view.cx + halfX) - Math.max(bounds.left, view.cx - halfX));
      const overlapY = Math.max(0, Math.min(bounds.top, view.cy + halfY) - Math.max(bounds.bottom, view.cy - halfY));
      const coverage = overlapX * overlapY / Math.max(1e-30, bounds.span * bounds.spanY);
      const score = (1 - coverage) * 10 + Math.abs(Math.log(view.span / (bounds.span * 1.5)));
      if (score < bestScore) { bestScore = score; best = view; }
    }
    return best;
  }

  draw(immediate = false) {
    if (!this.manifest) return;
    const c = this.canvas, ctx = this.ctx, b = this.worldBounds();
    ctx.fillStyle = '#03060f'; ctx.fillRect(0, 0, c.width, c.height);
    const drawView = (view, alpha = 1) => {
      if (!view) return;
      const aspect = view.height / view.width;
      const left = view.cx - view.span / 2, top = view.cy + view.span * aspect / 2;
      const sourceLeft = view.cx - view.span / 2, sourceRight = view.cx + view.span / 2;
      const sourceTop = view.cy + view.span * aspect / 2, sourceBottom = view.cy - view.span * aspect / 2;
      const coverageX = Math.max(0, Math.min(b.right, sourceRight) - Math.max(b.left, sourceLeft));
      const coverageY = Math.max(0, Math.min(b.top, sourceTop) - Math.max(b.bottom, sourceBottom));
      ctx.imageSmoothingEnabled = true;
      ctx.globalAlpha = alpha;
      // On a rapid zoom-out an old source may not yet cover the new edges.
      // Fill with that source temporarily rather than flashing a navy canvas;
      // the exact coherent image replaces it as soon as the debounced request
      // completes.
      if (coverageX * coverageY < b.span * b.spanY * .98) {
        ctx.drawImage(view.image, 0, 0, c.width, c.height);
      } else {
        ctx.drawImage(view.image, (left - b.left) * this.scale, (b.top - top) * this.scale,
                      view.span * this.scale, view.span * aspect * this.scale);
      }
      ctx.globalAlpha = 1;
    };
    const level = this.targetLevel();
    const coarse = this.bestView(b, level);
    const fine = level < this.maxLevel ? this.bestView(b, level + 1) : null;
    drawView(coarse || fine);
    if (coarse && fine) drawView(fine, this.detailBlend());
    this.scheduleFetch(immediate);
    if (this.onstatus) this.onstatus(this.zoomFactor, level, this.maxLevel);
  }
}
window.DeepZoom = DeepZoom;
