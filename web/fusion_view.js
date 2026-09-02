class FusionView {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.data = null;
    this.showPlasma = true;
    this.showMagnetic = false;
    this.particleFilter = 'all';
    this.yaw = -0.18;
    this.pitch = 0.92;
    this.zoom = 1;
    this.drag = null;
    this.dpr = 1;
    canvas.addEventListener('pointerdown', e => this.onDown(e));
    canvas.addEventListener('pointermove', e => this.onMove(e));
    canvas.addEventListener('pointerup', e => this.onUp(e));
    canvas.addEventListener('pointercancel', e => this.onUp(e));
    canvas.addEventListener('dblclick', () => this.reset());
    canvas.addEventListener('wheel', e => {
      e.preventDefault(); e.stopPropagation();
      this.zoom = Math.max(.62, Math.min(1.8, this.zoom * (e.deltaY < 0 ? 1.09 : .92)));
      this.draw();
    }, {passive: false});
  }

  async load(url, preserveView=false) {
    const request=(this.loadRequest||0)+1;this.loadRequest=request;
    const response = await fetch(url, {cache: 'no-store'});
    if (!response.ok) throw new Error(`interactive fusion state returned ${response.status}`);
    const data = await response.json();
    if (data.kind !== 'fusion-torus' || !Array.isArray(data.shape) || !Array.isArray(data.texture)) {
      throw new Error('interactive fusion state is invalid');
    }
    if (data.texture.length !== data.shape[0] * data.shape[1]) {
      throw new Error('interactive fusion texture has the wrong size');
    }
    if(request!==this.loadRequest)return;
    this.data = data;
    if(preserveView)this.resize();else this.reset();
  }

  reset() {
    this.yaw = -0.18;
    this.pitch = 0.92;
    this.zoom = 1;
    this.resize();
  }

  resize() {
    const rect = this.canvas.getBoundingClientRect();
    if (!rect.width || !rect.height) return;
    this.dpr = Math.min(window.devicePixelRatio || 1, 2);
    const width = Math.round(rect.width * this.dpr), height = Math.round(rect.height * this.dpr);
    if (this.canvas.width !== width || this.canvas.height !== height) {
      this.canvas.width = width; this.canvas.height = height;
    }
    this.draw();
  }

  setLayer(layer, enabled) {
    if (layer === 'magnetic') this.showMagnetic = Boolean(enabled);
    if (layer === 'plasma') this.showPlasma = Boolean(enabled);
    this.draw();
  }

  setParticleFilter(filter) {
    this.particleFilter = filter === 'all' ? 'all' : String(Math.max(0, Math.min(3, Number(filter) || 0)));
    this.draw();
  }

  onDown(event) {
    event.preventDefault(); event.stopPropagation();
    this.drag = {id: event.pointerId, x: event.clientX, y: event.clientY, yaw: this.yaw, pitch: this.pitch};
    this.canvas.setPointerCapture(event.pointerId);
    this.canvas.classList.add('isPanning');
  }

  onMove(event) {
    if (!this.drag) return;
    event.preventDefault(); event.stopPropagation();
    this.yaw = this.drag.yaw + (event.clientX - this.drag.x) * .009;
    this.pitch = Math.max(-1.42, Math.min(1.42, this.drag.pitch + (event.clientY - this.drag.y) * .008));
    this.draw();
  }

  onUp(event) {
    if (!this.drag) return;
    const id = this.drag.id; this.drag = null;
    this.canvas.classList.remove('isPanning');
    try { this.canvas.releasePointerCapture(id); } catch (_) {}
  }

  project(uNorm, vNorm, minor=.40) {
    const u = uNorm * Math.PI * 2 + this.yaw;
    const v = vNorm * Math.PI * 2;
    const x = (1 + minor * Math.cos(v)) * Math.cos(u);
    const y = (1 + minor * Math.cos(v)) * Math.sin(u);
    const z = minor * Math.sin(v);
    const cp = Math.cos(this.pitch), sp = Math.sin(this.pitch);
    const yy = y * cp - z * sp;
    const zz = y * sp + z * cp;
    const w = this.canvas.width, h = this.canvas.height;
    const scale = Math.min(w / 3.15, h / 2.25) * this.zoom;
    return {x: w * .5 + x * scale, y: h * .52 - yy * scale, z: zz};
  }

  palette(value, light=1) {
    const x = Math.max(0, Math.min(1, value / 255));
    const r = Math.min(1, .08 + 1.3 * Math.pow(x, 1.6));
    const g = Math.min(1, .02 + 1.0 * Math.pow(x, 2.2));
    const b = Math.min(1, .18 + 1.2 * x * (1-x) + .6*x);
    return [Math.round(255*r*light), Math.round(255*g*light), Math.round(255*b*light)];
  }

  drawSurface(alpha) {
    const [ny, nx] = this.data.shape, texture = this.data.texture;
    const stride = Math.max(1, Math.ceil(ny / 64));
    const points = [];
    for (let row=0; row<ny; row+=stride) {
      for (let col=0; col<nx; col+=stride) {
        const p = this.project(col/nx, row/ny);
        points.push({...p, value: texture[row*nx+col]});
      }
    }
    points.sort((a,b) => a.z-b.z);
    const ctx=this.ctx, radius=Math.max(1.1*this.dpr, Math.min(this.canvas.width,this.canvas.height)/350);
    ctx.globalCompositeOperation='lighter';
    for (const p of points) {
      const front=Math.max(.35,Math.min(1,.54+.46*(p.z+.65)/1.3));
      const [r,g,b]=this.palette(p.value,front);
      ctx.fillStyle=`rgba(${r},${g},${b},${alpha*front})`;
      ctx.beginPath();ctx.arc(p.x,p.y,radius*(.72+.35*front),0,Math.PI*2);ctx.fill();
    }
    ctx.globalCompositeOperation='source-over';
  }

  drawTrails() {
    const trails=this.data.trails||[], segments=[];
    for (let particle=0; particle<trails.length; particle++) {
      if (this.particleFilter !== 'all' && particle % 4 !== Number(this.particleFilter)) continue;
      const points=trails[particle].map(uv=>this.project(uv[0],uv[1]));
      for (let k=1;k<points.length;k++) {
        const a=points[k-1],b=points[k];
        if (Math.abs(a.x-b.x)>this.canvas.width*.22||Math.abs(a.y-b.y)>this.canvas.height*.22) continue;
        segments.push({a,b,z:(a.z+b.z)/2,age:k/(points.length-1),particle});
      }
    }
    segments.sort((a,b)=>a.z-b.z);
    const colours=[[104,239,255],[255,173,83],[224,126,255],[190,255,228]],ctx=this.ctx;
    ctx.lineCap='round';ctx.globalCompositeOperation='lighter';
    for(const s of segments){
      const c=colours[s.particle%colours.length];
      const front=Math.max(.3,Math.min(1,.46+.54*(s.z+.65)/1.3));
      ctx.strokeStyle=`rgba(${c[0]},${c[1]},${c[2]},${(.14+.72*s.age)*front})`;
      ctx.lineWidth=(1.2+1.8*front)*this.dpr;ctx.beginPath();ctx.moveTo(s.a.x,s.a.y);ctx.lineTo(s.b.x,s.b.y);ctx.stroke();
    }
    ctx.globalCompositeOperation='source-over';
    for(let particle=0;particle<trails.length;particle++){
      if (this.particleFilter !== 'all' && particle % 4 !== Number(this.particleFilter)) continue;
      const uv=trails[particle][trails[particle].length-1],p=this.project(uv[0],uv[1]);
      const c=colours[particle%colours.length],front=Math.max(.35,Math.min(1,.48+.52*(p.z+.65)/1.3));
      ctx.fillStyle=`rgba(248,255,255,${front})`;ctx.strokeStyle=`rgb(${c[0]},${c[1]},${c[2]})`;ctx.lineWidth=1.2*this.dpr;
      ctx.beginPath();ctx.arc(p.x,p.y,(2.1+2.5*front)*this.dpr,0,Math.PI*2);ctx.fill();ctx.stroke();
    }
  }

  drawMagneticField() {
    const count=this.data.field_lines||14,pitch=this.data.field_pitch||.7,segments=[];
    const samples=220,turns=2.35;
    for(let line=0;line<count;line++){
      const minor=[.19,.30,.40][line%3],offset=line/count;
      let previous=this.project(0,offset,minor);
      for(let s=1;s<samples;s++){
        const toroidal=s/(samples-1)*turns;
        const current=this.project(toroidal,offset+pitch*toroidal,minor);
        segments.push({a:previous,b:current,z:(previous.z+current.z)/2,line});
        previous=current;
      }
    }
    // The magnetic axis is the centre of the nested flux surfaces.
    let previous=this.project(0,0,0),axis=[];
    for(let s=1;s<180;s++){const current=this.project(s/179,0,0);axis.push({a:previous,b:current,z:(previous.z+current.z)/2});previous=current;}
    segments.sort((a,b)=>a.z-b.z);axis.sort((a,b)=>a.z-b.z);
    const ctx=this.ctx;ctx.lineCap='round';ctx.globalCompositeOperation='lighter';
    for(const s of segments){
      const front=Math.max(.28,Math.min(1,.46+.54*(s.z+.65)/1.3));
      const hue=s.line%3===0?[88,235,255]:s.line%3===1?[112,155,255]:[179,115,255];
      ctx.strokeStyle=`rgba(${hue[0]},${hue[1]},${hue[2]},${.22+.70*front})`;
      ctx.lineWidth=(1.1+1.8*front)*this.dpr;ctx.beginPath();ctx.moveTo(s.a.x,s.a.y);ctx.lineTo(s.b.x,s.b.y);ctx.stroke();
    }
    for(const s of axis){ctx.strokeStyle='rgba(255,210,112,.86)';ctx.lineWidth=3*this.dpr;ctx.beginPath();ctx.moveTo(s.a.x,s.a.y);ctx.lineTo(s.b.x,s.b.y);ctx.stroke();}
    ctx.globalCompositeOperation='source-over';
  }

  draw() {
    if(!this.data||!this.canvas.width||!this.canvas.height)return;
    const ctx=this.ctx,w=this.canvas.width,h=this.canvas.height;
    const gradient=ctx.createRadialGradient(w*.48,h*.40,0,w*.5,h*.5,Math.max(w,h)*.65);
    gradient.addColorStop(0,this.showMagnetic?'#07152b':'#0a0b28');gradient.addColorStop(.55,'#030711');gradient.addColorStop(1,'#010309');
    ctx.fillStyle=gradient;ctx.fillRect(0,0,w,h);
    if(this.showPlasma){this.drawSurface(this.showMagnetic?.28:.42);this.drawTrails();}
    else if(this.showMagnetic){this.drawSurface(.10);}
    if(this.showMagnetic)this.drawMagneticField();
    // Titles, parameter readouts and method notes belong to the HTML view
    // layers. The canvas remains a clean, rotatable scientific viewport.
  }
}

window.FusionView=FusionView;
