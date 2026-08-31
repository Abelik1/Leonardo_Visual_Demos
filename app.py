from __future__ import annotations
import base64, binascii, io, json, math, os, threading, time, uuid
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Literal
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from PIL import Image, UnidentifiedImageError
from run_demo import run, load_profiles, load_specs
from leonardo_demos.registry import DEMOS
from leonardo_demos.backend import probe as probe_backends

ROOT=Path(__file__).resolve().parent; RUNS=ROOT/'runs'; RUNS.mkdir(exist_ok=True)
app=FastAPI(title='Leonardo Visual Demos')
app.mount('/static',StaticFiles(directory=ROOT/'web'),name='static')
app.mount('/runs',StaticFiles(directory=RUNS),name='runs')

class RunReq(BaseModel):
    profile: str = 'local'
    # Desktop GPU runs are deliberately allowed to be longer than the default
    # exhibition loop.  The viewer streams frames as they arrive, so 300-frame
    # living-mathematics runs are valid rather than a request-validation error.
    frames: int = Field(default=70, ge=1, le=600)
    params: dict[str, float] = Field(default_factory=dict)
    backend: Literal['auto', 'numpy', 'cpu', 'cupy', 'cuda', 'gpu', 'hybrid'] = 'auto'
    target_image: str | None = Field(default=None, max_length=400_000)

def save_target_image(data_url: str, destination: Path) -> None:
    """Validate a canvas PNG and save a bounded RGB target image."""
    prefix='data:image/png;base64,'
    if not data_url.startswith(prefix):
        raise HTTPException(422, 'target image must be a PNG drawing')
    try:
        raw=base64.b64decode(data_url[len(prefix):], validate=True)
        with Image.open(io.BytesIO(raw)) as image:
            if image.width < 1 or image.height < 1 or image.width*image.height > 1_000_000:
                raise HTTPException(422, 'target image dimensions are invalid')
            image.convert('RGB').save(destination, format='PNG')
    except HTTPException:
        raise
    except (binascii.Error, OSError, UnidentifiedImageError, ValueError):
        raise HTTPException(422, 'target image could not be read')

@app.get('/',response_class=HTMLResponse)
def index(): return (ROOT/'web/index.html').read_text(encoding='utf-8')

@app.get('/api/specs')
def specs():
    return {'demos':load_specs(),'profiles':load_profiles(),'available':list(DEMOS),
            'backends':probe_backends()}

@app.get('/api/runs')
def list_runs(demo:str|None=None,limit:int=120):
    """Saved runs, newest first, with everything the viewer needs to replay one.

    Every run already persists to runs/<id>/; this simply makes them reachable
    so a finished simulation can be replayed instead of recomputed. That is the
    same path an exhibition uses for its playback fallback.
    """
    out=[]
    for d in sorted(RUNS.glob('*'),key=lambda p:p.stat().st_mtime,reverse=True):
        if not d.is_dir() or d.name.startswith('_'): continue
        try: meta=json.loads((d/'meta.json').read_text())
        except Exception: continue
        if demo and meta.get('demo')!=demo: continue
        frames=sorted((d/'frames').glob('frame_*.jpg')) if (d/'frames').is_dir() else []
        if not frames: continue
        out.append({
            'id':d.name,
            'demo':meta.get('demo'),
            'profile':meta.get('profile'),
            'backend':meta.get('backend'),
            'status':meta.get('status'),
            'params':meta.get('params',{}),
            'frames':len(frames),
            'elapsed':meta.get('elapsed'),
            'created':meta.get('created') or d.stat().st_mtime,
            'thumb':f'/runs/{d.name}/{"reveal.jpg" if (d/"reveal.jpg").exists() else "frames/"+frames[-1].name}',
            'has_reveal':(d/'reveal.jpg').exists(),
            'zoom':meta.get('zoom'),
        })
        if len(out)>=max(1,min(500,limit)): break
    return out

@app.post('/api/run/{demo}')
def start(demo:str,req:RunReq):
    if demo not in DEMOS: raise HTTPException(404,'unknown demo')
    if req.profile not in load_profiles():
        raise HTTPException(422, 'unknown profile')
    spec_params=load_specs()[demo]['params']
    unknown=set(req.params)-set(spec_params)
    if unknown:
        raise HTTPException(422, f"unknown parameter(s): {', '.join(sorted(unknown))}")
    for name,value in req.params.items():
        limits=spec_params[name]
        if not math.isfinite(value) or not limits['min'] <= value <= limits['max']:
            raise HTTPException(422, f"{name} must be between {limits['min']} and {limits['max']}")
    rid=f'{demo}_{time.strftime("%Y%m%d_%H%M%S")}_{uuid.uuid4().hex[:5]}'; rd=RUNS/rid
    params=dict(req.params)
    if req.target_image is not None:
        if demo != 'neural_wall':
            raise HTTPException(422, 'a custom drawing is only supported by the neural-network wall')
        target_path=rd/'target.png'
        rd.mkdir(parents=True, exist_ok=True)
        save_target_image(req.target_image, target_path)
        params['_target_path']=str(target_path)
    def worker():
        try: run(demo,req.profile,req.frames,params,req.backend,rd)
        except Exception as e: print('run failed',e)
    threading.Thread(target=worker,daemon=True).start(); return {'id':rid}

ZOOMABLE={'crystal'}

# Deep-zoom levels are integers on a log2 ladder: level L means the model is
# rendered at 2**L tiles across. Quantising this way is what keeps magnification
# sharp. Rendering one image for the exact current view meant that between the
# render finishing and the next one arriving the browser was scaling a stale
# bitmap, so anything past a few hundred times looked like a blurred photograph.
MAX_ZOOM_LEVEL=40
# Tile generation is mostly Python tree traversal.  It needs processes (not
# threads) to use several CPU cores despite the GIL.  Eight workers give a
# fast first fill on a 64-GB desktop without reserving the whole machine.
TILE_WORKERS=max(2,min(8,(os.cpu_count() or 4)//2))
TILE_POOL=ProcessPoolExecutor(max_workers=TILE_WORKERS)
TILE_LOCK=threading.RLock()
TILE_PENDING={}
TILE_MEMORY=OrderedDict()
TILE_MEMORY_BYTES=0
try:
    TILE_MEMORY_LIMIT_MB=max(128,min(4096,int(os.getenv('LEONARDO_TILE_CACHE_MB','1024'))))
except ValueError:
    TILE_MEMORY_LIMIT_MB=1024
TILE_MEMORY_LIMIT=TILE_MEMORY_LIMIT_MB*1024*1024
DYNAMIC_TILE_VERSION=2


def _crystal_geometry(meta,cx,cy,span,size_px,budget):
    from leonardo_demos.crystal_growth import generate, MODE_NAMES
    p=meta.get('params',{}) or {}
    return generate(cx,cy,span,size_px=size_px,
                    symmetry=int(p.get('symmetry',6)),
                    mode=MODE_NAMES[int(p.get('mode',0))%len(MODE_NAMES)],
                    undercooling=float(p.get('undercooling',.75)),
                    anisotropy=float(p.get('anisotropy',.055)),
                    seed=int(p.get('seed',3)),
                    max_segments=budget)


def _render_crystal_tile_jpeg(meta,tcx,tcy,sub,tile,budget):
    """CPU-process worker: make one deterministic native-resolution tile."""
    from leonardo_demos.crystal_growth import render_window
    geometry=_crystal_geometry(meta,tcx,tcy,sub,tile,budget)
    image=render_window(geometry,tcx,tcy,sub,size=(tile,tile),progress=1.0,supersample=2)
    buf=io.BytesIO(); image.save(buf,'JPEG',quality=88)
    return buf.getvalue()


def _memory_tile(key):
    with TILE_LOCK:
        data=TILE_MEMORY.get(key)
        if data is not None:
            TILE_MEMORY.move_to_end(key)
        return data


def _remember_tile(key,data):
    global TILE_MEMORY_BYTES
    with TILE_LOCK:
        old=TILE_MEMORY.pop(key,None)
        if old is not None: TILE_MEMORY_BYTES-=len(old)
        TILE_MEMORY[key]=data; TILE_MEMORY_BYTES+=len(data)
        while TILE_MEMORY and TILE_MEMORY_BYTES>TILE_MEMORY_LIMIT:
            _,evicted=TILE_MEMORY.popitem(last=False)
            TILE_MEMORY_BYTES-=len(evicted)


def _tile_bytes(key,cached,meta,tcx,tcy,sub,tile,budget):
    """Return a tile from RAM/disk or join exactly one in-flight render."""
    hit=_memory_tile(key)
    if hit is not None: return hit
    with TILE_LOCK:
        future=TILE_PENDING.get(key)
        if future is None:
            if cached.exists():
                data=cached.read_bytes()
                _remember_tile(key,data)
                return data
            future=TILE_POOL.submit(_render_crystal_tile_jpeg,meta,tcx,tcy,sub,tile,budget)
            TILE_PENDING[key]=future
    try:
        data=future.result()
        cached.parent.mkdir(parents=True,exist_ok=True)
        temporary=cached.with_name(f'.{cached.stem}.{time.time_ns()}.tmp{cached.suffix}')
        temporary.write_bytes(data); temporary.replace(cached)
        _remember_tile(key,data)
        return data
    finally:
        with TILE_LOCK:
            if TILE_PENDING.get(key) is future: TILE_PENDING.pop(key,None)


def _run_dir(rid:str):
    rd=(RUNS/rid).resolve()
    if RUNS.resolve() not in rd.parents or not (rd/'meta.json').exists():
        raise HTTPException(404,'unknown run')
    return rd


@app.get('/api/zoom_tile/{rid}')
def zoom_tile(rid:str,level:int=0,col:int=0,row:int=0,tile:int=256):
    """One tile of the deep-zoom pyramid, rendered on demand and cached.

    Levels beyond the pre-baked ones are generated the first time they are
    asked for and written into the same runs/<id>/zoom/L<level>/ layout, so the
    baked pyramid and the on-demand tiles are one continuous structure and a
    revisited region is served straight from disk.
    """
    rd=_run_dir(rid)
    meta=json.loads((rd/'meta.json').read_text())
    if meta.get('demo') not in ZOOMABLE: raise HTTPException(404,'no deep zoom for this demo')
    level=max(0,min(MAX_ZOOM_LEVEL,int(level))); n=1<<level
    if not (0<=col<n and 0<=row<n): raise HTTPException(422,'tile out of range')
    tile=max(64,min(512,int(tile)))
    # Do not reuse tiles made by the pre-seam-fix renderer.  Baked levels are
    # still served directly from zoom/L*, while on-demand tiles live here.
    cached=rd/'zoom'/f'dynamic-v{DYNAMIC_TILE_VERSION}'/f'L{level}'/f'{col}_{row}.jpg'
    man=meta.get('zoom') or {}
    base_span=float(man.get('span') or 2.0); bcx=float(man.get('cx') or 0.0); bcy=float(man.get('cy') or 0.0)
    sub=base_span/n
    tcx=bcx-base_span/2+sub*(col+.5)
    tcy=bcy+base_span/2-sub*(row+.5)
    # Larger local budget prevents a dense visible branch from being cut off
    # midway through its detail.  Adjacent tiles share deterministic geometry,
    # so a segment crossing the seam is identical on both sides.
    budget=max(12000,int(meta.get('zoom_budget',12000)))
    key=f'v{DYNAMIC_TILE_VERSION}:{rid}:{level}:{col}:{row}:{tile}'
    data=_tile_bytes(key,cached,meta,tcx,tcy,sub,tile,budget)
    return Response(data,media_type='image/jpeg',
                    headers={'Cache-Control':'public, max-age=31536000, immutable'})


@app.get('/api/zoom/{rid}')
def zoom(rid:str,cx:float=0.0,cy:float=0.0,span:float=1.0,w:int=960,h:int=540):
    """Render one arbitrary window of a run's model, live.

    A pre-baked pyramid runs out of levels after a handful of doublings; past
    that the viewer can only upscale its deepest tiles, which is why deep zoom
    still looked like magnifying an image. Regenerating the geometry for the
    requested window has no depth limit at all.
    """
    rd=(RUNS/rid).resolve()
    if RUNS.resolve() not in rd.parents or not (rd/'meta.json').exists():
        raise HTTPException(404,'unknown run')
    meta=json.loads((rd/'meta.json').read_text())
    demo=meta.get('demo')
    if demo not in ZOOMABLE: raise HTTPException(404,'live zoom unavailable for this demo')
    if not (math.isfinite(cx) and math.isfinite(cy) and math.isfinite(span)) or span<=0:
        raise HTTPException(422,'bad window')
    w=max(64,min(1600,int(w))); h=max(64,min(1000,int(h)))
    from leonardo_demos.crystal_growth import generate, render_window, MODE_NAMES
    p=meta.get('params',{}) or {}
    g=generate(cx,cy,span,size_px=max(w,h),
               symmetry=int(p.get('symmetry',6)),
               mode=MODE_NAMES[int(p.get('mode',0))%len(MODE_NAMES)],
               undercooling=float(p.get('undercooling',.75)),
               anisotropy=float(p.get('anisotropy',.055)),
               seed=int(p.get('seed',3)),
               max_segments=int(meta.get('zoom_budget',22000)))
    im=render_window(g,cx,cy,span,size=(w,h),progress=1.0,supersample=2)
    buf=io.BytesIO(); im.save(buf,'JPEG',quality=86)
    return Response(buf.getvalue(),media_type='image/jpeg',
                    headers={'Cache-Control':'no-store'})

@app.get('/api/run/{rid}')
def status(rid:str):
    rd=RUNS/rid; p=rd/'meta.json'
    if not p.exists(): return {'status':'starting','frame':-1}
    return json.loads(p.read_text())

def _free_port(preferred=8000, tries=20):
    """First free port at or after `preferred`.

    Binding is not optional here: if 8000 is already held by an older viewer,
    starting silently fails and the browser keeps talking to that stale
    process. A stale server re-reads config/profiles.json from disk but still
    holds the previously imported demo modules, so every run dies with a
    confusing KeyError. Moving to a free port makes the new server reachable.
    """
    import socket
    for i in range(tries):
        port=preferred+i
        with socket.socket(socket.AF_INET,socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
            try:
                s.bind(('127.0.0.1',port)); return port,i>0
            except OSError:
                continue
    return preferred,False


if __name__=='__main__':
    import uvicorn, webbrowser
    port,moved=_free_port(8000)
    url=f'http://127.0.0.1:{port}'
    if moved:
        print('=' * 68)
        print(f'  Port 8000 is already in use by another program.')
        print(f'  Starting this viewer on {port} instead.')
        print(f'  If an older viewer window is still open, close it: a stale')
        print(f'  viewer runs old code against the current config and every')
        print(f'  demo it launches will fail.')
        print('=' * 68)
    print(f'Leonardo Visual Demos -> {url}')
    try: webbrowser.open(url)
    except Exception: pass
    uvicorn.run(app,host='127.0.0.1',port=port)
