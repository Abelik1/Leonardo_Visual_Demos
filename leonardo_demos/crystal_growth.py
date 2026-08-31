"""Recursive crystal growth geometry and a resolution-independent renderer.

The previous crystal demo evaluated a fixed Gaussian distance field on the
simulation grid, so "zooming" could only ever enlarge pixels: there was no
detail underneath to find. Here the crystal is built once as a recursive set of
line segments in continuous coordinates, which can then be rasterised over an
arbitrary window at an arbitrary resolution. Zooming re-renders the geometry
instead of magnifying an image, and deeper levels legitimately expose finer
branch generations.
"""
from __future__ import annotations
import heapq
import itertools
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from .colors import palette

# Each mode is a different growth habit. Real snowflake morphology depends
# strongly on temperature and supersaturation (the Nakaya diagram), so offering
# only one six-fold dendrite made every run look identical.
MODES={
    "classic":  dict(branch_angle=32, fractions=(.30,.50,.70), ratio=.36, taper=.62, curl=0.0,  tip=.55, jitter=.00),
    "fern":     dict(branch_angle=44, fractions=(.22,.40,.58,.76), ratio=.42, taper=.58, curl=0.0, tip=.62, jitter=.02, one_sided=True),
    "seaweed":  dict(branch_angle=58, fractions=(.28,.55,.80), ratio=.50, taper=.70, curl=.22, tip=.40, jitter=.30),
    "star":     dict(branch_angle=18, fractions=(.62,), ratio=.26, taper=.50, curl=0.0, tip=.78, jitter=.00),
    "coral":    dict(branch_angle=68, fractions=(.20,.38,.56,.74,.90), ratio=.46, taper=.74, curl=.10, tip=.30, jitter=.18),
    "plate":    dict(branch_angle=60, fractions=(.88,), ratio=.92, taper=.96, curl=0.0, tip=.20, jitter=.00, facet=True),
}
MODE_NAMES=list(MODES)


def _mix64(value):
    """Stable 64-bit mixer used to make branching independent of tile order."""
    value=(value ^ (value >> 30)) * 0xbf58476d1ce4e5b9 & ((1 << 64)-1)
    value=(value ^ (value >> 27)) * 0x94d049bb133111eb & ((1 << 64)-1)
    return value ^ (value >> 31)


def _normal_from_key(key):
    """A deterministic N(0,1) sample for one branch path.

    A sequential RNG made jitter depend on the order in which the current
    tile's priority queue happened to visit branches.  Adjacent tiles therefore
    drew different versions of a branch crossing their seam.  Branch identity,
    rather than render order, is the only valid source of procedural noise.
    """
    a=(_mix64(key) >> 11) * (1.0 / (1 << 53))
    b=(_mix64(key ^ 0x9e3779b97f4a7c15) >> 11) * (1.0 / (1 << 53))
    return math.sqrt(-2.0*math.log(max(a,1e-15))) * math.cos(2.0*math.pi*b)


def _clip_segment(x0,y0,x1,y1,xmin,ymin,xmax,ymax):
    """Liang-Barsky clip of a segment to a rectangle. None if fully outside.

    At deep zoom a branch that merely crosses the tile starts far outside it,
    and scaling its endpoints produced pixel coordinates around 1e15, which
    overflows the C long PIL rasterises with. Clipping first keeps every
    coordinate on the order of the tile while drawing exactly the same line.
    """
    dx=x1-x0; dy=y1-y0
    t0,t1=0.0,1.0
    for p,q in ((-dx,x0-xmin),(dx,xmax-x0),(-dy,y0-ymin),(dy,ymax-y0)):
        if p==0.0:
            if q<0.0: return None
            continue
        r=q/p
        if p<0.0:
            if r>t1: return None
            if r>t0: t0=r
        else:
            if r<t0: return None
            if r<t1: t1=r
    return (x0+t0*dx,y0+t0*dy,x0+t1*dx,y0+t1*dy)


def _circle_hits_rect(cx,cy,r,x0,y0,x1,y1):
    nx=min(max(cx,x0),x1); ny=min(max(cy,y0),y1)
    dx=cx-nx; dy=cy-ny
    return dx*dx+dy*dy<=r*r


def _dist_point_segment(px,py,ax,ay,bx,by):
    vx=bx-ax; vy=by-ay
    L2=vx*vx+vy*vy
    if L2<=1e-30: return math.hypot(px-ax,py-ay)
    tt=((px-ax)*vx+(py-ay)*vy)/L2
    tt=0.0 if tt<0.0 else (1.0 if tt>1.0 else tt)
    return math.hypot(px-(ax+tt*vx),py-(ay+tt*vy))


# How far beyond its own segment a node's subtree can reach, as a multiple of
# the node's length. Descendants attach along the segment and extend outward by
# a converging cascade; carriers stay on the segment so they add nothing.
# Deliberately generous so pruning can never clip a branch that should show.
REACH=1.9


def generate(cx, cy, span, size_px=720, symmetry=6, mode="classic",
             undercooling=.75, anisotropy=.055, seed=3,
             max_segments=60000, max_depth=90):
    """Lazily generate the branches visible in a window, to whatever depth it takes.

    Two things make the zoom effectively unbounded:

    * **Scale-free branching.** Besides its side branches, every segment also
      spawns two half-length *carriers* covering its own two halves. Carriers
      draw nothing themselves; they exist so the same branching rule is applied
      again at half the scale, and again, so side branches appear at every
      scale along every segment. Without them a segment was a bare straight
      line between two branch points, and magnifying a point in that gap showed
      a straight line forever however deep the recursion went.

    * **Window pruning.** A node's subtree is bounded by a circle of radius
      REACH*length about its start, so a node whose circle misses the window is
      dropped with all of its descendants. Recursion stops once a branch is
      shorter than about a pixel, so the depth reached follows the zoom instead
      of being fixed in advance.
    """
    cfg=dict(MODES.get(mode,MODES["classic"]))
    branch_angle=math.radians(cfg["branch_angle"]+anisotropy*260.0)
    ratio=cfg["ratio"]; curl=cfg["curl"]
    fractions=cfg["fractions"]; jitter=cfg["jitter"]
    one_sided=cfg.get("one_sided",False)
    main_len=.30+.36*float(undercooling)
    # Slim branches: a fat branch fills the view within a couple of decades of
    # magnification and any further zoom is a blank white field.
    base_w=.0035+.0075*float(anisotropy)/.06
    # Width follows length, so every generation keeps the same proportions and
    # the structure looks identical at any magnification.
    aspect=base_w/main_len
    sym=max(2,int(symmetry))

    wpp=span/max(1,size_px)
    min_len=wpp*0.6
    half=span*.5
    x0,x1=cx-half,cx+half; y0,y1=cy-half,cy+half
    wcx,wcy=cx,cy
    wrad=half*math.sqrt(2.0)     # window circumradius

    segs=[]
    # Best-first by length. A LIFO stack dives depth-first into whichever arm
    # it happens to start with and spends the whole segment budget there, so a
    # truncated crystal comes out visibly lopsided. Always expanding the
    # largest remaining branch means the budget buys the most visible
    # structure first and truncation degrades detail evenly.
    tie=itertools.count()
    def key(x,y,angle,length):
        # Distance from the window to this node's own segment, clamped at zero
        # inside it. Ordering by (distance, -length) expands everything that
        # overlaps the view first, largest first, and otherwise walks toward the
        # view. Ordering by length alone spent the whole budget on large-scale
        # structure elsewhere in the crystal and never descended to the scale
        # actually on screen, so a deep zoom came out empty.
        ex=x+length*math.cos(angle); ey=y+length*math.sin(angle)
        d=_dist_point_segment(wcx,wcy,x,y,ex,ey)-wrad
        return (d if d>0.0 else 0.0,-length)
    stack=[]
    for k in range(sym):
        a=2*math.pi*k/sym
        node_key=_mix64(int(seed) ^ (k+1)*0x9e3779b97f4a7c15)
        stack.append((key(0.0,0.0,a,main_len),next(tie),0.0,0.0,a,main_len,0,True,node_key))
    heapq.heapify(stack)
    def push(x,y,angle,length,level,draw,node_key):
        heapq.heappush(stack,(key(x,y,angle,length),next(tie),x,y,angle,length,level,draw,node_key))
    while stack and len(segs)<max_segments:
        _,_,x,y,angle,length,level,draw,node_key=heapq.heappop(stack)
        if length<min_len or level>max_depth: continue
        ca=math.cos(angle); sa=math.sin(angle)
        ex=x+length*ca; ey=y+length*sa
        # Prune against the node's own SEGMENT, not a circle about its start.
        # A circle of radius REACH*length around the start point contains the
        # target for every ancestor as well as both carriers of every segment,
        # so the carrier chain survived pruning and multiplied as 2^depth.
        # Measuring to the segment keeps only the carrier that actually covers
        # the viewed region.
        if _dist_point_segment(wcx,wcy,x,y,ex,ey)-wrad > length*REACH: continue
        if draw:
            w=length*aspect
            if _circle_hits_rect((x+ex)*.5,(y+ey)*.5,length*.5+w,x0,y0,x1,y1):
                segs.append((x,y,ex,ey,w))
        for f_index,f in enumerate(fractions):
            bx=x+f*(ex-x); by=y+f*(ey-y)
            dirs=(1,) if one_sided else (-1,1)
            for d in dirs:
                child_key=_mix64(node_key ^ ((f_index+1)*0x517cc1b727220a95) ^
                                 ((d+2)*0x6eed0e9da4d94a4f))
                j=(_normal_from_key(child_key)*jitter if jitter else 0.0)
                push(bx,by,angle+d*branch_angle+curl*(level+1)+j,
                     length*ratio*(1-.18*f),level+1,True,child_key)
        if cfg["tip"]>0:
            tip_key=_mix64(node_key ^ 0x94d049bb133111eb)
            push(ex,ey,angle+curl,length*cfg["tip"]*ratio*1.6,level+1,True,tip_key)
        # Carriers: same line, half the scale, drawing nothing.
        h=length*.5
        push(x,y,angle,h,level+1,False,_mix64(node_key ^ 0x243f6a8885a308d3))
        push(x+h*ca,y+h*sa,angle,h,level+1,False,_mix64(node_key ^ 0x13198a2e03707344))

    if not segs: segs=[(0.0,0.0,0.0,0.0,main_len*aspect)]
    a=np.asarray(segs,dtype=np.float64)
    sx,sy,tx,ty,w=a[:,0],a[:,1],a[:,2],a[:,3],a[:,4]
    r=np.sqrt(sx*sx+sy*sy)
    birth=r/max(1e-9,r.max()) if r.max()>0 else np.zeros_like(r)
    return dict(x0=sx,y0=sy,x1=tx,y1=ty,w=w,birth=birth,
                extent=float(max(np.abs(np.concatenate([sx,tx,sy,ty])).max(),1e-6)))


def build_crystal(symmetry=6, depth=5, mode="classic", undercooling=.75, anisotropy=.055,
                  seed=3, max_segments=26000):
    """Whole-crystal geometry, from the same generator the deep zoom uses.

    `depth` is a detail level: it sets the effective resolution the generator
    culls against, so higher values resolve finer branch generations. Frames and
    zoom tiles must come from one generator, otherwise magnifying a frame would
    reveal a subtly different crystal.
    """
    main_len=.30+.36*float(undercooling)
    span=main_len*2.6
    size_px=int(min(6000,180*(2**max(0,int(depth)-3))))
    return generate(0.0,0.0,span,size_px=size_px,symmetry=symmetry,mode=mode,
                    undercooling=undercooling,anisotropy=anisotropy,seed=seed,
                    max_segments=max_segments)


def render_window(cr, cx, cy, span, size=(720,720), progress=1.0, supersample=2,
                  glow=True, kind="ice"):
    """Rasterise the crystal over the square world window centred at (cx,cy).

    `span` is the window's world width. Halving it doubles the magnification and
    thinner branch generations become resolvable, which is the whole point.
    """
    W,H=size; ss=max(1,int(supersample))
    Wf,Hf=W*ss,H*ss
    scale=Wf/max(1e-9,span)
    # `span` is the window width; the height follows the requested aspect so a
    # non-square panel shows more world rather than a stretched crystal.
    span_y=span*H/max(1,W)
    layer=Image.new("F",(Wf,Hf),0.0)
    d=ImageDraw.Draw(layer)
    x0,y0,x1,y1,w=cr["x0"],cr["y0"],cr["x1"],cr["y1"],cr["w"]
    if progress>=1.0:
        # Fully grown: draw every segment at full length. `birth` is normalised
        # against the radii present in whatever set was generated, so for a
        # windowed generation every segment sits near birth≈1 and the growth
        # interpolation below would draw them all as ~10% stubs — a deep zoom
        # rendered essentially empty.
        grown=np.ones(len(x0),dtype=bool)
        frac=np.ones(len(x0))
    else:
        grown=cr["birth"]<=progress
        # Partially extend the segments that are currently growing.
        frac=np.clip((progress-cr["birth"])/.16,0,1)
    ex=x0+(x1-x0)*frac; ey=y0+(y1-y0)*frac
    half=span*.5; half_y=span_y*.5
    lo_x,hi_x=cx-half,cx+half; lo_y,hi_y=cy-half_y,cy+half_y
    # Cull anything outside the window or thinner than a pixel; at deep zoom
    # this is what keeps the tile cost roughly constant.
    bminx=np.minimum(x0,ex)-w; bmaxx=np.maximum(x0,ex)+w
    bminy=np.minimum(y0,ey)-w; bmaxy=np.maximum(y0,ey)+w
    keep=grown&(bmaxx>=lo_x)&(bminx<=hi_x)&(bmaxy>=lo_y)&(bminy<=hi_y)
    idx=np.nonzero(keep)[0]
    for i in idx:
        px0=(x0[i]-lo_x)*scale; py0=(hi_y-y0[i])*scale
        px1=(ex[i]-lo_x)*scale; py1=(hi_y-ey[i])*scale
        # Width is capped as well: a branch far wider than the tile only ever
        # needs to cover it, and an unbounded value overflows the rasteriser.
        pw=min(max(1.0,w[i]*scale),4.0*max(Wf,Hf))
        pad=pw+2.0
        clipped=_clip_segment(px0,py0,px1,py1,-pad,-pad,Wf+pad,Hf+pad)
        if clipped is None: continue
        cx0,cy0,cx1,cy1=clipped
        d.line((cx0,cy0,cx1,cy1),fill=1.0,width=int(round(pw)))
        if -pad<=px0<=Wf+pad and -pad<=py0<=Hf+pad:
            d.ellipse((px0-pw/2,py0-pw/2,px0+pw/2,py0+pw/2),fill=1.0)
    arr=np.asarray(layer,dtype=np.float32)
    if ss>1:
        arr=arr.reshape(H,ss,W,ss).mean(axis=(1,3))
    field=arr
    if glow:
        soft=np.asarray(Image.fromarray((field*255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(3)),dtype=np.float32)/255.0
        field=np.clip(field+.55*soft,0,1)
    return Image.fromarray(palette(field,kind,normalize_input=False),"RGB")


def build_crystal_window(cx, cy, span, size_px=720, **kw):
    kw.pop("depth",None)
    return generate(cx,cy,span,size_px=size_px,**kw)
