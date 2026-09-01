from __future__ import annotations
from pathlib import Path
import io, json, math
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from .colors import palette, normalize

# The browser owns titles, readouts and legends.  Main and reveal images are
# scientific imagery only; this constant exists for a future legacy-export
# path, not for the interactive viewer.
BAKE_PRESENTATION = False


def font(size=26, bold=False):
    candidates=[
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
    ]
    for c in candidates:
        try: return ImageFont.truetype(c,size=size)
        except Exception: pass
    return ImageFont.load_default()


def array_image(a, kind="plasma", size=None):
    rgb=palette(a,kind)
    im=Image.fromarray(rgb,"RGB")
    if size: im=im.resize(size,Image.Resampling.BILINEAR)
    return im


def add_title(im, title, subtitle="", badge="LIVE COMPUTE"):
    if not BAKE_PRESENTATION:
        return im.convert("RGB")
    im=im.convert("RGB")
    d=ImageDraw.Draw(im,"RGBA")
    w,h=im.size
    d.rectangle((0,0,w,92),fill=(3,7,18,208))
    d.text((24,14),title,font=font(31,True),fill=(245,248,255,255))
    if subtitle:
        d.text((26,52),subtitle,font=font(16),fill=(182,198,226,255))
    if badge:
        tw=d.textbbox((0,0),badge,font=font(14,True))[2]
        d.rounded_rectangle((w-tw-44,20,w-20,57),radius=12,fill=(17,179,145,230))
        d.text((w-tw-32,29),badge,font=font(14,True),fill="white")
    return im


def add_progress(im, progress, left="COMPUTE", right="RESULT"):
    if not BAKE_PRESENTATION:
        return im
    d=ImageDraw.Draw(im,"RGBA"); w,h=im.size
    y=h-28
    d.rounded_rectangle((26,y,w-26,y+8),radius=4,fill=(255,255,255,35))
    d.rounded_rectangle((26,y,26+(w-52)*float(progress),y+8),radius=4,fill=(116,228,255,210))
    d.text((26,y-24),left,font=font(12,True),fill=(205,220,245,220))
    rb=d.textbbox((0,0),right,font=font(12,True)); d.text((w-26-(rb[2]-rb[0]),y-24),right,font=font(12,True),fill=(205,220,245,220))
    return im


def save_frame(im, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    im.save(path, quality=92)


def mosaic(images, cols, size=(1280,720), gap=8, title=None, top=None, subtitle=None,
           labels=None, label_fill=(255,255,255)):
    """Grid of thumbnails.

    `top` reserves vertical space for a header. Callers that paint their own
    banner afterwards (see add_title) must pass it, otherwise the banner is
    drawn straight over the first row of tiles.

    `labels` are drawn by the mosaic rather than baked into each thumbnail, so
    the text is sized against the *final* tile height and stays readable.
    """
    W,H=size
    rows=math.ceil(len(images)/cols)
    if top is None: top=86 if (title and BAKE_PRESENTATION) else 8
    tile_w=(W-gap*(cols+1))//cols
    tile_h=(H-top-gap*(rows+1))//rows
    out=Image.new("RGB",size,(3,6,15))
    for i,im in enumerate(images):
        r,c=divmod(i,cols)
        thumb=im.convert("RGB").resize((tile_w,tile_h),Image.Resampling.LANCZOS)
        out.paste(thumb,(gap+c*(tile_w+gap),top+r*(tile_h+gap)))
    if labels and BAKE_PRESENTATION:
        d=ImageDraw.Draw(out,"RGBA")
        fs=max(9,min(15,int(tile_h*.22)))
        band=fs+7
        if band < tile_h*.6:
            for i,text in enumerate(list(labels)[:len(images)]):
                if not text: continue
                r,c=divmod(i,cols)
                x=gap+c*(tile_w+gap); y=top+r*(tile_h+gap)
                d.rectangle((x,y+tile_h-band,x+tile_w,y+tile_h),fill=(2,5,15,205))
                d.text((x+4,y+tile_h-band+3),str(text),font=font(fs,True),fill=label_fill)
    if title and BAKE_PRESENTATION:
        d=ImageDraw.Draw(out,"RGBA")
        d.text((22,18),title,font=font(30,True),fill="white")
        d.text((24,54),subtitle or "The same question, explored in parallel.",font=font(16),fill=(180,203,235))
    return out


def scatter_canvas(xy, size=(1280,720), background=(2,5,13), point=(180,220,255), radius=1, glow=False):
    im=Image.new("RGB",size,background)
    d=ImageDraw.Draw(im,"RGBA")
    W,H=size
    pts=np.asarray(xy)
    if len(pts)==0: return im
    xs=((pts[:,0]+1)*0.5*(W-1)).astype(int)
    ys=((1-(pts[:,1]+1)*0.5)*(H-1)).astype(int)
    if glow:
        layer=Image.new("RGBA",size,(0,0,0,0)); ld=ImageDraw.Draw(layer,"RGBA")
        for x,y in zip(xs,ys): ld.ellipse((x-3,y-3,x+3,y+3),fill=(*point,70))
        layer=layer.filter(ImageFilter.GaussianBlur(4)); im=Image.alpha_composite(im.convert("RGBA"),layer).convert("RGB")
        d=ImageDraw.Draw(im,"RGBA")
    for x,y in zip(xs,ys):
        d.ellipse((x-radius,y-radius,x+radius,y+radius),fill=(*point,190))
    return im
