from __future__ import annotations
import math
import numpy as np
from PIL import Image, ImageDraw
from ..base import Demo
from ..render import add_title, add_progress, save_frame, mosaic, font
from ..crystal_growth import build_crystal, render_window, MODE_NAMES
from ..deepzoom import build_pyramid


class CrystalDemo(Demo):
    id="crystal"; title="Crystal growth"
    def geometry(self,depth=None):
        p=self.ctx.params
        mode=MODE_NAMES[int(p.get('mode',0))%len(MODE_NAMES)]
        return build_crystal(
            symmetry=int(p.get('symmetry',6)),
            depth=int(self.settings.get('depth',5) if depth is None else depth),
            mode=mode,
            undercooling=float(p.get('undercooling',.75)),
            anisotropy=float(p.get('anisotropy',.055)),
            seed=int(p.get('seed',3)),
        ),mode
    def frame(self,cr,progress,mode,i,frames):
        span=cr['extent']*2.35
        # The crystal is square; painting it into a square panel keeps the
        # six-fold symmetry six-fold instead of stretching it to 16:9.
        panel=render_window(cr,0,0,span,size=(700,700),progress=progress,supersample=2)
        im=Image.new('RGB',(1280,720),(2,6,16))
        im.paste(panel,(20,14))
        d=ImageDraw.Draw(im,'RGBA')
        p=self.ctx.params
        d.rounded_rectangle((748,120,1256,300),radius=16,fill=(5,10,24,215))
        d.text((770,138),"GROWTH HABIT",font=font(15,True),fill=(150,226,255))
        d.text((770,166),mode.upper(),font=font(30,True),fill='white')
        d.text((770,212),f"{int(p.get('symmetry',6))}-fold symmetry",font=font(17,True),fill=(206,226,250))
        d.text((770,238),f"undercooling {float(p.get('undercooling',.75)):.2f}",font=font(15),fill=(170,192,222))
        d.text((770,262),f"anisotropy {float(p.get('anisotropy',.055)):.3f}",font=font(15),fill=(170,192,222))
        d.rounded_rectangle((748,320,1256,440),radius=16,fill=(5,10,24,215))
        d.text((770,338),"Each branch grows the same rule again,",font=font(16,True),fill='white')
        d.text((770,362),"one scale smaller. Zoom in after the run",font=font(16,True),fill='white')
        d.text((770,386),"to resolve the deeper generations.",font=font(16,True),fill='white')
        d.text((770,412),f"{len(cr['x0']):,} branch segments",font=font(15,True),fill=(120,236,255))
        im=add_title(im,"Crystal growth",f"recursive anisotropic growth · habit: {mode} · {self.ctx.backend_name}")
        add_progress(im,(i+1)/frames,"MICROSCOPIC SEED","BRANCHED CRYSTAL")
        return im
    def run(self):
        frames=self.ctx.frames
        cr,mode=self.geometry()
        for i in range(frames):
            progress=(i+1)/frames
            im=self.frame(cr,progress,mode,i,frames)
            self.ctx.save_frame(im,self.ctx.frame_path(i))
            self.ctx.write_status(i,f"Growing {mode} habit")
        # Reveal: the same growth rule across habits and symmetries.
        ens=int(self.settings.get('ensemble',25)); side=max(2,int(math.sqrt(ens)))
        ims=[];labels=[]
        base=self.ctx.params
        for j in range(side*side):
            m=MODE_NAMES[j%len(MODE_NAMES)]
            sym=3+(j//len(MODE_NAMES))%10
            c=build_crystal(symmetry=sym,depth=max(3,int(self.settings.get('depth',5))-1),mode=m,
                            undercooling=float(base.get('undercooling',.75))*(.7+.6*((j%5)/4)),
                            anisotropy=float(base.get('anisotropy',.055)),seed=3+j)
            ims.append(render_window(c,0,0,c['extent']*2.3,size=(240,240),progress=1.0,supersample=1))
            labels.append(f"{m} · {sym}-fold")
        rev=mosaic(ims,side,title="One rule, many habits",
                   subtitle="Temperature and supersaturation pick the habit; the recursion is the same.",
                   labels=labels,label_fill=(190,226,255))
        rp=self.ctx.run_dir/'reveal.jpg'; self.ctx.save_frame(rev,rp)
        # Deep-zoom pyramid, rendered from the geometry at a higher recursion
        # depth than the frames use, so zooming finds branches that were never
        # resolvable in the 1280x720 frames.
        levels=int(self.settings.get('zoom_levels',4))
        if levels>0:
            deep,_=self.geometry(depth=int(self.settings.get('zoom_depth',7)))
            span=deep['extent']*2.35
            self.ctx.write_status(frames-1,"Baking deep-zoom tiles")
            manifest=build_pyramid(
                lambda cx,cy,sp,size: render_window(deep,cx,cy,sp,size=size,progress=1.0,supersample=2),
                self.ctx.run_dir/'zoom',cx=0,cy=0,span=span,
                levels=levels,tile=int(self.settings.get('zoom_tile',256)),
                progress_cb=lambda m,t: self.ctx.write_status(frames-1,f"Baking zoom tiles {m}/{t}"))
            self.ctx.write_meta({"zoom":manifest,"zoom_segments":int(len(deep['x0']))})
        self.ctx.finish(rp)
