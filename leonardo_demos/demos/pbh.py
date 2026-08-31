from __future__ import annotations
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from ..base import Demo
from ..render import add_title, add_progress, save_frame, mosaic, font, palette

class PBHDemo(Demo):
    id="pbh"; title="Primordial black-hole threshold"
    delta_c=0.49774
    def profile(self,r,delta,width,t):
        """Reduced exhibition model, not the full Misner-Sharp research solver.

        It is designed to reproduce the qualitative narrative in the supplied
        capstone: subcritical perturbations disperse while supercritical ones
        localise and accelerate toward collapse. See docs/SCIENTIFIC_NOTES.md.
        """
        s=(delta-self.delta_c)/.018
        if s>=0:
            growth=np.exp(min(3.3,(0.4+3.8*s)*t*2.1))
            sig=max(.035,width*(.62/(1+2.6*t*(.3+s))))
            amp=1+5.5*max(.05,s+.12)*growth
            rho=1+amp*np.exp(-(r/sig)**2)
            comp=delta*(1+2.7*max(0,s)*t**1.7)*np.exp(-((r-.55*sig)/(.75*sig+.04))**2)
        else:
            damp=np.exp(-t*(.55+2.8*abs(s)))
            sig=width*(.55+.55*t)
            amp=5*delta*damp
            rho=1+amp*np.exp(-(r/sig)**2)-.32*amp*np.exp(-((r-1.5*sig)/(.65*sig+.05))**2)
            comp=delta*damp*np.exp(-((r-.55*width)/(width*.55))**2)
        return np.clip(rho,1e-4,None),comp
    def spherical(self,r,rho,size=720):
        y,x=np.mgrid[-1:1:complex(size),-1:1:complex(size)]
        R=np.sqrt(x*x+y*y)
        vals=np.interp(R,r,np.log10(np.maximum(rho,1e-3)),left=np.log10(rho[0]),right=0)
        rgb=palette(vals,'cosmic')
        alpha=np.exp(-R*R*2.4)[...,None]
        bg=np.zeros_like(rgb)+np.array([2,5,14],dtype=np.uint8)
        out=(bg*(1-alpha)+rgb*alpha).astype(np.uint8)
        return Image.fromarray(out)
    def run(self):
        n=int(self.settings['radial_points']); delta=float(self.ctx.params.get('delta',.505)); width=float(self.ctx.params.get('width',1.0)); r=np.linspace(0,4,n)
        regime="COLLAPSE" if delta>self.delta_c else "DISPERSION"
        for i in range(self.ctx.frames):
            t=(i+1)/self.ctx.frames
            rho,comp=self.profile(r,delta,width,t)
            sphere=self.spherical(r,rho,620).resize((720,720),Image.Resampling.LANCZOS)
            canvas=Image.new('RGB',(1280,720),(2,5,14)); canvas.paste(sphere,(0,0))
            d=ImageDraw.Draw(canvas,'RGBA')
            # scientific radial trace
            x0,y0,x1,y1=760,195,1230,565; d.rounded_rectangle((735,160,1250,595),radius=18,fill=(6,10,24,225))
            d.text((760,175),"Radial density contrast",font=font(18,True),fill='white')
            logr=np.log10(np.maximum(rho,1e-4)); lo,hi=min(0,float(logr.min())),max(.2,float(logr.max())); pts=[]
            for xx,v in zip(r,logr):
                px=x0+(x1-x0)*xx/r[-1]; py=y1-(y1-y0)*(v-lo)/(hi-lo+1e-9); pts.append((px,py))
            d.line(pts,fill=(102,231,255,235),width=3); d.line((x0,y1,x1,y1),fill=(255,255,255,60),width=1)
            d.rounded_rectangle((760,505,975,552),radius=12,fill=(235,72,86,210) if regime=='COLLAPSE' else (50,175,210,210)); d.text((780,518),regime,font=font(17,True),fill='white')
            canvas=add_title(canvas,"The early Universe: will it collapse?",f"δ={delta:.5f} · reference threshold ≈ {self.delta_c:.5f} · reduced exhibition model",badge="LIVE MODEL")
            add_progress(canvas,t,"PERTURBATION","COLLAPSE / DISPERSION")
            self.ctx.save_frame(canvas,self.ctx.frame_path(i)); self.ctx.write_status(i,regime)
        ens=int(self.settings.get('ensemble',64)); side=max(4,int(math.sqrt(ens))); ims=[]
        ds=np.linspace(self.delta_c-.025,self.delta_c+.025,side)
        ws=np.linspace(.65,1.35,side)
        for ww in ws:
            for dd in ds:
                rr=np.linspace(0,4,120); rho,_=self.profile(rr,float(dd),float(ww),1.0)
                im=self.spherical(rr,rho,150); dr=ImageDraw.Draw(im,'RGBA');
                col=(245,90,85,220) if dd>self.delta_c else (65,190,235,220); dr.rectangle((0,137,150,150),fill=col)
                ims.append(im)
        rev=mosaic(ims,side,title="Scientists do not run one universe")
        d=ImageDraw.Draw(rev,'RGBA'); d.text((880,25),"blue: disperses   red: collapses",font=font(17,True),fill=(220,235,255))
        rp=self.ctx.run_dir/'reveal.jpg'; self.ctx.save_frame(rev,rp); self.ctx.finish(rp)
