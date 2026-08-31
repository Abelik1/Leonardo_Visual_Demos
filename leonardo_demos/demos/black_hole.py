from __future__ import annotations
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from ..base import Demo
from ..backend import to_numpy
from ..render import add_title, add_progress, save_frame, mosaic, font

class BlackHoleDemo(Demo):
    id="black_hole"; title="Black-hole lensing"
    @staticmethod
    def smoothstep(a,b,x):
        x=np.clip((x-a)/(b-a),0,1)
        return x*x*(3-2*x)
    def background(self,w,h,seed=8):
        rng=np.random.default_rng(seed)
        y,x=np.mgrid[0:h,0:w]; X=(x-w/2)/(h/2); Y=(y-h/2)/(h/2)
        # synthetic spiral galaxy + stars, so the repository is self-contained
        r=np.sqrt((X+.42)**2+(Y+.05)**2); th=np.arctan2(Y+.05,X+.42)
        spiral=np.exp(-r*2.8)*(0.35+0.65*np.exp(2.2*np.cos(2*th-7.5*r)))
        rgb=np.zeros((h,w,3),dtype=np.float32)
        rgb[...,0]=.12*spiral+.75*spiral**1.7; rgb[...,1]=.20*spiral+.55*spiral**1.4; rgb[...,2]=.40*spiral+.80*spiral
        for _ in range(max(150,w*h//1600)):
            sx=rng.integers(0,w); sy=rng.integers(0,h); lum=rng.uniform(.4,1); rr=rng.choice([1,1,1,2])
            y0=max(0,sy-rr);y1=min(h,sy+rr+1);x0=max(0,sx-rr);x1=min(w,sx+rr+1); rgb[y0:y1,x0:x1]+=lum
        return np.clip(rgb,0,1)
    def lens(self,bg,mass,spin,t):
        xp=self.ctx.xp
        a=xp.asarray(bg)
        h,w=bg.shape[:2]; y,x=xp.mgrid[0:h,0:w]
        X=(x-w*.5)/(h*.5); Y=(y-h*.5)/(h*.5)
        # Exhibition-grade thin-lens mapping plus a spin-dependent tangential shear.
        r2=X*X+Y*Y+1e-4; theta_e=.16*mass*(0.15+0.85*t)
        bx=X-theta_e*theta_e*X/r2; by=Y-theta_e*theta_e*Y/r2
        shear=.025*spin*(0.15+0.85*t)/(r2+.06)
        bx2=bx-shear*Y; by2=by+shear*X
        sx=xp.clip((bx2*(h*.5)+w*.5).astype(xp.int32),0,w-1)
        sy=xp.clip((by2*(h*.5)+h*.5).astype(xp.int32),0,h-1)
        out=a[sy,sx]
        rr=xp.sqrt(r2)
        shadow=rr < (.055+.018*mass)*(0.15+0.85*t)
        out=xp.where(shadow[...,None],0,out)
        # analytic emission ring evokes an accretion disk without pretending this is GRMHD
        ring=xp.exp(-((rr-(.105+.020*mass))/.018)**2)*(0.2+0.8*t)
        col=xp.stack([1.5*ring,.52*ring,.12*ring],axis=-1)
        out=xp.clip(out+col,0,1)
        return to_numpy(out)
    def source_sheet(self,d,cx,cy,scale,seed=8):
        """A perspective-like source plane for the explanatory, exploded view."""
        rng=np.random.default_rng(seed)
        w,h=270*scale,184*scale
        corners=[(cx-w/2,cy-h/2+13*scale),(cx+w/2,cy-h/2-13*scale),
                 (cx+w/2,cy+h/2-13*scale),(cx-w/2,cy+h/2+13*scale)]
        d.polygon(corners,fill=(8,19,44,225),outline=(76,185,255,225),width=2)
        # A self-contained background galaxy: distant points and two spiral arms.
        for _ in range(170):
            u,v=rng.uniform(-.47,.47),rng.uniform(-.43,.43)
            x=cx+u*w; y=cy+v*h-u*26*scale
            lum=int(rng.uniform(80,240)); r=1 if lum<190 else 2
            d.ellipse((x-r,y-r,x+r,y+r),fill=(150,198,255,lum))
        for arm in (0,math.pi):
            for r in np.linspace(.03,.46,52):
                th=arm+8.2*r
                x=cx+math.cos(th)*r*w; y=cy+math.sin(th)*r*h-u*0
                d.ellipse((x-2,y-2,x+2,y+2),fill=(255,176,106,170))
        d.text((cx-w/2+12,cy-h/2+18),"DISTANT GALAXY",font=font(12,True),fill=(180,222,255))
    def observer_plane(self,d,cx,cy,scale):
        w,h=178*scale,250*scale
        corners=[(cx-w/2,cy-h/2),(cx+w/2,cy-h/2+15*scale),
                 (cx+w/2,cy+h/2+15*scale),(cx-w/2,cy+h/2)]
        d.polygon(corners,fill=(5,18,36,235),outline=(91,235,222,235),width=2)
        for q in (-.25,0,.25):
            d.line((cx-w/2,cy+q*h,cx+w/2,cy+q*h+15*scale),fill=(90,203,230,70),width=1)
        for q in (-.25,0,.25):
            d.line((cx+q*w,cy-h/2,cx+q*w,cy+h/2+15*scale),fill=(90,203,230,70),width=1)
        d.text((cx-w/2+12,cy-h/2+14),"OBSERVER IMAGE",font=font(11,True),fill=(139,245,224))
    def lens_object(self,d,cx,cy,mass,scale=1.0):
        r=(34+11*mass)*scale
        for k in range(7,0,-1):
            rr=r*(.42+.1*k); alpha=int(8+8*k)
            d.ellipse((cx-rr,cy-rr*.48,cx+rr,cy+rr*.48),outline=(255,138,67,alpha),width=2)
        d.ellipse((cx-r,cy-r,cx+r,cy+r),fill=(1,2,7,255),outline=(255,178,83,205),width=2)
        d.ellipse((cx-r*.72,cy-r*.21,cx+r*.72,cy+r*.21),outline=(255,112,50,230),width=3)
        d.ellipse((cx-r*.34,cy-r*.34,cx+r*.34,cy+r*.34),fill=(0,0,0,255))
        d.text((cx-r-20,cy+r+17),"GRAVITATIONAL LENS",font=font(12,True),fill=(255,191,128))
    @staticmethod
    def curve(points,steps=30):
        p0,p1,p2,p3=[np.asarray(p,dtype=float) for p in points]
        out=[]
        for u in np.linspace(0,1,steps):
            q=(1-u)**3*p0+3*(1-u)**2*u*p1+3*(1-u)*u*u*p2+u**3*p3
            out.append(tuple(q))
        return out
    def exploded_view(self,mass,spin,progress):
        """Fixed camera, exploded light-path diagram. It explains — not replaces — lens()."""
        im=Image.new('RGB',(1280,720),(2,6,17)); d=ImageDraw.Draw(im,'RGBA')
        # Layers slide together along the viewing axis before becoming the 2-D image.
        gather=self.smoothstep(.12,.86,progress)
        source_x=1015*(1-gather)+646*gather
        observer_x=250*(1-gather)+646*gather
        dust_x=105*(1-gather)+646*gather
        lens_x=646
        # Deep-space depth cues.
        for r,a in ((420,12),(280,16),(145,26)):
            d.ellipse((lens_x-r,360-r,lens_x+r,360+r),outline=(60,130,255,a),width=1)
        self.source_sheet(d,source_x,294,.96)
        self.observer_plane(d,observer_x,386,.90)
        # A foreground dust veil gives the viewer a third spatial layer.
        rng=np.random.default_rng(21)
        for _ in range(70):
            x=dust_x+rng.normal(0,54); y=360+rng.normal(0,140); rr=rng.choice([1,1,2,3])
            d.ellipse((x-rr,y-rr,x+rr,y+rr),fill=(112,149,202,70))
        d.text((dust_x-70,535),"FOREGR. DUST",font=font(11,True),fill=(152,179,214))
        # Curved paths are sampled rays from the source sheet. A bright bead travels
        # along each one, which makes the otherwise static ray tracing legible.
        ray_phase=(progress*2.15)%1
        for j in range(-3,4):
            sy=292+j*22; bend=360+j*34+spin*j*5
            ey=388-j*14
            pts=[(source_x-116,sy),(lens_x+120,bend-j*13),(lens_x-100,bend+j*18),(observer_x+76,ey)]
            path=self.curve(pts)
            d.line(path,fill=(94,220,255,108),width=2)
            k=min(len(path)-1,int(((ray_phase+j*.13)%1)*(len(path)-1)))
            x,y=path[k]; d.ellipse((x-5,y-5,x+5,y+5),fill=(255,238,172,245))
        self.lens_object(d,lens_x,360,mass)
        # Axis line and layer call-outs turn the visual into a readable assembly.
        d.line((125,620,1155,620),fill=(104,180,235,80),width=2)
        for x,label,col in ((dust_x,'FOREGROUND',(150,181,220)),(lens_x,'LENS',(255,183,115)),
                            (source_x,'SOURCES',(156,220,255)),(observer_x,'OBSERVER',(139,245,224))):
            d.ellipse((x-4,616,x+4,624),fill=(*col,245)); d.text((x-38,634),label,font=font(10,True),fill=(*col,230))
        d.rounded_rectangle((30,112,510,170),radius=14,fill=(3,10,25,214))
        text="Separate layers show where each ray comes from" if gather<.65 else "The layers compact into one observer image"
        d.text((48,126),text,font=font(16,True),fill='white')
        d.text((48,148),"sources  →  curved paths around the lens  →  detector pixels",font=font(12),fill=(167,195,229))
        return im
    def run(self):
        w=int(self.settings['width']); h=int(self.settings['height']); mass=float(self.ctx.params.get('mass',1.35)); spin=float(self.ctx.params.get('spin',.55))
        bg=self.background(w,h)
        for i in range(self.ctx.frames):
            t=(i+1)/self.ctx.frames
            # Keep the actual pixel-parallel lensing calculation as the result. The
            # exploded scene is a separate explanatory render that dissolves into it.
            arr=self.lens(bg,mass,spin,max(.16,(t-.38)/.62))
            lensed=Image.fromarray((arr*255).astype(np.uint8)).resize((1280,720),Image.Resampling.LANCZOS)
            glow=lensed.filter(ImageFilter.GaussianBlur(9)); lensed=Image.blend(lensed,glow,.10)
            merge=self.smoothstep(.48,.78,t)
            im=Image.blend(self.exploded_view(mass,spin,t/.78),lensed,merge) if merge<1 else lensed
            if merge<.98:
                im=add_title(im,"How a lensed image is assembled",f"exploded light-path view · {w*h:,} eventual observer sight lines",badge="LAYER VIEW")
            else:
                im=add_title(im,"Black-hole lensing",f"image-space light mapping · mass={mass:.2f} · spin={spin:.2f} · {w*h:,} sight lines")
            d=ImageDraw.Draw(im,'RGBA')
            if merge>=.98:
                d.rounded_rectangle((26,112,390,176),radius=14,fill=(3,7,17,190)); d.text((44,125),"Every pixel asks: where did this light originate?",font=font(15,True),fill='white')
            add_progress(im,t,"LAYERED LIGHT PATH","LENSED OBSERVER")
            self.ctx.save_frame(im,self.ctx.frame_path(i)); self.ctx.write_status(i,"Tracing image-space rays" if merge>.7 else "Assembling the light-path layers")
        # Actual distinct observer positions/Einstein radii, small images.
        ens=int(self.settings.get('ensemble',16)); side=max(2,int(math.sqrt(ens))); ims=[]
        bg2=self.background(260,150)
        for j in range(side*side):
            m=max(.45,mass*(.65+0.7*(j%side)/(max(1,side-1))))
            s=-.9+1.8*(j//side)/(max(1,side-1))
            arr=self.lens(bg2,m,s,1.0); ims.append(Image.fromarray((arr*255).astype(np.uint8)))
        rev=mosaic(ims,side,title="That was one observer. Leonardo can explore many.")
        d=ImageDraw.Draw(rev,'RGBA'); d.text((910,28),f"{side*side} independent views",font=font(17,True),fill=(124,234,255))
        rp=self.ctx.run_dir/'reveal.jpg'; self.ctx.save_frame(rev,rp); self.ctx.finish(rp)
