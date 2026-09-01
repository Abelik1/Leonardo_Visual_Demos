from __future__ import annotations
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from ..base import Demo
from ..backend import to_numpy
from ..render import mosaic


class BlackHoleDemo(Demo):
    """Observer image plus a separately rendered 3-D photon-path view.

    The 2-D image uses a fast thin-lens mapping. The 3-D mode numerically
    advances photon directions through a weak-field point-lens acceleration,
    including a small signed frame-dragging term, and projects those spatial
    trajectories with a fixed camera. It is an exhibition model rather than a
    Kerr geodesic code, but the rays are computed 3-D curves, not Beziers.
    """
    timing_methods={"background":"initialization","lens":"simulation",
                    "integrate_rays":"simulation","render_3d":"render"}
    id="black_hole"; title="Black-hole lensing"

    def background(self,w,h,seed=8):
        rng=np.random.default_rng(seed)
        y,x=np.mgrid[0:h,0:w]; X=(x-w/2)/(h/2); Y=(y-h/2)/(h/2)
        r=np.sqrt((X+.42)**2+(Y+.05)**2); th=np.arctan2(Y+.05,X+.42)
        spiral=np.exp(-r*2.8)*(0.35+0.65*np.exp(2.2*np.cos(2*th-7.5*r)))
        rgb=np.zeros((h,w,3),dtype=np.float32)
        rgb[...,0]=.12*spiral+.75*spiral**1.7
        rgb[...,1]=.20*spiral+.55*spiral**1.4
        rgb[...,2]=.40*spiral+.80*spiral
        for _ in range(max(150,w*h//1600)):
            sx=rng.integers(0,w); sy=rng.integers(0,h); lum=rng.uniform(.4,1); rr=rng.choice([1,1,1,2])
            rgb[max(0,sy-rr):min(h,sy+rr+1),max(0,sx-rr):min(w,sx+rr+1)]+=lum
        return np.clip(rgb,0,1)

    def lens(self,bg,mass,spin,t=1.0):
        """Fast observer-plane thin-lens image; the companion mode is 3-D."""
        xp=self.ctx.xp; a=xp.asarray(bg)
        h,w=bg.shape[:2]; y,x=xp.mgrid[0:h,0:w]
        X=(x-w*.5)/(h*.5); Y=(y-h*.5)/(h*.5)
        r2=X*X+Y*Y+1e-4; strength=max(.05,float(t)); theta_e=.16*mass*strength
        bx=X-theta_e*theta_e*X/r2; by=Y-theta_e*theta_e*Y/r2
        shear=.025*spin*strength/(r2+.06)
        bx2=bx-shear*Y; by2=by+shear*X
        sx=xp.clip((bx2*(h*.5)+w*.5).astype(xp.int32),0,w-1)
        sy=xp.clip((by2*(h*.5)+h*.5).astype(xp.int32),0,h-1)
        out=a[sy,sx]; rr=xp.sqrt(r2)
        out=xp.where((rr < (.055+.018*mass))[...,None],0,out)
        ring=xp.exp(-((rr-(.105+.020*mass))/.018)**2)
        out=xp.clip(out+xp.stack([1.5*ring,.52*ring,.12*ring],axis=-1),0,1)
        return to_numpy(out)

    def integrate_rays(self,mass,spin,count=33,steps=360):
        """Advance spatial light rays through a weak-field lens equation."""
        impacts=[]
        for radius,n in ((.14,5),(.28,8),(.46,10),(.72,10)):
            for j in range(n):
                angle=2*math.pi*j/n+radius*1.7
                impacts.append((radius*math.cos(angle),radius*math.sin(angle)))
        impacts=np.asarray(impacts[:count],dtype=np.float64)
        pos=np.zeros((len(impacts),3),dtype=np.float64); pos[:,2]=-5.8
        vel=np.column_stack([impacts[:,0]/5.8,impacts[:,1]/5.8,np.ones(len(impacts))])
        vel/=np.linalg.norm(vel,axis=1)[:,None]
        paths=np.empty((steps,len(impacts),3),dtype=np.float32)
        alive=np.ones(len(impacts),dtype=bool); captured=np.zeros(len(impacts),dtype=bool)
        dt=.035; spin_axis=np.array([0.,0.,1.])
        for k in range(steps):
            paths[k]=pos
            radius=np.linalg.norm(pos,axis=1); unit=pos/np.maximum(radius[:,None],1e-7)
            transverse=unit-vel*np.sum(unit*vel,axis=1)[:,None]
            # Scene units are scaled so the source and observer fit on screen;
            # this coefficient keeps the outer bundles weakly deflected while
            # the innermost rays can still cross the capture radius.
            accel=-.045*mass*transverse/np.maximum(radius[:,None]**2+.055,1e-5)
            drag=.006*spin*np.cross(spin_axis[None,:],vel)/np.maximum(radius[:,None]**2+.12,.12)
            vel[alive]+=(accel[alive]+drag[alive])*dt
            vel[alive]/=np.maximum(np.linalg.norm(vel[alive],axis=1)[:,None],1e-8)
            pos[alive]+=vel[alive]*dt
            hit=(radius<(.13+.025*mass))&alive
            captured|=hit; alive&=~hit; vel[~alive]=0
        return paths,captured

    @staticmethod
    def project(points):
        points=np.asarray(points); yaw=math.radians(31); pitch=math.radians(-17)
        x,y,z=points[...,0],points[...,1],points[...,2]
        horizontal=z*math.cos(yaw)-x*math.sin(yaw)
        depth=x*math.cos(yaw)+z*math.sin(yaw)
        vertical=y*math.cos(pitch)-depth*math.sin(pitch)
        return np.stack([640+horizontal*91,360-vertical*91],axis=-1)

    def render_3d(self,paths,captured,mass,spin,progress):
        im=Image.new('RGB',(1280,720),(1,3,11)); d=ImageDraw.Draw(im,'RGBA')
        rng=np.random.default_rng(19)
        for _ in range(360):
            x=int(rng.uniform(0,1280)); y=int(rng.uniform(0,720)); lum=int(rng.uniform(45,180))
            d.ellipse((x,y,x+1,y+1),fill=(150,194,255,lum))
        plane=np.array([[-1.8,-1.25,5.5],[1.8,-1.25,5.5],[1.8,1.25,5.5],[-1.8,1.25,5.5]])
        pp=self.project(plane); d.polygon([tuple(p) for p in pp],fill=(17,31,64,125),outline=(90,181,255,160),width=2)
        for _ in range(75):
            star=np.array([[rng.uniform(-1.7,1.7),rng.uniform(-1.15,1.15),5.5]])
            sx,sy=self.project(star)[0]; rr=1 if rng.random()<.88 else 2
            d.ellipse((sx-rr,sy-rr,sx+rr,sy+rr),fill=(180,216,255,int(rng.uniform(90,235))))
        observer=self.project(np.array([[0,0,-5.8]]))[0]
        d.ellipse((observer[0]-11,observer[1]-11,observer[0]+11,observer[1]+11),fill=(83,240,216,240),outline=(214,255,249,255),width=2)
        visible=max(2,min(len(paths),int(progress*(len(paths)-1))+1))
        for j in range(paths.shape[1]):
            projected=self.project(paths[:visible,j]); colour=(255,116,92,130) if captured[j] else (91,221,255,145)
            d.line([tuple(q) for q in projected],fill=colour,width=2)
            packet=min(visible-1,int(((progress*1.8+j*.071)%1)*visible))
            px,py=projected[packet]; d.ellipse((px-4,py-4,px+4,py+4),fill=(255,246,183,245))
        centre=self.project(np.array([[0,0,0]]))[0]; radius=20+10*mass
        for k in range(7,0,-1):
            rr=radius*(1+.18*k); d.ellipse((centre[0]-rr,centre[1]-rr*.42,centre[0]+rr,centre[1]+rr*.42),outline=(255,112,54,10+k*8),width=2)
        d.ellipse((centre[0]-radius,centre[1]-radius,centre[0]+radius,centre[1]+radius),fill=(0,0,0,255),outline=(255,173,86,220),width=2)
        d.ellipse((centre[0]-radius*1.6,centre[1]-radius*.30,centre[0]+radius*1.6,centre[1]+radius*.30),outline=(255,98,42,225),width=4)
        return im

    def run(self):
        w=int(self.settings['width']); h=int(self.settings['height'])
        mass=float(self.ctx.params.get('mass',1.35)); spin=float(self.ctx.params.get('spin',.55))
        bg=self.background(w,h); lensed=self.lens(bg,mass,spin)
        observer=Image.fromarray((lensed*255).astype(np.uint8)).resize((1280,720),Image.Resampling.LANCZOS)
        observer=Image.blend(observer,observer.filter(ImageFilter.GaussianBlur(9)),.10)
        paths,captured=self.integrate_rays(mass,spin)
        self.ctx.write_meta({"view_modes":[{"id":"3d","label":"3D ray space","folder":"modes/3d"},{"id":"frames","label":"2D observer image","folder":"frames"}],"default_view_mode":"3d","ray_model":"numerically integrated weak-field 3D photon paths"})
        for i in range(self.ctx.frames):
            progress=(i+1)/self.ctx.frames
            self.ctx.save_frame(observer,self.ctx.frame_path(i))
            self.ctx.save_frame(self.render_3d(paths,captured,mass,spin,progress),self.ctx.run_dir/'modes'/'3d'/f'frame_{i:04d}.jpg')
            self.ctx.write_status(i,"Integrating photon paths through 3-D space",{
                "lens mass":f"{mass:.2f}","dimensionless spin":f"{spin:+.2f}","observer sight lines":f"{w*h:,}",
                "3D rays":f"{paths.shape[1]}","integration samples":f"{paths.shape[0]}","captured rays":f"{captured.sum()} / {len(captured)}"})
        ens=int(self.settings.get('ensemble',16)); side=max(2,int(math.sqrt(ens))); ims=[]; bg2=self.background(260,150)
        for j in range(side*side):
            m=max(.45,mass*(.65+0.7*(j%side)/max(1,side-1))); s=-.9+1.8*(j//side)/max(1,side-1)
            ims.append(Image.fromarray((self.lens(bg2,m,s)*255).astype(np.uint8)))
        rev=mosaic(ims,side); rp=self.ctx.run_dir/'reveal.jpg'; self.ctx.save_frame(rev,rp); self.ctx.finish(rp)
