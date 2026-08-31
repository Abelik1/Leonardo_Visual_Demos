from __future__ import annotations
import math
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from ..base import Demo
from ..backend import to_numpy
from ..render import add_title, add_progress, save_frame, mosaic, font

# Working units: kpc, km/s, solar masses. One time unit is then
# 1 kpc / (km/s) = 0.9778 Gyr, which keeps the displayed clock honest.
G=4.30091e-6           # kpc (km/s)^2 / Msun
TIME_UNIT_GYR=0.97779
# Plummer softening standing in for the extended dark-matter halo. The disc
# rotation curve is derived from this same potential; deriving it from a bare
# point mass instead gave inner stars ~2000 km/s and blew the discs apart.
EPS_GAL=30.0
M31_CATALOGUE=Path(__file__).resolve().parents[2]/"data"/"m31_catalog_reduced.npz"

# Published Local Group values. Masses are virial estimates and carry large
# uncertainties; the transverse velocity in particular is still debated, so
# these are presented as "a" plausible encounter, not "the" prediction.
# van der Marel et al. 2012 (ApJ 753, 8) and Gaia-era refinements.
MW_M31=dict(
    m1=1.5e12, m2=1.5e12,          # Msun, Milky Way and Andromeda
    separation=770.0,              # kpc
    v_radial=-109.0,               # km/s, approaching
    v_transverse=17.0,             # km/s
    r1=20.0, r2=30.0,              # disc scale radii, kpc
    label="Milky Way  ×  Andromeda (M31)",
    note="770 kpc apart, closing at 109 km/s",
)


class GalaxyCollisionDemo(Demo):
    id="galaxy_collision"; title="Galaxy collision"

    def setup(self,N,preset,impact,speed,tilt):
        """Return positions, velocities and the two galaxy centres."""
        if preset>=0.5:
            p=MW_M31
            m1,m2=p['m1'],p['m2']; sep=p['separation']
            vr,vt=p['v_radial'],p['v_transverse']
            r1,r2=p['r1'],p['r2']; label=p['label']; note=p['note']
        else:
            m1=1.5e12*max(.2,speed); m2=1.5e12*max(.2,2-speed)
            sep=400.0+500.0*impact
            vr=-60.0-160.0*speed; vt=10.0+120.0*impact
            r1,r2=20.0,26.0
            label="Custom encounter"; note=f"{sep:.0f} kpc apart, closing at {abs(vr):.0f} km/s"
        c1=np.array([-sep*.5,0.0]); c2=np.array([sep*.5,0.0])
        # Split the relative velocity between the two centres about the
        # centre of mass so the pair does not drift out of frame.
        mu1=m2/(m1+m2); mu2=m1/(m1+m2)
        # vrel is galaxy 2 relative to galaxy 1, and the separation vector
        # points along +x, so a negative radial component means approaching.
        # Splitting it about the centre of mass requires v2-v1 == vrel.
        vrel=np.array([vr,vt])
        vc1=-vrel*mu1; vc2=+vrel*mu2
        n1=N//2
        # Observationally motivated morphology: the Milky Way is drawn as a
        # four-arm disc, while M31 has two prominent arms plus its star-forming
        # ring. These are tracer-density structures, not literal catalog stars.
        p1,v1=self.disc(n1,c1,vc1,m1,r1,tilt,seed=2,arms=4,ring_radius=None)
        # A curator can place a reduced, deprojected M31 star catalogue in
        # data/m31_catalog_reduced.npz.  It then supplies the M31 tracer
        # positions; otherwise the explicitly labelled morphology model keeps
        # the demo self-contained.
        catalogued=self.catalogue_disc(N-n1,c2,vc2,m2,r2,seed=7)
        p2,v2=catalogued if catalogued is not None else self.disc(
            N-n1,c2,vc2,m2,r2,-tilt*.7,seed=7,arms=2,ring_radius=10.0)
        xp=self.ctx.xp
        return (xp.asarray(np.vstack([p1,p2]),dtype=xp.float32),
                xp.asarray(np.vstack([v1,v2]),dtype=xp.float32),
                xp.asarray(c1,dtype=xp.float32),xp.asarray(c2,dtype=xp.float32),
                xp.asarray(vc1,dtype=xp.float32),xp.asarray(vc2,dtype=xp.float32),
                m1,m2,label,note,sep)

    def disc(self,n,centre,vel,mass,scale,tilt,seed,arms=2,ring_radius=None):
        """A rotating exponential disc with visible arm/ring density structure."""
        rng=np.random.default_rng(seed)
        r=np.clip(rng.gamma(2.0,scale*.45,n),scale*.06,scale*2.6)
        a=rng.uniform(0,2*np.pi,n)
        # Most particles are concentrated around gently winding logarithmic-like
        # arms. The remaining diffuse particles retain a believable disc.
        arm_mask=rng.random(n)<.78
        arm_id=rng.integers(0,arms,n)
        winding=2*np.pi*1.28*r/(scale*2.6)
        a[arm_mask]=2*np.pi*arm_id[arm_mask]/arms+winding[arm_mask]+rng.normal(0,.20,n)[arm_mask]
        if ring_radius is not None:
            ring_mask=rng.random(n)<.24
            r[ring_mask]=np.clip(rng.normal(ring_radius,scale*.11,ring_mask.sum()),scale*.10,scale*2.6)
            a[ring_mask]=rng.uniform(0,2*np.pi,ring_mask.sum())
        ct=math.cos(math.radians(tilt))
        x=r*np.cos(a); y=r*np.sin(a)*ct
        pts=np.stack([x,y],axis=1)+np.asarray(centre)
        # Circular speed in the softened (Plummer) potential actually used by
        # accel(), so the discs start in equilibrium instead of flying apart.
        vcirc=r*np.sqrt(G*mass/np.power(r*r+EPS_GAL*EPS_GAL,1.5))
        tang=np.stack([-np.sin(a),np.cos(a)*ct],axis=1)
        disp=rng.normal(0,.06,(n,2))*vcirc[:,None]
        return pts,tang*vcirc[:,None]+disp+np.asarray(vel)

    def catalogue_disc(self,n,centre,vel,mass,scale,seed):
        """Sample a curator-reduced M31 catalogue, if one is installed.

        The reduction tool writes deprojected x/y positions in kpc and one
        representative mass/brightness per spatial cell.  Sampling from that
        distribution lets a large public catalogue preserve its actual arms
        and ring without pretending that every catalogue row is dynamically
        integrated.  The gravitational model remains the same restricted
        N-body approximation used by the synthetic fallback.
        """
        if not M31_CATALOGUE.exists():
            return None
        try:
            data=np.load(M31_CATALOGUE)
            xy=np.asarray(data['xy_kpc'],dtype=np.float64)
            weight=np.asarray(data.get('weight',np.ones(len(xy))),dtype=np.float64)
            good=np.isfinite(xy).all(axis=1)&np.isfinite(weight)&(weight>0)
            xy,weight=xy[good],weight[good]
            if len(xy)<32:
                return None
            rng=np.random.default_rng(seed)
            # Weighted resampling conserves the catalogue's luminous/mass
            # emphasis while keeping the number of dynamical bodies bounded.
            pick=rng.choice(len(xy),size=n,replace=len(xy)<n,p=weight/weight.sum())
            xy=xy[pick].copy()
            r=np.hypot(xy[:,0],xy[:,1])
            extent=max(float(np.percentile(r,95)),1e-3)
            xy*=scale*2.35/extent
            r=np.maximum(np.hypot(xy[:,0],xy[:,1]),scale*.04)
            a=np.arctan2(xy[:,1],xy[:,0])
            vcirc=r*np.sqrt(G*mass/np.power(r*r+EPS_GAL*EPS_GAL,1.5))
            tang=np.stack([-np.sin(a),np.cos(a)],axis=1)
            disp=rng.normal(0,.045,(n,2))*vcirc[:,None]
            return xy+np.asarray(centre),tang*vcirc[:,None]+disp+np.asarray(vel)
        except Exception:
            # A malformed optional asset must never stop exhibition playback.
            return None

    def accel(self,p,c1,c2,m1,m2,eps=EPS_GAL):
        xp=self.ctx.xp; a=xp.zeros_like(p)
        for c,m in ((c1,m1),(c2,m2)):
            d=c[None,:]-p; rr=xp.sum(d*d,axis=1)+eps*eps
            a+=G*m*d/(rr[:,None]**1.5)
        return a
    def centre_accel(self,c1,c2,m1,m2,eps=EPS_GAL):
        xp=self.ctx.xp; d=c2-c1; rr=float(xp.sum(d*d))+eps*eps
        a=G*d/(rr**1.5)
        return a*m2,-a*m1
    def step(self,p,v,c1,c2,vc1,vc2,m1,m2,dt,steps):
        for _ in range(steps):
            v+=dt*self.accel(p,c1,c2,m1,m2); p+=dt*v
            a1,a2=self.centre_accel(c1,c2,m1,m2)
            vc1+=dt*a1; vc2+=dt*a2; c1+=dt*vc1; c2+=dt*vc2
        return p,v,c1,c2,vc1,vc2

    def render(self,p,c1,c2,half,n1,m1,m2,size=(1280,720)):
        pts=to_numpy(p); W,H=size
        im=Image.new('RGB',size,(2,5,13))
        # Square world window mapped to a 16:9 frame without distortion.
        halfx=half*W/H
        x=((pts[:,0]+halfx)/(2*halfx)*W).astype(int)
        y=((1-(pts[:,1]+half)/(2*half))*H).astype(int)
        good=(x>=0)&(x<W)&(y>=0)&(y<H)
        origin=(np.arange(len(pts))>=n1)[good]
        flat=(y[good]*W+x[good]).astype(np.int64)
        # Each tracer represents an equal share of its parent galaxy mass.
        # Accumulating those shares is a surface-mass estimate, so bright areas
        # genuinely correspond to many overlapping tracer masses, not overdraw.
        mass=np.where(origin,m2/max(1,len(pts)-n1),m1/max(1,n1))
        d_mw=np.bincount(flat[~origin],weights=mass[~origin],minlength=W*H).reshape(H,W)
        d_m31=np.bincount(flat[origin],weights=mass[origin],minlength=W*H).reshape(H,W)
        def glow_field(field):
            nz=field[field>0]
            if not len(nz): return np.zeros_like(field,dtype=np.float32)
            ref=max(float(np.percentile(nz,99.5)),1.0)
            v=np.log1p(field/(ref*.05))/np.log1p(ref/(ref*.05))
            v=np.clip(v,0,1).astype(np.float32)
            soft=np.asarray(Image.fromarray((v*255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(2.2)),dtype=np.float32)/255
            return np.maximum(v*.68,soft)
        mw,m31=glow_field(d_mw),glow_field(d_m31)
        overlap=np.minimum(mw,m31)
        base=np.zeros((H,W,3),dtype=np.float32)+np.array([2,5,13],dtype=np.float32)
        # Blue-white Milky Way and warm Andromeda components remain visible as
        # they interpenetrate; overlap blooms toward white rather than hiding.
        base+=mw[...,None]*np.array([62,154,255],dtype=np.float32)
        base+=m31[...,None]*np.array([255,105,54],dtype=np.float32)
        base+=overlap[...,None]*np.array([130,150,110],dtype=np.float32)
        im=Image.fromarray(np.clip(base,0,255).astype(np.uint8),'RGB')
        layer=Image.new('RGBA',size,(0,0,0,0)); d=ImageDraw.Draw(layer,'RGBA')
        stride=max(1,len(flat)//60000)
        for xx,yy,is_m31 in zip(x[good][::stride],y[good][::stride],origin[::stride]):
            color=(255,184,120,170) if is_m31 else (178,222,255,170)
            d.ellipse((xx-1,yy-1,xx+1,yy+1),fill=color)
        im=Image.alpha_composite(im.convert('RGBA'),layer).convert('RGB')
        return im

    def run(self):
        N=int(self.settings['particles'])
        preset=float(self.ctx.params.get('preset',1))
        impact=float(self.ctx.params.get('impact',.55))
        speed=float(self.ctx.params.get('speed',.75))
        tilt=float(self.ctx.params.get('tilt',18))
        p,v,c1,c2,vc1,vc2,m1,m2,label,note,sep=self.setup(N,preset,impact,speed,tilt)
        # Integrate far enough to pass through the merger (~6 Gyr for MW/M31).
        span_gyr=float(self.settings.get('span_gyr',7.5))
        total=span_gyr/TIME_UNIT_GYR
        substeps=int(self.settings.get('substeps',6))
        dt=total/(self.ctx.frames*substeps)
        half=sep*.85
        for i in range(self.ctx.frames):
            p,v,c1,c2,vc1,vc2=self.step(p,v,c1,c2,vc1,vc2,m1,m2,dt,substeps)
            t_gyr=(i+1)/self.ctx.frames*span_gyr
            pts=to_numpy(p)
            # Track the encounter: the frame follows the shrinking separation
            # so the merger does not happen inside two pixels.
            # Radial percentile, not per-coordinate: taking |x| and |y|
            # together is dominated by the many small y values and framed the
            # pair too tightly, clipping both galaxies off the edges. The 90th
            # percentile still ignores the few ejected tracers.
            reach=float(np.percentile(np.hypot(pts[:,0],pts[:,1]),92))
            half=float(np.clip(.85*half+.15*reach*1.45,70.0,sep*1.6))
            im=self.render(p,c1,c2,half,N//2,m1,m2)
            d=ImageDraw.Draw(im,'RGBA')
            # A global physical view has to include the 770-kpc separation, so
            # the discs are necessarily small. These are independently rendered
            # close windows of the *same current particles*, not decorative art.
            morph_alpha=1.0-self._smoothstep(.24,.48,(i+1)/self.ctx.frames)
            if morph_alpha>0:
                for x0,centre,label in ((590,c1,"MILKY WAY · 4 ARMS"),(925,c2,"ANDROMEDA · 2 ARMS + RING")):
                    detail=self.render(p-centre,c1*0,c2*0,88,N//2,m1,m2,(300,175))
                    patch=im.crop((x0,112,x0+300,287))
                    im.paste(Image.blend(patch,detail,morph_alpha),(x0,112))
                    d.rectangle((x0,112,x0+300,287),outline=(137,228,255,int(220*morph_alpha)),width=2)
                    d.rectangle((x0,112,x0+300,137),fill=(4,10,24,int(220*morph_alpha)))
                    d.text((x0+10,119),label,font=font(11,True),fill=(225,242,255,int(255*morph_alpha)))
                d=ImageDraw.Draw(im,'RGBA')
            d.rounded_rectangle((26,112,556,252),radius=16,fill=(4,9,22,205))
            d.text((44,126),label,font=font(19,True),fill='white')
            d.text((44,156),note,font=font(15),fill=(196,214,240))
            d.text((44,180),f"t = +{t_gyr:.2f} Gyr from today",font=font(17,True),fill=(120,236,255))
            d.text((44,206),f"separation {float(np.hypot(*(to_numpy(c2)-to_numpy(c1)))):.0f} kpc",font=font(15),fill=(206,224,248))
            d.text((44,228),f"{N:,} tracers · brightness = mass per pixel · view {2*half:,.0f} kpc",font=font(14),fill=(150,170,200))
            im=add_title(im,"Galaxy collision",
                         f"restricted N-body · M₁={m1/1e12:.1f}×10¹² M☉ · M₂={m2/1e12:.1f}×10¹² M☉ · {self.ctx.backend_name}")
            add_progress(im,(i+1)/self.ctx.frames,"FIRST APPROACH","MERGER / TIDAL DEBRIS")
            self.ctx.save_frame(im,self.ctx.frame_path(i)); self.ctx.write_status(i,f"t=+{t_gyr:.2f} Gyr")
        # Reveal: the same encounter under the observational uncertainty on the
        # transverse velocity, which is what actually decides the outcome.
        side=4; ims=[]; labels=[]
        for j in range(side*side):
            vt=0.0+j*(80.0/(side*side-1))
            st=self.setup(max(2500,N//8),0,impact,speed,tilt)
            st=list(st)
            xp=self.ctx.xp
            st[4]=xp.asarray([MW_M31['v_radial']*.5,vt*.5],dtype=xp.float32)
            st[5]=xp.asarray([-MW_M31['v_radial']*.5,-vt*.5],dtype=xp.float32)
            pp,vv,cc1,cc2,vv1,vv2=st[0],st[1],st[2],st[3],st[4],st[5]
            for _ in range(max(10,self.ctx.frames//2)):
                pp,vv,cc1,cc2,vv1,vv2=self.step(pp,vv,cc1,cc2,vv1,vv2,st[6],st[7],dt*2,substeps)
            ims.append(self.render(pp,cc1,cc2,520,len(pp)//2,st[6],st[7],(260,146)))
            labels.append(f"v⊥ {vt:.0f} km/s")
        rev=mosaic(ims,side,title="There was never only one possible collision",
                   subtitle="The transverse velocity is measured to ±tens of km/s — and it decides the outcome.",
                   labels=labels,label_fill=(190,226,255))
        rp=self.ctx.run_dir/'reveal.jpg'; self.ctx.save_frame(rev,rp); self.ctx.finish(rp)

    @staticmethod
    def _smoothstep(a,b,x):
        x=np.clip((x-a)/(b-a),0,1)
        return x*x*(3-2*x)
