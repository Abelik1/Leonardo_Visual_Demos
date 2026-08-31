from __future__ import annotations
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from ..base import Demo
from ..backend import to_numpy
from ..render import add_title, add_progress, save_frame, font
from ..colors import palette

W,H=1280,720


class FluidDemo(Demo):
    id="fluid"; title="Virtual wind tunnel"
    def init(self,nx,ny,u0):
        xp=self.ctx.xp
        # D2Q9 LBM
        c=xp.asarray([[0,0],[1,0],[0,1],[-1,0],[0,-1],[1,1],[-1,1],[-1,-1],[1,-1]],dtype=xp.int32)
        w=xp.asarray([4/9,1/9,1/9,1/9,1/9,1/36,1/36,1/36,1/36],dtype=xp.float32)
        rho=xp.ones((ny,nx),dtype=xp.float32); ux=xp.full((ny,nx),u0,dtype=xp.float32); uy=xp.zeros((ny,nx),dtype=xp.float32)
        f=xp.empty((9,ny,nx),dtype=xp.float32)
        # A little transverse noise seeds the wake instability. A perfectly
        # uniform inlet leaves the flow symmetric, so the twin standing eddies
        # never break down into an alternating street.
        rng=np.random.default_rng(5)
        uy=uy+xp.asarray(rng.normal(0,u0*.05,(ny,nx)).astype(np.float32))
        usq=ux*ux+uy*uy
        for q in range(9):
            cu=3*(c[q,0]*ux+c[q,1]*uy); f[q]=w[q]*rho*(1+cu+.5*cu*cu-1.5*usq)
        # A half-cell vertical offset breaks the lattice-symmetric stagnation
        # point. Without it the wake stays perfectly symmetric for a very long
        # time and the von Karman street never appears.
        yy,xx=xp.mgrid[0:ny,0:nx]
        self.obstacle=(float(nx)*.28,float(ny)*.5+.5,float(ny)*.15)
        ox,oy,orad=self.obstacle
        mask=(xx-ox)**2+(yy-oy)**2 < orad*orad
        return f,c,w,mask
    def step(self,f,c,w,mask,u0,steps):
        xp=self.ctx.xp; tau=.57; omega=1/tau
        opposite=[0,3,4,1,2,7,8,5,6]
        for _ in range(steps):
            rho=xp.sum(f,axis=0); ux=xp.sum(f*c[:,0,None,None],axis=0)/(rho+1e-8); uy=xp.sum(f*c[:,1,None,None],axis=0)/(rho+1e-8)
            ux[:,0]=u0; uy[:,0]=0
            usq=ux*ux+uy*uy
            for q in range(9):
                cu=3*(c[q,0]*ux+c[q,1]*uy); feq=w[q]*rho*(1+cu+.5*cu*cu-1.5*usq); f[q]+=omega*(feq-f[q])
            for q in range(9): f[q]=xp.roll(xp.roll(f[q],int(c[q,1]),axis=0),int(c[q,0]),axis=1)
            old=f.copy()
            for q in range(9): f[q][mask]=old[opposite[q]][mask]
        rho=xp.sum(f,axis=0); ux=xp.sum(f*c[:,0,None,None],axis=0)/(rho+1e-8); uy=xp.sum(f*c[:,1,None,None],axis=0)/(rho+1e-8)
        ux=xp.where(mask,0,ux); uy=xp.where(mask,0,uy)
        vort=xp.roll(uy,-1,1)-xp.roll(uy,1,1) - (xp.roll(ux,-1,0)-xp.roll(ux,1,0))
        return f,ux,uy,vort,rho

    # ---- tracers -------------------------------------------------------
    def seed_tracers(self,n,nx,ny,rng):
        x=rng.uniform(0,nx,n); y=rng.uniform(0,ny,n)
        return np.stack([x,y],axis=1)
    def sample(self,field,pts,nx,ny):
        """Bilinear sample of a lattice field at particle positions."""
        x=np.clip(pts[:,0],0,nx-1.001); y=np.clip(pts[:,1],0,ny-1.001)
        x0=x.astype(np.int32); y0=y.astype(np.int32)
        x1=np.minimum(x0+1,nx-1); y1=np.minimum(y0+1,ny-1)
        fx=x-x0; fy=y-y0
        return (field[y0,x0]*(1-fx)*(1-fy)+field[y0,x1]*fx*(1-fy)
                +field[y1,x0]*(1-fx)*fy+field[y1,x1]*fx*fy)
    def advect(self,trail,ux,uy,nx,ny,rng,dt,substeps=4):
        """Advance the tracers and push the new position onto their trail.

        A single position per particle produced streaks under a pixel long at
        these lattice speeds, so the frame showed a static dot field. Carrying a
        short history and drawing it as a polyline turns each particle into a
        visible streakline, which is what makes the flow direction readable in a
        still frame.
        """
        pts=trail[:,-1,:].copy()
        for _ in range(substeps):
            vx=self.sample(ux,pts,nx,ny); vy=self.sample(uy,pts,nx,ny)
            pts=pts+np.stack([vx,vy],axis=1)*(dt/substeps)
        ox,oy,orad=self.obstacle
        inside=(pts[:,0]-ox)**2+(pts[:,1]-oy)**2 < orad*orad
        stalled=np.hypot(pts[:,0]-trail[:,-1,0],pts[:,1]-trail[:,-1,1])<1e-3
        gone=(pts[:,0]>=nx-1)|(pts[:,0]<0)|(pts[:,1]<0)|(pts[:,1]>=ny-1)|inside|stalled
        trail=np.concatenate([trail[:,1:,:],pts[:,None,:]],axis=1)
        k=int(gone.sum())
        if k:
            # Reinject at the inlet and collapse the trail so no streak is drawn
            # spanning the whole domain.
            nx0=rng.uniform(0,3.0,k); ny0=rng.uniform(0,ny-1,k)
            trail[gone]=np.stack([nx0,ny0],axis=1)[:,None,:]
        return trail

    # ---- rendering -----------------------------------------------------
    def render(self,ux,uy,vort,rho,trail,nx,ny,speed):
        sx,sy=W/nx,H/ny
        spd=np.sqrt(ux*ux+uy*uy)
        # Background is flow SPEED on a cool palette. The old frame showed
        # |vorticity| on a fire palette, which read as "this region is hot"
        # rather than "the air is moving here".
        bg=palette(np.clip(spd/max(1e-6,speed*1.9),0,1),'ice')
        im=Image.fromarray(bg,'RGB').resize((W,H),Image.Resampling.BILINEAR)
        # Vorticity tints the wake so shed vortices stay readable.
        v=np.abs(to_numpy(vort)); v=np.clip(np.log1p(v*220)/3.2,0,1)
        tint=Image.fromarray((np.stack([v*255,v*90,v*30],-1)).astype(np.uint8),'RGB').resize((W,H),Image.Resampling.BILINEAR)
        im=Image.blend(im,tint,.20)
        d=ImageDraw.Draw(im,'RGBA')
        # Streaklines: the tail of each tracer's path, brightening toward the
        # head, so a single still frame already shows which way the air goes.
        K=trail.shape[1]
        for k in range(1,K):
            a=int(38+150*(k/(K-1))**1.7); wdt=1 if k<K*.6 else 2
            seg=np.stack([trail[:,k-1,:],trail[:,k,:]],axis=1)
            jump=np.hypot(seg[:,1,0]-seg[:,0,0],seg[:,1,1]-seg[:,0,1])>nx*.25
            for (p0,p1),bad in zip(seg,jump):
                if bad: continue
                d.line((p0[0]*sx,p0[1]*sy,p1[0]*sx,p1[1]*sy),fill=(214,244,255,a),width=wdt)
        for cx,cy in trail[:,-1,:]:
            d.ellipse((cx*sx-1.7,cy*sy-1.7,cx*sx+1.7,cy*sy+1.7),fill=(255,255,255,230))
        # Direction glyphs on a coarse grid.
        stepx=max(1,nx//26); stepy=max(1,ny//14)
        for j in range(stepy//2,ny,stepy):
            for i in range(stepx//2,nx,stepx):
                vx,vy=float(ux[j,i]),float(uy[j,i])
                m=math.hypot(vx,vy)
                if m<speed*.09: continue
                L=min(26,7+m/max(1e-6,speed)*13)
                x0,y0=i*sx,j*sy; dx,dy=vx/m*L,vy/m*L
                a=int(90+120*min(1,m/max(1e-6,speed*1.5)))
                d.line((x0-dx*.5,y0-dy*.5,x0+dx*.5,y0+dy*.5),fill=(150,225,255,a),width=2)
                ang=math.atan2(dy,dx)
                for s in (2.5,-2.5):
                    d.line((x0+dx*.5,y0+dy*.5,
                            x0+dx*.5-6*math.cos(ang+s*.4),y0+dy*.5-6*math.sin(ang+s*.4)),
                           fill=(180,236,255,a),width=2)
        ox,oy,orad=self.obstacle
        d.ellipse((ox*sx-orad*sx,oy*sy-orad*sy,ox*sx+orad*sx,oy*sy+orad*sy),
                  fill=(7,12,24,252),outline=(190,232,255,220),width=3)
        return im,spd
    def panel(self,im,ux,uy,rho,nx,ny,speed,mach_note):
        d=ImageDraw.Draw(im,'RGBA')
        ox,oy,orad=self.obstacle
        # Pressure from the LBM density: p = rho * cs^2, cs^2 = 1/3.
        p=(rho-1.0)/3.0
        front=float(p[int(oy),max(0,int(ox-orad-2))])
        back=float(p[int(oy),min(nx-1,int(ox+orad+2))])
        d.rounded_rectangle((26,112,470,246),radius=16,fill=(4,9,22,205))
        d.text((44,126),"AIRFLOW  →  left to right",font=font(17,True),fill=(160,230,255))
        d.text((44,154),f"inlet speed {speed:.3f} lattice units",font=font(15),fill=(206,224,248))
        d.text((44,178),f"stagnation pressure (front)  {front:+.5f}",font=font(15),fill=(255,206,150))
        d.text((44,200),f"wake pressure (behind)       {back:+.5f}",font=font(15),fill=(150,226,255))
        d.text((44,222),mach_note,font=font(14),fill=(150,170,200))
        return im
    def budget(self):
        """Total lattice updates for the run, independent of frame count.

        Vortex shedding needs a few thousand updates to grow out of the initial
        transient; a frame-derived budget stopped at a couple of hundred, so the
        wake was still perfectly steady when the run ended.
        """
        s=self.settings
        if 'total_steps' in s: return max(1,int(s['total_steps']))
        return max(1,int(s.get('steps_per_frame',6))*self.ctx.frames)
    def run(self):
        nx,ny=int(self.settings['nx']),int(self.settings['ny'])
        speed=float(self.ctx.params.get('speed',.06))
        total=self.budget(); done=0
        f,c,w,mask=self.init(nx,ny,speed)
        rng=np.random.default_rng(11)
        ntr=int(self.settings.get('tracers',900))
        K=int(self.settings.get('trail',12))
        trail=np.repeat(self.seed_tracers(ntr,nx,ny,rng)[:,None,:],K,axis=1)
        # Tracers are a visualisation of the same velocity field, advanced with
        # an amplified visual time step so the streaks span useful distances.
        boost=float(self.settings.get('tracer_boost',9.0))
        for i in range(self.ctx.frames):
            step_target=int(round(total*(i+1)/self.ctx.frames))
            spf=max(1,step_target-done); done=step_target
            f,ux,uy,vort,rho=self.step(f,c,w,mask,speed,spf)
            uxn,uyn,rhon=to_numpy(ux),to_numpy(uy),to_numpy(rho)
            # Several small advection sub-steps keep streaks smooth and stop
            # particles tunnelling through the cylinder.
            trail=self.advect(trail,uxn,uyn,nx,ny,rng,min(spf,14)*boost)
            im,spd=self.render(uxn,uyn,vort,rhon,trail,nx,ny,speed)
            re=speed*(2*self.obstacle[2])/((.57-.5)/3)
            im=self.panel(im,uxn,uyn,rhon,nx,ny,speed,f"Reynolds number ≈ {re:,.0f}  ·  {nx*ny:,} lattice cells")
            im=add_title(im,"Virtual wind tunnel",f"D2Q9 lattice-Boltzmann · {nx}×{ny} cells · {self.ctx.backend_name}")
            add_progress(im,(i+1)/self.ctx.frames,"LAMINAR START","VORTEX WAKE")
            self.ctx.save_frame(im,self.ctx.frame_path(i)); self.ctx.write_status(i,"Updating lattice cells")
        rev=im.copy(); d=ImageDraw.Draw(rev,'RGBA')
        cols,rows=4,2
        for r in range(rows):
            for cc in range(cols):
                x0=cc*W/cols; x1=(cc+1)*W/cols; y0=r*H/rows; y1=(r+1)*H/rows
                d.rectangle((x0,y0,x1,y1),outline=(124,232,255,190),width=4)
                d.rounded_rectangle((x0+12,y0+12,x0+92,y0+46),radius=8,fill=(3,9,18,190))
                d.text((x0+24,y0+20),f"GPU {r*cols+cc+1}",font=font(13,True),fill='white')
        d.rectangle((0,0,W,90),fill=(3,6,15,225))
        d.text((24,15),"Zoom out: the fluid domain can be divided across GPUs",font=font(30,True),fill='white')
        d.text((26,54),"Neighbouring domains exchange boundary data while most cell updates remain local.",font=font(16),fill=(185,208,238))
        rp=self.ctx.run_dir/'reveal.jpg'; self.ctx.save_frame(rev,rp); self.ctx.finish(rp)
