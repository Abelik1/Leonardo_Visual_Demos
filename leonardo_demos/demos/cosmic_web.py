from __future__ import annotations
import math
import numpy as np
from PIL import Image, ImageDraw
from ..base import Demo
from ..backend import to_numpy
from ..render import add_title, add_progress, save_frame, mosaic, font, palette

# Atomic masses in atomic mass units. Helium-4 has atomic NUMBER 2 but MASS
# number 4 (two protons plus two neutrons), and it is the mass that sets the
# gas dynamics. Primordial abundances: about 76% hydrogen / 24% helium by mass.
M_HYDROGEN=1.008
M_HELIUM=4.003
PRIMORDIAL_HELIUM=0.24


def mean_molecular_weight(helium_mass_fraction):
    """Mean molecular weight mu of a neutral H/He gas, in units of m_H.

    mu = 1 / (X/A_H + Y/A_He) with X + Y = 1. Pure hydrogen gives 1.0, the
    primordial mix gives about 1.22, pure helium gives 4.0.
    """
    Y=float(np.clip(helium_mass_fraction,0.0,1.0)); X=1.0-Y
    return 1.0/(X/M_HYDROGEN + Y/M_HELIUM)


class CosmicWebDemo(Demo):
    id="cosmic_web"; title="Cosmic-web formation"
    def init(self,Np,seed,n=None,ns=-1.0,kmin=3.0,amplitude=None):
        """Zel'dovich initial conditions from a power-law power spectrum.

        Displacing a particle grid by small *random* jitter, as this demo did
        before, puts nearly all the power in the box-scale mode: the whole
        volume then collapses to a single clump and no filaments or voids ever
        appear. Proper Zel'dovich ICs seed a spectrum of modes, so structure
        grows on many scales at once and the web emerges.

        Modes below `kmin` (the few longest wavelengths in the box) are
        suppressed, since a periodic box cannot represent them honestly.
        """
        rng=np.random.default_rng(seed)
        side=int(np.ceil(np.sqrt(Np)))
        m=int(n or max(64,side))
        white=rng.normal(0,1,(m,m))
        wk=np.fft.rfft2(white)
        ky=np.fft.fftfreq(m)*m; kx=np.fft.rfftfreq(m)*m
        KX,KY=np.meshgrid(kx,ky)
        k2=KX*KX+KY*KY; k2[0,0]=1.0
        k=np.sqrt(k2)
        power=np.where(k>=kmin,k**ns,0.0); power[0,0]=0.0
        dk=wk*np.sqrt(power)
        # Zel'dovich displacement field: psi_k = i k / k^2 * delta_k
        psix=np.fft.irfft2(1j*KX/k2*dk,s=(m,m)).real
        psiy=np.fft.irfft2(1j*KY/k2*dk,s=(m,m)).real
        scale=max(psix.std(),psiy.std(),1e-12)
        amp=float(self.settings.get('ic_amplitude',0.9) if amplitude is None else amplitude)
        psix*=amp/scale/m; psiy*=amp/scale/m
        gx,gy=np.meshgrid((np.arange(side)+.5)/side,(np.arange(side)+.5)/side,indexing='xy')
        gx=gx.ravel()[:Np]; gy=gy.ravel()[:Np]
        ix=(gx*m).astype(int)%m; iy=(gy*m).astype(int)%m
        dx=psix[iy,ix]; dy=psiy[iy,ix]
        pos=np.stack([(gx+dx)%1.0,(gy+dy)%1.0],axis=1)
        # In the Zel'dovich approximation the velocity is proportional to the
        # displacement, which is what makes the flow converge onto sheets.
        vel=np.stack([dx,dy],axis=1)*float(self.settings.get('ic_velocity',0.55))
        xp=self.ctx.xp
        return xp.asarray(pos,dtype=xp.float32),xp.asarray(vel,dtype=xp.float32)
    def density(self,pos,n):
        """Density contrast delta = rho/rho_bar - 1 on the mesh.

        The Poisson source has to be the dimensionless contrast. Using raw
        particle counts made the force scale with the number of particles per
        cell, so the forces came out around a hundred times too strong and the
        whole box free-fell within a hundred steps.
        """
        xp=self.ctx.xp; ix=(pos[:,0]*n).astype(xp.int32)%n; iy=(pos[:,1]*n).astype(xp.int32)%n
        rho=xp.zeros((n,n),dtype=xp.float32); xp.add.at(rho,(iy,ix),1.0)
        mean=xp.mean(rho)
        return rho/(mean+1e-12)-1.0
    def force_grid(self,rho,g,jeans=0.0):
        xp=self.ctx.xp; n=rho.shape[0]
        rk=xp.fft.rfft2(rho)
        # k in radians per BOX length. fftfreq gives cycles per sample, so the
        # extra factor of n is required: without it k is n times too small, the
        # potential n^2 too large and the net force n times too strong.
        ky=xp.fft.fftfreq(n)*2*xp.pi*n; kx=xp.fft.rfftfreq(n)*2*xp.pi*n
        KX,KY=xp.meshgrid(kx,ky); k2=KX*KX+KY*KY; k2[0,0]=1
        phi=-g*rk/k2; phi[0,0]=0
        if jeans>0:
            # Gas pressure resists collapse below the Jeans scale. A heavier
            # mean molecular weight means a lower sound speed, a smaller Jeans
            # length, and therefore finer structure.
            phi=phi*xp.exp(-.5*k2*jeans*jeans)
        fx=xp.fft.irfft2(-1j*KX*phi,s=(n,n)).real; fy=xp.fft.irfft2(-1j*KY*phi,s=(n,n)).real
        return fx,fy
    def sample_force(self,pos,fx,fy):
        xp=self.ctx.xp; n=fx.shape[0]; ix=(pos[:,0]*n).astype(xp.int32)%n; iy=(pos[:,1]*n).astype(xp.int32)%n
        return xp.stack([fx[iy,ix],fy[iy,ix]],axis=1)
    def step(self,pos,vel,n,g,steps=1,jeans=0.0):
        """Particle-mesh integration on an expanding background.

        Comoving coordinates with peculiar velocity v = a dx/dt, matter
        dominated so a is proportional to t^(2/3) and H = 2/(3t):

            dx/dt = v / a
            dv/dt = -grad(phi)/a - H v

        Without the expansion the box simply collapsed into a single halo and
        the filaments and voids never survived to be seen.
        """
        xp=self.ctx.xp; dt=.018
        for _ in range(steps):
            self.time+=dt
            a=(self.time/self.t0)**(2.0/3.0)
            H=(2.0/3.0)/self.time
            rho=self.density(pos,n)
            fx,fy=self.force_grid(rho,g,jeans)
            acc=self.sample_force(pos,fx,fy)/a
            vel=vel+dt*(acc-H*vel)
            pos=(pos+dt*vel/a)%1.0
        return pos,vel,self.density(pos,n)
    def render_density(self,rho,size=(1280,720)):
        a=to_numpy(rho); a=np.log1p(np.maximum(0,a-a.min())*2.5); rgb=palette(a,'cosmic')
        return Image.fromarray(rgb).resize(size,Image.Resampling.BILINEAR)
    def jeans_scale(self,mu,n):
        """Pressure-support length in grid units.

        Sound speed c_s is proportional to sqrt(T/mu), and the Jeans length
        scales with c_s, so lambda_J is proportional to 1/sqrt(mu). Normalised
        so the primordial mix sits at the reference scale.
        """
        ref=mean_molecular_weight(PRIMORDIAL_HELIUM)
        return float(self.settings.get('jeans_ref',2.4))*math.sqrt(ref/mu)/n
    def budget(self):
        """Total particle-mesh steps, independent of frame count.

        Filaments and voids need several hundred steps to grow out of the
        near-uniform initial conditions; a frame-derived budget stopped at a few
        dozen and left the initial particle lattice still visible.
        """
        s=self.settings
        if 'total_steps' in s: return max(1,int(s['total_steps']))
        return max(1,int(s.get('steps_per_frame',1))*self.ctx.frames)
    def run(self):
        n=int(self.settings['grid']); Np=int(self.settings['particles'])
        total=self.budget(); done=0
        seed=int(self.ctx.params.get('seed',42)); g=float(self.ctx.params.get('gravity',1.0))
        Y=float(self.ctx.params.get('helium',PRIMORDIAL_HELIUM))
        mu=mean_molecular_weight(Y); jeans=self.jeans_scale(mu,n)
        self.t0=float(self.settings.get('t0',1.0)); self.time=self.t0
        pos,vel=self.init(Np,seed)
        for i in range(self.ctx.frames):
            step_target=int(round(total*(i+1)/self.ctx.frames))
            pos,vel,rho=self.step(pos,vel,n,g,max(1,step_target-done),jeans); done=step_target
            im=self.render_density(rho)
            im=add_title(im,"Cosmic-web formation",f"particle–mesh gravity · {Np:,} particles · {n}² mesh · seed {seed}")
            d=ImageDraw.Draw(im,'RGBA')
            d.text((28,116),"Tiny initial differences grow into clusters, filaments and voids.",font=font(17,True),fill=(220,238,255))
            d.rounded_rectangle((26,142,470,262),radius=16,fill=(4,9,22,205))
            d.text((44,156),"GAS COMPOSITION (by mass)",font=font(14,True),fill=(150,226,255))
            d.text((44,180),f"hydrogen  {100*(1-Y):.0f}%   ·   A = {M_HYDROGEN:.3f}",font=font(15,True),fill=(206,224,248))
            d.text((44,202),f"helium    {100*Y:.0f}%   ·   A = {M_HELIUM:.3f}",font=font(15,True),fill=(255,206,150))
            d.text((44,226),f"mean molecular weight µ = {mu:.3f}",font=font(15,True),fill=(120,236,255))
            d.text((44,244),f"Jeans smoothing ≈ {jeans*n:.2f} cells",font=font(13),fill=(150,170,200))
            add_progress(im,(i+1)/self.ctx.frames,"NEARLY UNIFORM","EMERGENT STRUCTURE")
            self.ctx.save_frame(im,self.ctx.frame_path(i)); self.ctx.write_status(i,f"µ={mu:.3f}")
        # Reveal: the same initial universe run at different H/He mixes, which
        # is a real composition sweep rather than sixteen random seeds.
        side=4; ims=[]; labels=[]
        m=max(64,n//2); pcount=max(3000,min(9000,Np//4))
        for j in range(side*side):
            Yj=j/(side*side-1)
            muj=mean_molecular_weight(Yj); jj=self.jeans_scale(muj,m)
            p_,v_=self.init(pcount,seed); r=None
            self.time=self.t0
            p_,v_,r=self.step(p_,v_,m,g,int(self.settings.get('sweep_steps',600)),jj)
            ims.append(self.render_density(r,(260,146)))
            labels.append(f"He {100*Yj:.0f}%  µ{muj:.2f}")
        rev=mosaic(ims,side,title="One universe, sixteen gas compositions",
                   subtitle="Pure hydrogen (µ=1.00) through pure helium (µ=4.00). Heavier gas, finer structure.",
                   labels=labels,label_fill=(190,226,255))
        rp=self.ctx.run_dir/'reveal.jpg'; self.ctx.save_frame(rev,rp); self.ctx.finish(rp)
