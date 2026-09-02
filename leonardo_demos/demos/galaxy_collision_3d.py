from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from ..backend import to_numpy
from ..base import Demo
from ..render import add_progress, add_title
from .galaxy_collision import G, MW_M31, TIME_UNIT_GYR


OBSERVATIONS = Path(__file__).resolve().parents[2] / "data" / "galaxy_observations.npz"


class GalaxyCollision3DDemo(Demo):
    """A softened, direct, fully self-gravitating 3-D galaxy encounter.

    Every simulation particle carries mass and contributes to every other
    particle's acceleration. The GPU kernel follows the source repository's
    shared-memory tiling strategy; the CPU path evaluates the same equation in
    bounded NumPy tiles. These particles are galaxy-scale super-particles, not
    literal one-particle-per-observed-star bodies.
    """

    id = "galaxy_collision_3d"
    title = "Self-gravitating galaxy collision"
    default_method = "leapfrog"
    methods = ("leapfrog", "murb_kinematic")
    method_labels = {
        "leapfrog": "Leapfrog (recommended)",
        "murb_kinematic": "MUrB constant-acceleration",
    }
    method_descriptions = {
        "leapfrog": "Symplectic kick–drift–kick integration of the full all-pairs 3D force.",
        "murb_kinematic": "The reference repository's ordinary x += v·dt + ½a·dt² update.",
    }
    timing_methods = {"setup": "initialization", "step": "simulation", "render": "render"}

    _gpu_kernel = None

    @staticmethod
    def rotation_x(angle):
        c, s = math.cos(angle), math.sin(angle)
        return np.array(((1, 0, 0), (0, c, -s), (0, s, c)), dtype=np.float64)

    @staticmethod
    def rotation_z(angle):
        c, s = math.cos(angle), math.sin(angle)
        return np.array(((c, -s, 0), (s, c, 0), (0, 0, 1)), dtype=np.float64)

    def observations(self):
        if OBSERVATIONS.exists():
            with np.load(OBSERVATIONS) as data:
                metadata = json.loads(str(data["metadata"]))
                return (data["mw_xyz"].copy(), data["m31_xy"].copy(),
                        data["m31_colour"].copy(), metadata)
        # Offline fallback keeps a batch job runnable, but metadata makes the
        # absence of observational seeds explicit in the UI and run record.
        return None, None, None, {
            "gaia": {"meaning": "analytic fallback", "rows": 0},
            "phat": {"meaning": "analytic fallback", "rows": 0},
        }

    @staticmethod
    def _isotropic(count, radius, rng):
        direction = rng.normal(size=(count, 3))
        direction /= np.maximum(np.linalg.norm(direction, axis=1, keepdims=True), 1e-12)
        return direction * radius[:, None]

    def disc_positions(self, galaxy, count, scale, rng, mw_xyz, m31_xy, m31_colour):
        if galaxy == "mw" and mw_xyz is not None and len(mw_xyz):
            seed = np.asarray(mw_xyz)[rng.integers(0, len(mw_xyz), count)]
            observed_r = np.linalg.norm(seed[:, :2], axis=1)
            # Gaia's distance sample is concentrated near the Sun. Its local
            # radial texture and its measured vertical coordinates condition a
            # global analytic disc; copying its 8-kpc ring directly would make
            # four clumps, not an honest visualisation of a complete spiral.
            radial_texture = np.clip(observed_r / max(np.median(observed_r), .1), .55, 1.55)
            radius = np.clip(rng.gamma(2.15, scale * .52, count) * radial_texture,
                             .45, scale * 3.25)
            arm = rng.integers(0, 4, count)
            # Four logarithmic-like arms, with a radius-dependent phase. The
            # broad scatter represents finite arm width, not catalogue error.
            angle = 2 * np.pi * arm / 4 + .34 * radius + rng.normal(0, .16, count)
            z = np.clip(seed[:, 2] + rng.normal(0, .08, count), -2.2, 2.2)
            colour = np.full(count, 1.0, dtype=np.float32)
            catalogue = np.ones(count, dtype=np.int8)
        elif galaxy == "m31" and m31_xy is not None and len(m31_xy):
            pick = rng.integers(0, len(m31_xy), count)
            xy = np.asarray(m31_xy)[pick].astype(np.float64, copy=True)
            colour = np.asarray(m31_colour)[pick].astype(np.float32, copy=True)
            # PHAT covers roughly one third of M31. A 180-degree completion
            # preserves its measured rings/arms without inventing a false
            # observed line-of-sight coordinate.
            xy[rng.random(count) < .5] *= -1
            radius = np.linalg.norm(xy, axis=1)
            angle = np.arctan2(xy[:, 1], xy[:, 0])
            young = np.clip((2.3 - colour) / 2.3, 0, 1)
            height = .28 + .52 * (1 - young)
            z = rng.laplace(0, height)
            catalogue = np.ones(count, dtype=np.int8)
        else:
            radius = np.clip(rng.gamma(2.0, scale * .55, count), .4, scale * 3.2)
            arms = 4 if galaxy == "mw" else 2
            arm = rng.integers(0, arms, count)
            angle = 2 * np.pi * arm / arms + .27 * radius + rng.normal(0, .24, count)
            z = rng.laplace(0, .45 if galaxy == "mw" else .65, count)
            colour = np.full(count, 1.0, dtype=np.float32)
            catalogue = np.zeros(count, dtype=np.int8)
        xyz = np.column_stack((radius * np.cos(angle), radius * np.sin(angle), z))
        return xyz, colour, catalogue

    def galaxy(self, count, total_mass, scale, galaxy, orientation, rng,
               mw_xyz, m31_xy, m31_colour):
        disk_n = max(24, int(count * .52))
        bulge_n = max(12, int(count * .13))
        halo_n = count - disk_n - bulge_n
        if halo_n < 12:
            halo_n = 12
            disk_n = max(12, count - bulge_n - halo_n)
        counts = (disk_n, bulge_n, halo_n)
        mass_fractions = (.07, .02, .91)

        disk, colour, catalogue = self.disc_positions(
            galaxy, disk_n, scale, rng, mw_xyz, m31_xy, m31_colour)
        bulge_scale = scale * .22
        u = np.clip(rng.random(bulge_n), 1e-5, .92)
        bulge_r = np.clip(bulge_scale * np.sqrt(u) / (1 - np.sqrt(u)), .05, scale * 1.4)
        bulge = self._isotropic(bulge_n, bulge_r, rng)
        halo_scale = scale * 3.2
        u = np.clip(rng.random(halo_n), 1e-5, .86)
        halo_r = np.clip(halo_scale * np.sqrt(u) / (1 - np.sqrt(u)), .3, scale * 8.0)
        halo = self._isotropic(halo_n, halo_r, rng)
        positions = np.vstack((disk, bulge, halo))

        masses = np.concatenate([
            np.full(n, total_mass * fraction / n, dtype=np.float64)
            for n, fraction in zip(counts, mass_fractions)
        ])
        component = np.concatenate([
            np.zeros(disk_n, dtype=np.int8),
            np.ones(bulge_n, dtype=np.int8),
            np.full(halo_n, 2, dtype=np.int8),
        ])
        catalogue_flag = np.concatenate((catalogue, np.zeros(bulge_n + halo_n, dtype=np.int8)))
        colours = np.concatenate((colour, np.full(bulge_n + halo_n, 1.5, dtype=np.float32)))

        radius = np.maximum(np.linalg.norm(positions, axis=1), .2)
        enclosed = total_mass * (
            .07 * (1 - np.exp(-radius / scale) * (1 + radius / scale))
            + .02 * radius * radius / (radius + bulge_scale) ** 2
            + .91 * radius * radius / (radius + halo_scale) ** 2
        )
        circular = np.sqrt(G * enclosed / radius)
        velocities = np.zeros_like(positions)
        disk_r = np.maximum(np.linalg.norm(disk[:, :2], axis=1), .1)
        velocities[:disk_n, 0] = -disk[:, 1] / disk_r * circular[:disk_n]
        velocities[:disk_n, 1] = disk[:, 0] / disk_r * circular[:disk_n]
        velocities[:disk_n] += rng.normal(0, 8.0, (disk_n, 3))
        # Isotropic pressure support for bulge and dark halo super-particles.
        sigma = np.sqrt(np.maximum(G * enclosed[disk_n:] / (3 * radius[disk_n:]), 1.0))
        velocities[disk_n:] = rng.normal(size=(bulge_n + halo_n, 3)) * sigma[:, None]

        positions = positions @ orientation.T
        velocities = velocities @ orientation.T
        return positions, velocities, masses, component, catalogue_flag, colours

    def setup(self, count, impact, speed, disc_tilt,
              milky_way_mass=MW_M31["m1"], andromeda_mass=MW_M31["m2"]):
        rng = np.random.default_rng(20260901)
        mw_xyz, m31_xy, m31_colour, metadata = self.observations()
        n1 = count // 2
        n2 = count - n1
        mw = self.galaxy(n1, milky_way_mass, 9.0, "mw", np.eye(3), rng,
                         mw_xyz, m31_xy, m31_colour)
        m31_orientation = self.rotation_z(math.radians(28)) @ self.rotation_x(math.radians(disc_tilt))
        m31 = self.galaxy(n2, andromeda_mass, 12.0, "m31", m31_orientation, rng,
                          mw_xyz, m31_xy, m31_colour)
        separation = MW_M31["separation"]
        offset = 80.0 * float(impact)
        c1 = np.array((-separation * .5, -offset * .5, 0.0))
        c2 = np.array((separation * .5, offset * .5, 0.0))
        relative = np.array((MW_M31["v_radial"] * float(speed),
                             MW_M31["v_transverse"] + 42.0 * float(impact), 0.0))
        mu1 = andromeda_mass / (milky_way_mass + andromeda_mass)
        mu2 = milky_way_mass / (milky_way_mass + andromeda_mass)
        pos = np.vstack((mw[0] + c1, m31[0] + c2))
        vel = np.vstack((mw[1] - relative * mu1, m31[1] + relative * mu2))
        mass = np.concatenate((mw[2], m31[2]))
        origin = np.concatenate((np.zeros(n1, dtype=np.int8), np.ones(n2, dtype=np.int8)))
        component = np.concatenate((mw[3], m31[3]))
        catalogue = np.concatenate((mw[4], m31[4]))
        colour = np.concatenate((mw[5], m31[5]))
        xp = self.ctx.xp
        return (xp.asarray(pos, dtype=xp.float32), xp.asarray(vel, dtype=xp.float32),
                xp.asarray(mass, dtype=xp.float32), origin, component, catalogue,
                colour, metadata)

    @classmethod
    def gpu_kernel(cls, xp):
        if cls._gpu_kernel is None:
            cls._gpu_kernel = xp.RawKernel(r'''
            extern "C" __global__
            void all_pairs(const float* p, const float* m, float* a,
                           const int n, const float soft2, const float grav) {
                const int i = blockDim.x * blockIdx.x + threadIdx.x;
                __shared__ float sx[256], sy[256], sz[256], sm[256];
                float ix=0, iy=0, iz=0, ax=0, ay=0, az=0;
                if (i < n) { ix=p[3*i]; iy=p[3*i+1]; iz=p[3*i+2]; }
                for (int base=0; base<n; base+=blockDim.x) {
                    const int j=base+threadIdx.x;
                    if (j<n) { sx[threadIdx.x]=p[3*j]; sy[threadIdx.x]=p[3*j+1];
                               sz[threadIdx.x]=p[3*j+2]; sm[threadIdx.x]=m[j]; }
                    else { sx[threadIdx.x]=sy[threadIdx.x]=sz[threadIdx.x]=sm[threadIdx.x]=0; }
                    __syncthreads();
                    if (i<n) {
                        const int remaining=n-base;
                        const int width=remaining < blockDim.x ? remaining : blockDim.x;
                        for (int k=0;k<width;k++) {
                            const float dx=sx[k]-ix, dy=sy[k]-iy, dz=sz[k]-iz;
                            const float inv=rsqrtf(dx*dx+dy*dy+dz*dz+soft2);
                            const float f=grav*sm[k]*inv*inv*inv;
                            ax+=f*dx; ay+=f*dy; az+=f*dz;
                        }
                    }
                    __syncthreads();
                }
                if (i<n) { a[3*i]=ax; a[3*i+1]=ay; a[3*i+2]=az; }
            }''', "all_pairs")
        return cls._gpu_kernel

    def acceleration(self, positions, masses, softening):
        xp = self.ctx.xp
        count = len(positions)
        acceleration = xp.zeros_like(positions)
        if xp is not np:
            block = 256
            self.gpu_kernel(xp)(((count + block - 1) // block,), (block,),
                                (positions, masses, acceleration, np.int32(count),
                                 np.float32(softening * softening), np.float32(G)))
            return acceleration
        tile = max(32, int(self.settings.get("force_tile", 160)))
        soft2 = float(softening) ** 2
        for i0 in range(0, count, tile):
            i1 = min(count, i0 + tile)
            target = positions[i0:i1]
            value = np.zeros_like(target)
            for j0 in range(0, count, tile):
                source = positions[j0:j0 + tile]
                delta = source[None, :, :] - target[:, None, :]
                radius2 = np.sum(delta * delta, axis=2) + soft2
                weight = masses[j0:j0 + tile][None, :] / (radius2 * np.sqrt(radius2))
                value += G * np.sum(delta * weight[:, :, None], axis=1)
            acceleration[i0:i1] = value
        return acceleration

    def step(self, positions, velocities, masses, dt, steps, softening):
        method = self.ctx.method
        if method in {"default", "leapfrog"}:
            half = .5 * dt
            acceleration = self.acceleration(positions, masses, softening)
            for _ in range(steps):
                velocities += half * acceleration
                positions += dt * velocities
                acceleration = self.acceleration(positions, masses, softening)
                velocities += half * acceleration
            return positions, velocities
        if method == "murb_kinematic":
            half_dt2 = .5 * dt * dt
            for _ in range(steps):
                acceleration = self.acceleration(positions, masses, softening)
                positions += dt * velocities + half_dt2 * acceleration
                velocities += dt * acceleration
            return positions, velocities
        raise ValueError(f"unknown 3-D collision solver: {method}")

    @staticmethod
    def centres(positions, masses, origin):
        values = []
        for which in (0, 1):
            select = origin == which
            values.append(np.sum(positions[select] * masses[select, None], axis=0) / masses[select].sum())
        return values

    @staticmethod
    def project(points, yaw=-.34, pitch=.48):
        ry = GalaxyCollision3DDemo.rotation_z(yaw)
        rx = GalaxyCollision3DDemo.rotation_x(pitch)
        return points @ (rx @ ry).T

    def render(self, positions, origin, component, extent, size=(1280, 720)):
        points = self.project(np.asarray(positions, dtype=np.float64))
        centre = np.median(points, axis=0)
        points -= centre
        width, height = size
        scale = min(width / (2.15 * extent), height / (1.45 * extent))
        x = width * .5 + points[:, 0] * scale
        y = height * .5 - points[:, 1] * scale
        visible = (x >= -8) & (x < width + 8) & (y >= -8) & (y < height + 8)
        order = np.flatnonzero(visible)[np.argsort(points[visible, 2])]
        base = Image.new("RGB", size, (1, 3, 10))
        glow = Image.new("RGBA", size, (0, 0, 0, 0))
        gd = ImageDraw.Draw(glow, "RGBA")
        sharp = Image.new("RGBA", size, (0, 0, 0, 0))
        sd = ImageDraw.Draw(sharp, "RGBA")
        for index in order:
            galaxy = int(origin[index]); kind = int(component[index])
            if kind == 2:
                colour = (42, 82, 125) if galaxy == 0 else (120, 61, 45)
                radius, alpha = 1.0, 38
            elif kind == 1:
                colour = (225, 238, 255) if galaxy == 0 else (255, 220, 174)
                radius, alpha = 2.0, 210
            else:
                colour = (92, 194, 255) if galaxy == 0 else (255, 128, 66)
                radius, alpha = 1.45, 205
            xx, yy = float(x[index]), float(y[index])
            gd.ellipse((xx-radius*3, yy-radius*3, xx+radius*3, yy+radius*3), fill=(*colour, alpha//3))
            sd.ellipse((xx-radius, yy-radius, xx+radius, yy+radius), fill=(*colour, alpha))
        glow = glow.filter(ImageFilter.GaussianBlur(3.2))
        return Image.alpha_composite(Image.alpha_composite(base.convert("RGBA"), glow), sharp).convert("RGB")

    def write_interactive(self, frame, positions, origin, component, catalogue,
                          colour, extent, time_gyr):
        directory = self.ctx.run_dir / "interactive"
        directory.mkdir(exist_ok=True)
        points = np.asarray(positions, dtype=np.float32)
        # Bound browser payloads for large Leonardo runs with a deterministic
        # stride while keeping the physical simulation at full particle count.
        stride = max(1, math.ceil(len(points) / 9000))
        select = np.arange(0, len(points), stride)
        payload = {
            "kind": "nbody-galaxy-3d",
            "positions": np.round(points[select], 3).tolist(),
            "origin": np.asarray(origin)[select].astype(int).tolist(),
            "component": np.asarray(component)[select].astype(int).tolist(),
            "catalogue": np.asarray(catalogue)[select].astype(int).tolist(),
            "colour": np.round(np.asarray(colour)[select], 2).tolist(),
            "extent_kpc": round(float(extent), 3),
            "time_gyr": round(float(time_gyr), 4),
            "simulated_particles": int(len(points)),
        }
        path = directory / f"frame_{frame:04d}.json"
        path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")

    def run(self):
        count = max(96, int(self.settings.get("particles", 700)))
        impact = float(self.ctx.params.get("impact", .35))
        speed = float(self.ctx.params.get("speed", 1.0))
        disc_tilt = float(self.ctx.params.get("disc_tilt", 35.0))
        softening = float(self.ctx.params.get("softening", self.settings.get("softening", 4.0)))
        milky_way_mass = float(self.ctx.params.get("milky_way_mass", 1.5)) * 1e12
        andromeda_mass = float(self.ctx.params.get("andromeda_mass", 1.5)) * 1e12
        positions, velocities, masses, origin, component, catalogue, colour, metadata = self.setup(
            count, impact, speed, disc_tilt, milky_way_mass, andromeda_mass)
        span_gyr = float(self.settings.get("span_gyr", 7.5))
        total_time = span_gyr / TIME_UNIT_GYR
        requested_substeps = max(1, int(self.settings.get("substeps", 3)))
        # A frame count is a presentation choice, not a numerical-accuracy
        # control. Enforce a ~24 Myr ceiling even for two-frame tests/runs so a
        # low frame count cannot catapult massive super-particles out of the
        # galaxies with one enormous update.
        max_step = float(self.settings.get("max_step", .025))
        intervals = max(1, self.ctx.frames - 1)
        substeps = max(requested_substeps,
                       int(math.ceil(total_time / (intervals * max_step))))
        dt = total_time / (intervals * substeps)
        manifest = {"kind": "nbody-galaxy-3d", "folder": "interactive",
                    "frames": self.ctx.frames, "particles": count,
                    "catalogue_sources": metadata,
                    "model_status": "illustrative catalogue-conditioned super-particle experiment; not a fitted equilibrium Local Group model"}
        self.ctx.write_meta({"galaxy3d_view": manifest, "catalogues": metadata,
                             "physics": {"force": "softened direct all-pairs 3-D gravity",
                                         "complexity": "O(N^2)", "softening_kpc": softening,
                                         "substeps_per_frame": substeps,
                                         "step_myr": dt * TIME_UNIT_GYR * 1000,
                                         "model_status": "illustrative, catalogue-conditioned super-particle model; not a fitted equilibrium Local Group prediction"}})
        final_image = None
        for frame in range(self.ctx.frames):
            # The first saved state is the actual t=0 initial condition. This
            # makes the catalogue-conditioned disc morphology inspectable
            # before the collision disrupts it.
            if frame:
                positions, velocities = self.step(positions, velocities, masses, dt, substeps, softening)
            host_positions = to_numpy(positions)
            host_masses = to_numpy(masses)
            c1, c2 = self.centres(host_positions, host_masses, origin)
            separation = float(np.linalg.norm(c2 - c1))
            extent = float(np.clip(.63 * separation + 145.0, 150.0, 640.0))
            time_gyr = frame / intervals * span_gyr
            final_image = self.render(host_positions, origin, component, extent)
            solver = self.method_labels.get(self.ctx.method, self.ctx.method)
            final_image = add_title(final_image, self.title,
                                    f"illustrative full 3-D all-pairs gravity · {solver} · {count:,} massive super-particles")
            add_progress(final_image, frame / intervals, "CURRENT LOCAL GROUP", "MERGER")
            self.ctx.save_frame(final_image, self.ctx.frame_path(frame))
            self.write_interactive(frame, host_positions, origin, component, catalogue,
                                   colour, extent, time_gyr)
            self.ctx.write_status(frame, f"t=+{time_gyr:.2f} Gyr", {
                "physics": "full self-gravity · O(N²)", "dimensions": "3 spatial",
                "solver": solver, "massive particles": f"{count:,}",
                "separation": f"{separation:.0f} kpc", "softening": f"{softening:.1f} kpc",
                "saved-frame interval": f"{span_gyr / intervals * 1000:.1f} Myr",
                "solver step": f"{dt * TIME_UNIT_GYR * 1000:.2f} Myr × {substeps}",
                "catalogues": f"Gaia DR3 {metadata['gaia']['rows']:,} · PHAT v3 {metadata['phat']['rows']:,}",
                "model status": "illustrative super-particle N-body; not a fitted equilibrium prediction",
                "compute": self.ctx.backend_name,
            })
        reveal = self.ctx.run_dir / "reveal.jpg"
        if final_image is not None:
            self.ctx.save_frame(final_image, reveal)
        self.ctx.finish(reveal)
