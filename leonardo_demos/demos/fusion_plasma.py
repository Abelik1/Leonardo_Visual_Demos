from __future__ import annotations

import json
import math

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from ..backend import to_numpy
from ..base import Demo
from ..colors import palette
from ..render import add_progress, add_title, font, mosaic, save_frame


class FusionPlasmaDemo(Demo):
    timing_methods={"initialise":"initialization","step":"simulation","hero":"render"}
    """A reduced nonlinear-wave model displayed on a tokamak torus.

    The complex Ginzburg--Landau equation is an exhibition-scale amplitude
    model, not a reactor prediction.  It is useful here because every lattice
    point participates in a nonlinear, diffusive update and its ordered-wave
    to defect-turbulence transition is both fast and visually unmistakable.
    """

    id = "fusion_plasma"
    title = "Star in a Bottle"
    aspect = 16 / 9

    def grid_shape(self, n):
        ny = max(20, int(n))
        return ny, max(32, int(round(ny * self.aspect)))

    def budget(self):
        if "total_steps" in self.settings:
            return max(1, int(self.settings["total_steps"]))
        return max(1, int(self.settings.get("steps_per_frame", 8)) * self.ctx.frames)

    def initialise(self, ny, nx, seed=12):
        rng = np.random.default_rng(seed)
        y, x = np.mgrid[0:ny, 0:nx]
        phase = 2 * np.pi * (x / nx * 2.0 + y / ny)
        envelope = 0.48 + 0.18 * np.cos(2 * np.pi * y / ny)
        real = envelope * np.cos(phase) + rng.normal(0, 0.055, (ny, nx))
        imag = envelope * np.sin(phase) + rng.normal(0, 0.055, (ny, nx))
        xp = self.ctx.xp
        return (
            xp.asarray(real.astype(np.float32)),
            xp.asarray(imag.astype(np.float32)),
        )

    def step(self, real, imag, magnetic_field, heating, density, steps):
        """Integrate a complex Ginzburg--Landau amplitude equation.

        Magnetic confinement shifts the dispersion coefficients and reduces
        the effective drive. Heating increases nonlinear drive; density adds
        damping. The coefficients are deliberately dimensionless.
        """
        xp = self.ctx.xp
        b = float(magnetic_field)
        heat = float(heating)
        dens = float(density)
        c1 = 0.55 + 2.1 / (b + 1.2)
        c3 = 0.45 + 0.032 * heat / max(0.45, dens)
        drive = 0.74 + 0.015 * heat
        damping = 0.36 + 0.22 * dens + 0.08 * b
        dt = 0.075
        for _ in range(steps):
            lap_r = 0.25 * (
                xp.roll(real, 1, 0) + xp.roll(real, -1, 0)
                + xp.roll(real, 1, 1) + xp.roll(real, -1, 1)
                - 4 * real
            )
            lap_i = 0.25 * (
                xp.roll(imag, 1, 0) + xp.roll(imag, -1, 0)
                + xp.roll(imag, 1, 1) + xp.roll(imag, -1, 1)
                - 4 * imag
            )
            amp2 = real * real + imag * imag
            dr = drive * real + lap_r - c1 * lap_i - amp2 * real + c3 * amp2 * imag
            di = drive * imag + lap_i + c1 * lap_r - amp2 * imag - c3 * amp2 * real
            real += dt * (dr - 0.12 * damping * real)
            imag += dt * (di - 0.12 * damping * imag)
            # A tiny poloidal shear represents the E x B rotation that winds
            # structures around a tokamak. It is part of the state update,
            # not a rendering effect.
            shear = 0.012 * (heat / 25.0) / max(0.6, b / 4.0)
            real += shear * (xp.roll(real, 1, 1) - xp.roll(real, -1, 1))
            imag += shear * (xp.roll(imag, 1, 1) - xp.roll(imag, -1, 1))
            xp.clip(real, -2.5, 2.5, out=real)
            xp.clip(imag, -2.5, 2.5, out=imag)
        return real, imag

    @staticmethod
    def turbulence(real, imag):
        amp = np.sqrt(real * real + imag * imag)
        phase = np.arctan2(imag, real)
        dx = np.angle(np.exp(1j * (np.roll(phase, 1, 1) - phase)))
        dy = np.angle(np.exp(1j * (np.roll(phase, 1, 0) - phase)))
        amp_grad = np.abs(np.roll(amp, 1, 0) - amp) + np.abs(np.roll(amp, 1, 1) - amp)
        return float(np.mean(np.abs(dx) + np.abs(dy)) / np.pi + 0.25 * np.mean(amp_grad))

    @staticmethod
    def initialise_tracers(count, trail, seed=91):
        """Seed passive tracer histories in periodic toroidal coordinates."""
        rng = np.random.default_rng(seed)
        head = rng.random((max(1, int(count)), 2)).astype(np.float32)
        return np.repeat(head[:, None, :], max(2, int(trail)), axis=1)

    @staticmethod
    def flow_field(real, imag, magnetic_field, heating):
        """Derive a passive drift field from the simulated complex amplitude.

        Phase gradients provide wave propagation while the rotated amplitude
        gradient supplies an E x B-like drift around coherent structures. The
        result is measured in turns of the torus per solver step.
        """
        real, imag = to_numpy(real), to_numpy(imag)
        amp = np.sqrt(real * real + imag * imag)
        phase = np.arctan2(imag, real)
        phase_x = 0.5 * np.angle(np.exp(1j * (np.roll(phase, -1, 1) - np.roll(phase, 1, 1))))
        phase_y = 0.5 * np.angle(np.exp(1j * (np.roll(phase, -1, 0) - np.roll(phase, 1, 0))))
        amp_x = 0.5 * (np.roll(amp, -1, 1) - np.roll(amp, 1, 1))
        amp_y = 0.5 * (np.roll(amp, -1, 0) - np.roll(amp, 1, 0))
        confinement = max(0.55, math.sqrt(float(magnetic_field) / 5.0))
        drive = max(0.25, float(heating) / 25.0)
        toroidal = 0.00105 * drive / confinement
        flow_u = toroidal + 0.0024 * phase_x / np.pi - 0.0042 * amp_y / confinement
        flow_v = 0.0019 * phase_y / np.pi + 0.0042 * amp_x / confinement
        return flow_u.astype(np.float32), flow_v.astype(np.float32)

    @staticmethod
    def _sample_periodic(field, uv):
        """Bilinearly sample a 2-D periodic field at normalised (u, v)."""
        ny, nx = field.shape
        x = np.mod(uv[:, 0], 1.0) * nx
        y = np.mod(uv[:, 1], 1.0) * ny
        x0 = np.floor(x).astype(np.int32) % nx
        y0 = np.floor(y).astype(np.int32) % ny
        x1, y1 = (x0 + 1) % nx, (y0 + 1) % ny
        fx, fy = x - np.floor(x), y - np.floor(y)
        return (
            field[y0, x0] * (1 - fx) * (1 - fy)
            + field[y0, x1] * fx * (1 - fy)
            + field[y1, x0] * (1 - fx) * fy
            + field[y1, x1] * fx * fy
        )

    def advance_tracers(self, trails, real, imag, magnetic_field, heating, solver_steps):
        """Advect passive tracers through the current simulated flow field."""
        flow_u, flow_v = self.flow_field(real, imag, magnetic_field, heating)
        head = trails[:, -1, :].copy()
        substeps = max(1, min(8, int(math.ceil(max(1, solver_steps) / 8))))
        dt = float(max(1, solver_steps)) / substeps
        sampled_speed = 0.0
        for _ in range(substeps):
            du = self._sample_periodic(flow_u, head)
            dv = self._sample_periodic(flow_v, head)
            head[:, 0] = np.mod(head[:, 0] + dt * du, 1.0)
            head[:, 1] = np.mod(head[:, 1] + dt * dv, 1.0)
            sampled_speed += float(np.mean(np.sqrt(du * du + dv * dv)))
        trails = np.roll(trails, -1, axis=1)
        trails[:, -1, :] = head
        return trails, sampled_speed / substeps

    @staticmethod
    def _project_torus(uv, size, angle):
        """Project normalised toroidal coordinates to screen x/y and depth."""
        uv = np.asarray(uv)
        u = uv[..., 0] * 2 * np.pi + angle
        v = uv[..., 1] * 2 * np.pi
        major, minor = 1.0, 0.40
        x = (major + minor * np.cos(v)) * np.cos(u)
        y = (major + minor * np.cos(v)) * np.sin(u)
        z = minor * np.sin(v)
        tilt = 0.92
        yy = y * math.cos(tilt) - z * math.sin(tilt)
        zz = y * math.sin(tilt) + z * math.cos(tilt)
        w, h = size
        scale = min(w / 3.15, h / 2.25)
        return w * 0.5 + x * scale, h * 0.52 - yy * scale, zz

    def _draw_tracers(self, image, trails, size, angle):
        """Draw luminous, depth-aware tracer ribbons over the torus field."""
        if trails is None or not len(trails):
            return image
        px, py, depth = self._project_torus(trails, size, angle)
        w, h = size
        glow = Image.new("RGBA", size, (0, 0, 0, 0))
        gd = ImageDraw.Draw(glow, "RGBA")
        sharp = Image.new("RGBA", size, (0, 0, 0, 0))
        sd = ImageDraw.Draw(sharp, "RGBA")
        colours = ((100, 239, 255), (255, 171, 83), (225, 126, 255), (188, 255, 228))
        steps = trails.shape[1]
        for particle in range(len(trails)):
            colour = colours[particle % len(colours)]
            for k in range(1, steps):
                x0, y0, x1, y1 = px[particle, k-1], py[particle, k-1], px[particle, k], py[particle, k]
                if abs(x1-x0) > w * 0.22 or abs(y1-y0) > h * 0.22:
                    continue
                age = k / max(1, steps - 1)
                front = np.clip(0.40 + 0.60 * (depth[particle, k] + 0.65) / 1.3, 0.28, 1.0)
                alpha = int((25 + 180 * age) * front)
                gd.line((x0, y0, x1, y1), fill=(*colour, max(12, alpha // 2)), width=7)
                sd.line((x0, y0, x1, y1), fill=(*colour, alpha), width=2 if front < 0.7 else 3)
            hx, hy = px[particle, -1], py[particle, -1]
            front = np.clip(0.45 + 0.55 * (depth[particle, -1] + 0.65) / 1.3, 0.32, 1.0)
            radius = 2.4 + 2.2 * front
            gd.ellipse((hx-radius*2.4, hy-radius*2.4, hx+radius*2.4, hy+radius*2.4), fill=(*colour, 105))
            sd.ellipse((hx-radius, hy-radius, hx+radius, hy+radius), fill=(245, 253, 255, int(225*front)), outline=(*colour, 245), width=1)
        glow = glow.filter(ImageFilter.GaussianBlur(4))
        return Image.alpha_composite(Image.alpha_composite(image.convert("RGBA"), glow), sharp).convert("RGB")

    def torus_image(self, real, imag, size=(820, 650), angle=0.0, compact=False, trails=None):
        """Project the simulated periodic field onto a luminous 3-D torus."""
        # Rendering is deliberately the CPU boundary; CuPy forbids implicit
        # conversion, so make the transfer explicit for GPU runs.
        real = to_numpy(real)
        imag = to_numpy(imag)
        ny, nx = real.shape
        amp = np.sqrt(real * real + imag * imag)
        phase = (np.arctan2(imag, real) + np.pi) / (2 * np.pi)
        texture = np.clip(0.62 * amp / 1.25 + 0.38 * phase, 0, 1)
        rgb = palette(texture, "plasma", normalize_input=False)

        # Sampling every second lattice row keeps the point renderer quick at
        # Leonardo resolutions while still deriving every colour from state.
        stride = max(1, int(math.ceil(ny / (80 if not compact else 45))))
        vv, uu = np.mgrid[0:ny:stride, 0:nx:stride]
        surface_uv = np.stack([uu.ravel() / nx, vv.ravel() / ny], axis=-1)
        px, py, zz = self._project_torus(surface_uv, size, angle)
        w, h = size
        cols = rgb[vv.ravel(), uu.ravel()].astype(np.float32)
        light = np.clip(0.45 + 0.55 * (zz + 1.1) / 2.2, 0.35, 1.0)[:, None]
        cols = np.clip(cols * light + np.array([12, 5, 25]), 0, 255).astype(np.uint8)

        order = np.argsort(zz)
        base = Image.new("RGB", size, (2, 4, 12))
        glow = Image.new("RGBA", size, (0, 0, 0, 0))
        gd = ImageDraw.Draw(glow, "RGBA")
        radius = max(1, int(min(w, h) / (260 if compact else 285)))
        for j in order:
            c = tuple(int(q) for q in cols[j])
            rr = radius + (1 if zz[j] > 0.25 and not compact else 0)
            gd.ellipse((px[j] - rr * 2, py[j] - rr * 2, px[j] + rr * 2, py[j] + rr * 2), fill=(*c, 38))
        halo = glow.filter(ImageFilter.GaussianBlur(max(2, radius * 3)))
        base = Image.alpha_composite(base.convert("RGBA"), halo)
        sharp = Image.new("RGBA", size, (0, 0, 0, 0))
        sd = ImageDraw.Draw(sharp, "RGBA")
        for j in order:
            c = tuple(int(q) for q in cols[j])
            rr = radius + (1 if zz[j] > 0.25 and not compact else 0)
            sd.ellipse((px[j] - rr, py[j] - rr, px[j] + rr, py[j] + rr), fill=(*c, 145 if not compact else 190))
        result = Image.alpha_composite(base, sharp).convert("RGB")
        return self._draw_tracers(result, trails, size, angle) if not compact else result

    def hero(self, real, imag, frame, done, total, b, heating, density, trails=None, tracer_speed=0.0):
        canvas = self.torus_image(real, imag, size=(1280,720), angle=frame * 0.010, trails=trails)
        r, im = to_numpy(real), to_numpy(imag)
        score = self.turbulence(r, im)
        canvas = add_title(
            canvas,
            "Star in a Bottle",
            f"field-driven plasma tracers · {r.shape[1]}×{r.shape[0]} periodic lattice · solver step {done:,}/{total:,} · {self.ctx.backend_name}",
            badge="LIVE FIELD + FLOW",
        )
        add_progress(canvas, done / total, "COHERENT WAVES", "TURBULENT PLASMA")
        return canvas

    def write_interactive_view(self, real, imag, trails, magnetic_field, heating, density, frame=None):
        """Persist a compact state for the browser's live rotatable canvas."""
        real, imag = to_numpy(real), to_numpy(imag)
        ny, nx = real.shape
        stride = max(1, int(math.ceil(ny / 96)))
        real, imag = real[::stride, ::stride], imag[::stride, ::stride]
        amp = np.sqrt(real * real + imag * imag)
        phase = (np.arctan2(imag, real) + np.pi) / (2 * np.pi)
        texture = np.clip(0.62 * amp / 1.25 + 0.38 * phase, 0, 1)
        # A simple pitch relationship is used only for the explanatory
        # magnetic-geometry view. It is not a solved equilibrium or q profile.
        pitch = float(np.clip(0.48 + 0.045 * float(magnetic_field), 0.5, 0.9))
        manifest = {
            "version": 1,
            "kind": "fusion-torus",
            "shape": [int(texture.shape[0]), int(texture.shape[1])],
            "texture": np.rint(texture * 255).astype(np.uint8).ravel().tolist(),
            "trails": np.round(np.asarray(trails, dtype=np.float32), 5).tolist(),
            "magnetic_field": float(magnetic_field),
            "heating": float(heating),
            "density": float(density),
            "field_lines": 14,
            "field_pitch": pitch,
            "note": "Magnetic lines are illustrative helical confinement geometry, not a solved equilibrium.",
        }
        if frame is None:
            path = self.ctx.run_dir / "fusion_view.json"
        else:
            path = self.ctx.run_dir / "modes" / "fusion3d" / f"frame_{int(frame):04d}.json"
            path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(manifest, separators=(",", ":")), encoding="utf-8")
        if frame == 0:
            self.ctx.write_meta({"fusion_view":{"folder":"modes/fusion3d","fallback":"fusion_view.json"},
                                 "default_view_mode":"fusion3d"})
        return path

    def run(self):
        ny, nx = self.grid_shape(self.settings["n"])
        total = self.budget()
        b = float(self.ctx.params.get("magnetic_field", 5.0))
        heating = float(self.ctx.params.get("heating", 25.0))
        density = float(self.ctx.params.get("density", 1.0))
        real, imag = self.initialise(ny, nx)
        trails = self.initialise_tracers(
            int(self.settings.get("tracers", 180)),
            int(self.settings.get("trail", 12)),
        )
        done = 0
        for i in range(self.ctx.frames):
            target = int(round(total * (i + 1) / self.ctx.frames))
            solver_steps = max(1, target - done)
            real, imag = self.step(real, imag, b, heating, density, solver_steps)
            done = target
            trails, tracer_speed = self.advance_tracers(trails, real, imag, b, heating, solver_steps)
            image = self.hero(real, imag, i, done, total, b, heating, density, trails, tracer_speed)
            self.ctx.save_frame(image, self.ctx.frame_path(i))
            self.write_interactive_view(real, imag, trails, b, heating, density, frame=i)
            score=self.turbulence(to_numpy(real),to_numpy(imag)); regime="phase turbulence" if score>.075 else "coherent confinement"
            self.ctx.write_status(i, f"plasma lattice step {done:,} · {len(trails):,} tracers",{
                "magnetic field":f"{b:.1f} T","heating power":f"{heating:.0f} MW","density":f"{density:.2f} n₀",
                "regime":regime,"turbulence index":f"{score:.3f}","passive tracers":f"{len(trails):,}",
                "mean drift":f"{tracer_speed*360:.3f}° / step","solver step":f"{done:,} / {total:,}"})

        # Retain one final-state file so older viewers can still open new runs.
        self.write_interactive_view(real, imag, trails, b, heating, density)

        ens = max(4, int(self.settings.get("ensemble", 16)))
        side = max(2, int(math.sqrt(ens)))
        sweep_n = int(self.settings.get("sweep_n", max(36, ny // 2)))
        my, mx = self.grid_shape(sweep_n)
        sweep_steps = int(self.settings.get("sweep_steps", max(160, total // 3)))
        tiles, labels = [], []
        b_values = np.linspace(2.5, 7.5, side)
        h_values = np.linspace(10.0, 45.0, side)
        for row, hj in enumerate(h_values):
            for col, bj in enumerate(b_values):
                rr, ii = self.initialise(my, mx, seed=40 + row * side + col)
                rr, ii = self.step(rr, ii, float(bj), float(hj), density, sweep_steps)
                rn, inn = to_numpy(rr), to_numpy(ii)
                tiles.append(self.torus_image(rn, inn, size=(260, 146), angle=0.35, compact=True))
                labels.append(f"B {bj:.1f}T · {hj:.0f}MW")
        reveal = mosaic(
            tiles,
            side,
            title="Actually… we tested a whole reactor operating map",
            subtitle="Magnetic field increases left to right; heating increases top to bottom. Every torus is a separate run.",
            labels=labels,
            label_fill=(180, 238, 255),
        )
        rp = self.ctx.run_dir / "reveal.jpg"
        self.ctx.save_frame(reveal, rp)
        self.ctx.finish(rp)
