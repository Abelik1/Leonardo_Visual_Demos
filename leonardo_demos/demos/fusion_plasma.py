from __future__ import annotations

import math

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from ..backend import to_numpy
from ..base import Demo
from ..colors import palette
from ..render import add_progress, add_title, font, mosaic, save_frame


class FusionPlasmaDemo(Demo):
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

    def torus_image(self, real, imag, size=(820, 650), angle=0.0, compact=False):
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
        u = uu.ravel() / nx * 2 * np.pi + angle
        v = vv.ravel() / ny * 2 * np.pi
        R, r = 1.0, 0.40
        x = (R + r * np.cos(v)) * np.cos(u)
        y = (R + r * np.cos(v)) * np.sin(u)
        z = r * np.sin(v)
        # Tilt the torus so both the central hole and far-side plasma are clear.
        tilt = 0.92
        yy = y * math.cos(tilt) - z * math.sin(tilt)
        zz = y * math.sin(tilt) + z * math.cos(tilt)
        w, h = size
        scale = min(w / 3.15, h / 2.25)
        px = w * 0.5 + x * scale
        py = h * 0.52 - yy * scale
        cols = rgb[vv.ravel(), uu.ravel()].astype(np.float32)
        light = np.clip(0.45 + 0.55 * (zz + 1.1) / 2.2, 0.35, 1.0)[:, None]
        cols = np.clip(cols * light + np.array([12, 5, 25]), 0, 255).astype(np.uint8)

        order = np.argsort(zz)
        base = Image.new("RGB", size, (2, 4, 12))
        glow = Image.new("RGBA", size, (0, 0, 0, 0))
        gd = ImageDraw.Draw(glow, "RGBA")
        radius = max(1, int(min(w, h) / (230 if compact else 210)))
        for j in order:
            c = tuple(int(q) for q in cols[j])
            rr = radius + (1 if zz[j] > 0.25 and not compact else 0)
            gd.ellipse((px[j] - rr * 2, py[j] - rr * 2, px[j] + rr * 2, py[j] + rr * 2), fill=(*c, 52))
        halo = glow.filter(ImageFilter.GaussianBlur(max(2, radius * 3)))
        base = Image.alpha_composite(base.convert("RGBA"), halo)
        sharp = Image.new("RGBA", size, (0, 0, 0, 0))
        sd = ImageDraw.Draw(sharp, "RGBA")
        for j in order:
            c = tuple(int(q) for q in cols[j])
            rr = radius + (1 if zz[j] > 0.25 and not compact else 0)
            sd.ellipse((px[j] - rr, py[j] - rr, px[j] + rr, py[j] + rr), fill=(*c, 205))
        return Image.alpha_composite(base, sharp).convert("RGB")

    def hero(self, real, imag, frame, done, total, b, heating, density):
        torus = self.torus_image(real, imag, angle=frame * 0.018)
        canvas = Image.new("RGB", (1280, 720), (2, 4, 12))
        canvas.paste(torus, (0, 70))
        r, im = to_numpy(real), to_numpy(imag)
        score = self.turbulence(r, im)
        regime = "PHASE TURBULENCE" if score > 0.075 else "COHERENT CONFINEMENT"
        d = ImageDraw.Draw(canvas, "RGBA")
        d.rounded_rectangle((858, 126, 1248, 486), radius=22, fill=(5, 9, 24, 225), outline=(91, 224, 244, 90), width=2)
        d.text((890, 152), "TOKAMAK CONTROL", font=font(17, True), fill=(139, 239, 255))
        d.text((890, 196), f"magnetic field   {b:.1f} T", font=font(22, True), fill="white")
        d.text((890, 236), f"heating power    {heating:.0f} MW", font=font(22, True), fill=(255, 185, 102))
        d.text((890, 276), f"density          {density:.2f} n0", font=font(22, True), fill=(189, 207, 244))
        d.line((890, 326, 1214, 326), fill=(255, 255, 255, 38), width=1)
        d.text((890, 350), regime, font=font(18, True), fill=(255, 116, 134) if score > 0.075 else (108, 241, 205))
        d.text((890, 384), f"turbulence index  {score:.3f}", font=font(16), fill=(195, 218, 247))
        d.text((890, 416), f"solver step       {done:,}/{total:,}", font=font(16), fill=(195, 218, 247))
        d.text((890, 448), f"lattice updates   {done * r.size:,}", font=font(16), fill=(120, 230, 255))
        canvas = add_title(
            canvas,
            "Star in a Bottle",
            f"reduced nonlinear plasma-wave model · {r.shape[1]}×{r.shape[0]} periodic lattice · {self.ctx.backend_name}",
            badge="LIVE PLASMA MODEL",
        )
        add_progress(canvas, done / total, "COHERENT WAVES", "TURBULENT PLASMA")
        return canvas

    def run(self):
        ny, nx = self.grid_shape(self.settings["n"])
        total = self.budget()
        b = float(self.ctx.params.get("magnetic_field", 5.0))
        heating = float(self.ctx.params.get("heating", 25.0))
        density = float(self.ctx.params.get("density", 1.0))
        real, imag = self.initialise(ny, nx)
        done = 0
        for i in range(self.ctx.frames):
            target = int(round(total * (i + 1) / self.ctx.frames))
            real, imag = self.step(real, imag, b, heating, density, max(1, target - done))
            done = target
            image = self.hero(real, imag, i, done, total, b, heating, density)
            self.ctx.save_frame(image, self.ctx.frame_path(i))
            self.ctx.write_status(i, f"plasma lattice step {done:,}")

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
