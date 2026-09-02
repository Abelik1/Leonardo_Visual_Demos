from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from ..backend import to_numpy
from ..base import Demo
from ..render import mosaic


HUBBLE_DEEP_FIELD = Path(__file__).resolve().parents[2] / "web" / "assets" / "hubble_deep_field.jpg"


class BlackHoleDemo(Demo):
    """Educational 2-D lens mapping plus numerically integrated 3-D rays."""

    id = "black_hole"
    title = "Black-hole lensing"
    timing_methods = {"background": "initialization", "lens": "simulation",
                      "integrate_rays": "simulation", "render_3d": "render"}

    def background(self, width, height, phase=0.0):
        """A moving crop of NASA's recorded Hubble Deep Field source plane."""
        if not HUBBLE_DEEP_FIELD.exists():
            raise FileNotFoundError("Missing Hubble Deep Field asset: web/assets/hubble_deep_field.jpg")
        with Image.open(HUBBLE_DEEP_FIELD) as source:
            source = source.convert("RGB")
            sw, sh = source.size
            crop_w = max(2, int(sw * .78))
            crop_h = max(2, int(crop_w * height / max(1, width)))
            if crop_h > sh:
                crop_h = max(2, int(sh * .78))
                crop_w = max(2, int(crop_h * width / max(1, height)))
            crop_w, crop_h = min(crop_w, sw), min(crop_h, sh)
            dx, dy = sw - crop_w, sh - crop_h
            p = float(phase) * 2 * math.pi
            left = int(round(dx * (.50 + .34 * math.sin(p))))
            top = int(round(dy * (.50 + .30 * math.cos(.83 * p))))
            image = source.crop((left, top, left + crop_w, top + crop_h))
            image = image.resize((int(width), int(height)), Image.Resampling.LANCZOS)
        return np.asarray(image, dtype=np.float32) / 255.0

    @staticmethod
    def lens_wells(params):
        count = max(1, min(3, int(round(float(params.get("lens_count", 1))))))
        x = float(params.get("lens_x", 0.0)) * 1.25
        y = float(params.get("lens_y", 0.0)) * .82
        separation = float(params.get("lens_separation", .62))
        angle = math.radians(float(params.get("lens_angle", 0.0)))
        wells = [(x, y, 1.0)]
        if count == 2:
            wells.append((x + separation * math.cos(angle), y + separation * math.sin(angle), .62))
        elif count == 3:
            for offset in (-math.pi / 3, math.pi / 3):
                wells.append((x + separation * math.cos(angle + offset),
                              y + separation * math.sin(angle + offset), .52))
        return wells

    def lens(self, background, mass, spin, progress=1.0, wells=None):
        """Map observer pixels through one or more weak-field point lenses."""
        xp = self.ctx.xp
        source = xp.asarray(background)
        height, width = background.shape[:2]
        y, x = xp.mgrid[0:height, 0:width]
        xx = (x - width * .5) / (height * .5)
        yy = (y - height * .5) / (height * .5)
        amount = .15 + .85 * float(progress)
        mapped_x, mapped_y = xx.copy(), yy.copy()
        capture = xp.zeros((height, width), dtype=bool)
        ring = xp.zeros((height, width), dtype=source.dtype)
        wells = wells or [(0.0, 0.0, 1.0)]
        for well_x, well_y, relative_mass in wells:
            dx, dy = xx - well_x, yy - well_y
            radius2 = dx * dx + dy * dy + 1e-4
            local_mass = float(mass) * float(relative_mass)
            einstein = .16 * local_mass * amount
            mapped_x -= einstein * einstein * dx / radius2
            mapped_y -= einstein * einstein * dy / radius2
            # Qualitative spin-like twist; this is not Kerr geodesic tracing.
            shear = .025 * spin * relative_mass * amount / (radius2 + .06)
            mapped_x -= shear * dy
            mapped_y += shear * dx
            radius = xp.sqrt(radius2)
            capture |= radius < (.055 + .018 * local_mass) * amount
            ring += xp.exp(-((radius - (.105 + .020 * local_mass)) / .018) ** 2) * amount
        if not getattr(self,"show_disk",True):
            ring *= 0
        sx = xp.clip((mapped_x * (height * .5) + width * .5).astype(xp.int32), 0, width - 1)
        sy = xp.clip((mapped_y * (height * .5) + height * .5).astype(xp.int32), 0, height - 1)
        output = source[sy, sx]
        output = xp.where(capture[..., None], 0, output)
        colour = xp.stack((1.5 * ring, .52 * ring, .12 * ring), axis=-1)
        return to_numpy(xp.clip(output + colour, 0, 1))

    @staticmethod
    def integrate_rays(mass, spin, wells=None, count=81, samples=150):
        """Advance photon directions through a reduced 3-D deflection field.

        This is intentionally labelled as a weak-field educational integrator,
        not a Schwarzschild/Kerr null-geodesic solver.  The resulting lines are
        nevertheless computed trajectories, including capture at a finite
        radius, rather than authored curves.
        """
        wells = wells or [(0.0, 0.0, 1.0)]
        side = max(3, int(math.ceil(math.sqrt(count))))
        axis = np.linspace(-1.55, 1.55, side)
        target = [(x, y, 4.2) for y in axis for x in axis]
        for well_x, well_y, _ in wells:
            target.extend((well_x + radius * math.cos(angle), well_y + radius * math.sin(angle), 4.2)
                          for radius in (.20, .43, .70) for angle in np.linspace(0, 2 * math.pi, 12, endpoint=False))
        target = np.asarray(target[:count], dtype=np.float64)
        position = np.repeat(np.array([[0.0, 0.0, -4.2]]), len(target), axis=0)
        direction = target - position
        direction /= np.linalg.norm(direction, axis=1, keepdims=True)
        paths = np.empty((samples, len(target), 3), dtype=np.float32)
        captured = np.zeros(len(target), dtype=bool)
        step = 8.8 / max(1, samples - 1)
        for sample in range(samples):
            paths[sample] = position
            active = ~captured
            if not active.any():
                paths[sample:] = position
                break
            p = position[active]
            d = direction[active]
            bend = np.zeros_like(p)
            hit = np.zeros(len(p), dtype=bool)
            for well_x, well_y, relative_mass in wells:
                offset = p - np.array((well_x, well_y, 0.0))
                radius2 = np.sum(offset * offset, axis=1) + .035
                # Only acceleration perpendicular to the current photon
                # direction changes it in this reduced weak-field model.
                radial = offset - d * np.sum(offset * d, axis=1, keepdims=True)
                local_mass = float(mass) * float(relative_mass)
                bend += -.055 * local_mass * radial / radius2[:, None] ** 1.5
                hit |= np.sqrt(np.sum(offset * offset, axis=1)) < (.16 + .035 * local_mass)
            radius2 = np.sum(p * p, axis=1) + .035
            transverse = np.column_stack((-p[:, 1], p[:, 0], np.zeros(len(p))))
            bend += .006 * float(spin) * transverse / (radius2[:, None] + .15)
            d += bend * step
            d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9)
            p += d * step
            direction[active] = d
            position[active] = p
            newly = active.copy()
            newly[active] = hit
            captured |= newly
        return paths, captured

    @staticmethod
    def _project(points):
        yaw, pitch = math.radians(-31), math.radians(18)
        cy, sy = math.cos(yaw), math.sin(yaw)
        cp, sp = math.cos(pitch), math.sin(pitch)
        x = points[..., 0] * cy + points[..., 2] * sy
        z = -points[..., 0] * sy + points[..., 2] * cy
        y = points[..., 1] * cp - z * sp
        depth = points[..., 1] * sp + z * cp
        scale = 98.0 / np.maximum(1.0, 1.0 + .035 * depth)
        return 640 + x * scale, 360 - y * scale

    def draw_source_plane(self, draw, background):
        """Map the same Hubble crop used by 2-D onto the 3-D source plane."""
        rows, cols = 16, 24
        height, width = background.shape[:2]
        for row in range(rows):
            for col in range(cols):
                x0, x1 = -1.8 + 3.6 * col / cols, -1.8 + 3.6 * (col + 1) / cols
                y0, y1 = -1.8 + 3.6 * row / rows, -1.8 + 3.6 * (row + 1) / rows
                corners = np.array(((x0, y0, 4.2), (x1, y0, 4.2),
                                    (x1, y1, 4.2), (x0, y1, 4.2)))
                px, py = self._project(corners)
                sample = background[min(height - 1, int((row + .5) * height / rows)),
                                    min(width - 1, int((col + .5) * width / cols))]
                draw.polygon(tuple(zip(px, py)), fill=(*[int(value * 255) for value in sample], 230))
        corners = np.array(((-1.8, -1.8, 4.2), (1.8, -1.8, 4.2),
                            (1.8, 1.8, 4.2), (-1.8, 1.8, 4.2), (-1.8, -1.8, 4.2)))
        px, py = self._project(corners)
        draw.line(tuple(zip(px, py)), fill=(135, 213, 255, 180), width=2)

    def render_3d(self, paths, captured, mass, wells, background, progress):
        image = Image.new("RGB", (1280, 720), (2, 5, 15))
        draw = ImageDraw.Draw(image, "RGBA")
        self.draw_source_plane(draw, background)
        # The observer grid makes the shared source-plane geometry legible.
        for z, colour in ((-4.2, (90, 230, 205, 85)),):
            grid=[]
            for value in np.linspace(-1.8, 1.8, 7):
                grid.extend((np.array(((-1.8, value, z), (1.8, value, z))),
                             np.array(((value, -1.8, z), (value, 1.8, z)))))
            for line in grid:
                xx, yy = self._project(line)
                draw.line(tuple(zip(xx, yy)), fill=colour, width=1)
        glow = Image.new("RGBA", image.size, (0, 0, 0, 0))
        gd = ImageDraw.Draw(glow, "RGBA")
        for well_x, well_y, relative_mass in wells:
            centre_x, centre_y = self._project(np.array([[well_x, well_y, 0.0]]))
            radius = (18 + 7 * mass) * relative_mass
            cx, cy = float(centre_x[0]), float(centre_y[0])
            gd.ellipse((cx-radius*2.2, cy-radius*2.2, cx+radius*2.2, cy+radius*2.2),
                       fill=(255, 104, 35, 72))
        glow = glow.filter(ImageFilter.GaussianBlur(17))
        image = Image.alpha_composite(image.convert("RGBA"), glow).convert("RGB")
        draw = ImageDraw.Draw(image, "RGBA")
        for well_x, well_y, relative_mass in wells:
            centre_x, centre_y = self._project(np.array([[well_x, well_y, 0.0]]))
            radius = (18 + 7 * mass) * relative_mass
            cx, cy = float(centre_x[0]), float(centre_y[0])
            draw.ellipse((cx-radius, cy-radius, cx+radius, cy+radius), fill=(0, 0, 0, 255),
                         outline=(255, 142, 62, 225), width=3)
        visible = max(2, min(len(paths), int(2 + progress * (len(paths) - 2))))
        for ray in range(paths.shape[1]):
            xx, yy = self._project(paths[:visible, ray])
            colour = (255, 116, 70, 145) if captured[ray] else (86, 220, 255, 135)
            draw.line(tuple(zip(xx, yy)), fill=colour, width=2)
            packet = min(visible - 1, max(0, int((progress * 1.9 % 1) * visible)))
            draw.ellipse((xx[packet]-3, yy[packet]-3, xx[packet]+3, yy[packet]+3),
                         fill=(255, 244, 190, 230))
        return image

    def run(self):
        width = int(self.settings["width"])
        height = int(self.settings["height"])
        mass = float(self.ctx.params.get("mass", 1.35))
        spin = float(self.ctx.params.get("spin", .55))
        wells = self.lens_wells(self.ctx.params)
        self.show_disk = bool(round(float(self.ctx.params.get("disk",1))))
        paths, captured = self.integrate_rays(mass, spin, wells=wells)
        mode_dir = self.ctx.run_dir / "modes" / "3d"
        mode_dir.mkdir(parents=True, exist_ok=True)
        self.ctx.write_meta({
            "view_modes": [
                {"id": "frames", "label": "2D observer image", "folder": "frames"},
                {"id": "3d", "label": "3D ray space", "folder": "modes/3d"},
            ],
            "default_view_mode": "frames",
            "source_plane": {"name":"Hubble Deep Field (PIA12110)",
                             "file":"web/assets/hubble_deep_field.jpg",
                             "credit":"NASA/JPL-Caltech/STScI"},
            "lens_wells": [{"x":x,"y":y,"relative_mass":weight}
                           for x,y,weight in wells],
            "physics": {"3d": "reduced weak-field numerical photon integration",
                        "2d": "parallel thin-lens image mapping",
                        "limitation": "not a Kerr geodesic or GRMHD solver"},
        })
        for frame in range(self.ctx.frames):
            progress = (frame + 1) / self.ctx.frames
            background = self.background(width,height,phase=.22*progress)
            observer = Image.fromarray((self.lens(background,mass,spin,progress,wells) * 255).astype(np.uint8))
            observer = observer.resize((1280, 720), Image.Resampling.LANCZOS)
            observer = Image.blend(observer, observer.filter(ImageFilter.GaussianBlur(9)), .10)
            self.ctx.save_frame(observer, self.ctx.frame_path(frame))
            self.ctx.save_frame(self.render_3d(paths,captured,mass,wells,background,progress),
                                mode_dir / f"frame_{frame:04d}.jpg")
            self.ctx.write_status(frame, "Integrating photon paths through 3-D space", {
                "lens mass": f"{mass:.2f}", "dimensionless spin": f"{spin:+.2f}",
                "observer sight lines": f"{width * height:,}", "3D rays": f"{paths.shape[1]}",
                "integration samples": f"{paths.shape[0]}",
                "captured rays": f"{captured.sum()} / {len(captured)}",
            })
        ensemble = max(1, int(self.ctx.params.get("_parallel_count", self.settings.get("ensemble", 16))))
        side = max(1, int(math.ceil(math.sqrt(ensemble))))
        tiles=[]
        small_background = self.background(260, 150, phase=.42)
        for j in range(ensemble):
            trial_mass = max(.45, mass * (.65 + .7 * (j % side) / max(1, side - 1)))
            trial_spin = -.9 + 1.8 * (j // side) / max(1, side - 1)
            tiles.append(Image.fromarray((self.lens(small_background,trial_mass,trial_spin,wells=wells)*255).astype(np.uint8)))
        reveal = mosaic(tiles, side, title="That was one observer. Leonardo can explore many.")
        reveal_path = self.ctx.run_dir / "reveal.jpg"
        self.ctx.save_frame(reveal, reveal_path)
        self.ctx.finish(reveal_path)
