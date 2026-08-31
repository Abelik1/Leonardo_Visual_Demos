"""Skeleton for a new demo. Copy this file, rename the class, register it.

Everything here exists because something in this repository once got it wrong.
Read docs/DEMO_TEMPLATE.md for the reasoning; the comments below mark the rules
that are easy to break silently.

To add a demo:

  1. cp _template.py my_demo.py         and rename the class + `id`
  2. register it in leonardo_demos/registry.py
  3. add a params block to config/demo_specs.json
  4. add a settings block to all three profiles in config/profiles.json
  5. add a story line to `stories` in web/app.js
  6. add a smoke test to tests/test_small_demos.py
  7. write its entry in docs/SCIENTIFIC_NOTES.md, honestly
"""
from __future__ import annotations

import numpy as np
from PIL import ImageDraw

from ..base import Demo
from ..backend import to_numpy
from ..render import add_title, add_progress, array_image, font, mosaic, save_frame


class TemplateDemo(Demo):
    # `id` must match the key used in demo_specs.json and profiles.json.
    id = "template"
    title = "Template demo"

    # Frame size is fixed by the viewer. Simulate at this aspect ratio rather
    # than stretching a square field into it: a square grid resized to 16:9
    # turns every circular feature into an ellipse.
    aspect = 16 / 9

    def grid_shape(self, n):
        ny = max(16, int(n))
        return ny, max(16, int(round(ny * self.aspect)))

    def budget(self):
        """Total solver steps for the run, independent of the frame count.

        Physical maturity and frame count are different things. Deriving work
        from `frames * steps_per_frame` means asking for fewer frames silently
        produces an unfinished simulation - which is exactly how a Gray-Scott
        run ended up stopping at 4% of the domain covered.

        Keep the `steps_per_frame` fallback so old profiles still load.
        """
        s = self.settings
        if "total_steps" in s:
            return max(1, int(s["total_steps"]))
        return max(1, int(s.get("steps_per_frame", 8)) * self.ctx.frames)

    # ---- simulation ---------------------------------------------------
    def initialise(self, ny, nx, seed=0):
        """Return the initial state.

        Use `self.ctx.xp` for every array operation so the same source runs on
        NumPy or CuPy. Seed any randomness explicitly: a run must be
        reproducible from its meta.json.
        """
        xp = self.ctx.xp
        rng = np.random.default_rng(seed)
        state = xp.asarray(rng.random((ny, nx)).astype(np.float32))
        return state

    def step(self, state, steps):
        """Advance the state by `steps`. No rendering, no overlays, no I/O."""
        xp = self.ctx.xp
        for _ in range(steps):
            state = 0.25 * (
                xp.roll(state, 1, 0) + xp.roll(state, -1, 0)
                + xp.roll(state, 1, 1) + xp.roll(state, -1, 1)
            )
        return state

    # ---- rendering ----------------------------------------------------
    def frame(self, state, i, frames):
        """Turn state into one frame. Never change the science to suit the camera.

        `array_image` normalises its input. That is right for a field with a
        meaningful dynamic range and wrong for one already scaled to [0, 1]:
        renormalising a legitimately uniform field maps it to zero, which once
        turned a solid white crystal zoom into solid background colour.
        """
        im = array_image(to_numpy(state), "plasma", size=(1280, 720))
        im = add_title(
            im,
            self.title,
            f"what the solver is · {self.ctx.backend_name}",
        )
        d = ImageDraw.Draw(im, "RGBA")
        d.rounded_rectangle((26, 112, 360, 166), radius=14, fill=(4, 8, 20, 180))
        d.text((44, 126), "One sentence a visitor can read", font=font(17, True), fill="white")
        add_progress(im, (i + 1) / frames, "START STATE", "END STATE")
        return im

    # ---- the run ------------------------------------------------------
    def run(self):
        ny, nx = self.grid_shape(self.settings["n"])
        frames = self.ctx.frames
        total = self.budget()
        state = self.initialise(ny, nx)

        done = 0
        for i in range(frames):
            # Distribute the step budget across frames rather than running a
            # fixed number per frame, so the end state is the same whatever the
            # frame count.
            target = int(round(total * (i + 1) / frames))
            state = self.step(state, max(1, target - done))
            done = target

            self.ctx.save_frame(self.frame(state, i, frames), self.ctx.frame_path(i))
            # write_status is what makes a partially finished run usable, which
            # is what lets the viewer stream frames synced from Leonardo.
            self.ctx.write_status(i, f"step {done:,}")

        # ---- the reveal ----
        # This must be REAL additional computation. The whole exhibition story
        # is "that was not the whole calculation"; a grid of decorative copies
        # of the last frame would make that claim false.
        ens = int(self.settings.get("ensemble", 16))
        side = max(2, int(np.sqrt(ens)))
        sweep = int(self.settings.get("sweep_steps", max(200, total // 4)))
        my, mx = self.grid_shape(max(48, ny // 3))
        tiles, labels = [], []
        for j in range(side * side):
            s = self.step(self.initialise(my, mx, seed=j + 1), sweep)
            tiles.append(array_image(to_numpy(s), "plasma", size=(260, 146)))
            labels.append(f"case {j:02d}")
        rev = mosaic(
            tiles, side,
            title="Actually… we ran many of these",
            subtitle="Say what varies across the tiles.",
            labels=labels,
        )
        rp = self.ctx.run_dir / "reveal.jpg"
        self.ctx.save_frame(rev, rp)
        self.ctx.finish(rp)
