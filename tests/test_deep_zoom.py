import unittest

import numpy as np

from leonardo_demos.colors import palette
from leonardo_demos.crystal_growth import build_crystal, generate, render_window


def _structure(im):
    """Fraction of pixels differing from the image's own most common value."""
    a = np.asarray(im.convert('L'))
    counts = np.bincount(a.ravel(), minlength=256)
    return 1.0 - counts.max() / a.size


class DeepZoomTests(unittest.TestCase):
    def setUp(self):
        self.crystal = build_crystal(depth=5)
        self.span = self.crystal['extent'] * 2.35

    def test_geometry_survives_extreme_magnification(self):
        """Depth must follow the zoom, not a baked level count."""
        for factor in (1, 100, 10_000, 1_000_000):
            geo = generate(0.18, 0.0, self.span / factor, size_px=200, max_segments=4000)
            self.assertGreater(len(geo['x0']), 50, f'no geometry at {factor}x')

    def test_deeper_zoom_reveals_new_geometry(self):
        """Successive magnifications must not be the same segments rescaled."""
        coarse = generate(0.18, 0.0, self.span / 50, size_px=200, max_segments=4000)
        fine = generate(0.18, 0.0, self.span / 5000, size_px=200, max_segments=4000)
        self.assertLess(fine['w'].min(), coarse['w'].min(),
                        'finer window produced no thinner branches')

    def test_rendered_deep_window_is_not_blank(self):
        """A regression guard for two separate blanking bugs.

        `birth` is normalised inside whatever set was generated, so a windowed
        generation left every segment near birth=1 and the growth interpolation
        drew them as stubs. Separately, palette() renormalised an already-unit
        field, turning a legitimately solid view into flat background colour.
        """
        geo = generate(0.372, 0.075, self.span / 400, size_px=200, max_segments=6000)
        im = render_window(geo, 0.372, 0.075, self.span / 400, size=(200, 200),
                           progress=1.0, supersample=1)
        self.assertGreater(_structure(im), 0.02, 'deep zoom rendered as a flat field')

    def test_palette_preserves_a_uniform_unit_field(self):
        rgb = palette(np.ones((8, 8), dtype=np.float32), 'ice', normalize_input=False)
        self.assertGreater(int(rgb.max()), 200,
                           'a saturated field must not be rescaled to background')

    def test_growth_animation_still_runs_for_partial_progress(self):
        early = render_window(self.crystal, 0, 0, self.span, size=(120, 120),
                              progress=0.15, supersample=1)
        late = render_window(self.crystal, 0, 0, self.span, size=(120, 120),
                             progress=1.0, supersample=1)
        self.assertLess(_structure(early), _structure(late),
                        'crystal should cover more of the frame as it grows')


if __name__ == '__main__':
    unittest.main()
