import unittest

import numpy as np

from leonardo_demos.colors import palette
from leonardo_demos.crystal_growth import build_crystal, generate, generate_stable, render_window


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

    def test_jittered_adjacent_tiles_share_branch_geometry(self):
        """Noise must be tied to a branch path, never tile traversal order."""
        options=dict(size_px=220, mode='seaweed', seed=7, max_segments=20_000)
        left=generate(-.12,.05,.24,**options)
        right=generate(.12,.05,.24,**options)

        def signatures(geometry):
            return {tuple(round(float(value),10) for value in row)
                    for row in zip(geometry['x0'],geometry['y0'],geometry['x1'],
                                   geometry['y1'],geometry['w'])}

        # Windows meet at x=0, but both must contain the same substantial set
        # of branches whose subtrees cross that boundary.
        self.assertGreater(len(signatures(left)&signatures(right)),300)

    def test_stable_grammar_only_adds_detail_between_levels(self):
        """A LOD transition must never replace its coarser crystal.

        The old viewport-local priority queue could choose a different set of
        branches at each zoom.  The stable grammar's depth d must be an exact
        subset of depth d+1, so colour blending can only fade in a residual.
        """
        options=dict(cx=.11,cy=-.04,span=.34,size_px=240,mode='coral',seed=7,
                     max_depth=12)
        coarse=generate_stable(**options,detail_depth=7)
        fine=generate_stable(**options,detail_depth=8)

        def signatures(geometry):
            return {tuple(round(float(value),12) for value in row)
                    for row in zip(geometry['x0'],geometry['y0'],geometry['x1'],
                                   geometry['y1'],geometry['w'])}

        self.assertTrue(signatures(coarse) <= signatures(fine))
        self.assertGreater(len(signatures(fine)-signatures(coarse)),0)

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
