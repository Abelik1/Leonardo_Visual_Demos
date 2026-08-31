import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from leonardo_demos.base import RunContext


class HybridPipelineTests(unittest.TestCase):
    def test_hybrid_overlaps_writes_and_finishes_with_complete_files(self):
        """Hybrid mode must drain CPU work before advertising a finished run."""
        with tempfile.TemporaryDirectory() as temp, patch(
            "leonardo_demos.base.choose_backend",
            return_value=(np, "cupy + CPU frame workers"),
        ):
            ctx = RunContext(Path(temp), "reaction_diffusion", "local", 2, {}, "hybrid")
            ctx.save_frame(Image.new("RGB", (12, 12), "red"), ctx.frame_path(0))
            ctx.save_frame(Image.new("RGB", (12, 12), "blue"), ctx.frame_path(1))
            reveal = ctx.run_dir / "reveal.jpg"
            ctx.save_frame(Image.new("RGB", (12, 12), "green"), reveal)
            ctx.finish(reveal)

            self.assertTrue(ctx.frame_path(0).exists())
            self.assertTrue(ctx.frame_path(1).exists())
            self.assertTrue(reveal.exists())
            meta = json.loads((ctx.run_dir / "meta.json").read_text())
            self.assertEqual(meta["status"], "complete")
            self.assertEqual(meta["backend"], "cupy + CPU frame workers")


if __name__ == "__main__":
    unittest.main()
