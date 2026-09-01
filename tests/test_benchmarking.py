import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from leonardo_demos.base import RunContext
from leonardo_demos.registry import DEMOS
from run_demo import load_profiles


class BenchmarkingContractTests(unittest.TestCase):
    def test_benchmark_profile_covers_every_registered_demo(self):
        self.assertEqual(set(load_profiles()["benchmark"]),set(DEMOS))

    def test_cpu_only_demos_do_not_advertise_gpu(self):
        self.assertEqual(DEMOS["pbh"].supported_backends,("cpu",))
        self.assertEqual(DEMOS["crystal"].supported_backends,("cpu",))

    def test_common_frame_pipeline_records_encode_and_write_stages(self):
        with tempfile.TemporaryDirectory() as directory:
            ctx=RunContext(Path(directory),"test","benchmark",1,{},"cpu",
                           timings_enabled=True)
            ctx.save_frame(Image.new("RGB",(32,24),"navy"),ctx.frame_path(0))
            ctx.finish()
            meta=json.loads((Path(directory)/"meta.json").read_text())
            self.assertEqual(meta["timings"]["jpeg_encode"]["count"],1)
            self.assertEqual(meta["timings"]["frame_write"]["count"],1)


if __name__ == "__main__":
    unittest.main()
