import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from tools.leonardo_preflight import check


class LeonardoPreflightTests(unittest.TestCase):
    def test_cpu_preflight_checks_dependencies_and_output(self):
        with tempfile.TemporaryDirectory() as temp:
            report=check("cpu",Path(temp))
            self.assertTrue(report["run_root_writable"])
            self.assertIn("numpy",report)
            self.assertIn("PIL",report)

    def test_unknown_backend_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(ValueError,"Unsupported Leonardo backend"):
                check("mystery",Path(temp))

    def test_cpu_backend_rejects_a_booster_allocation(self):
        with tempfile.TemporaryDirectory() as temp, patch.dict(
            "os.environ",{"SLURM_JOB_PARTITION":"boost_usr_prod"}
        ):
            with self.assertRaisesRegex(RuntimeError,"waste a Booster"):
                check("cpu",Path(temp))


if __name__ == "__main__":
    unittest.main()
