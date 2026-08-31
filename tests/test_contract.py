import json, tempfile, unittest
from pathlib import Path
from leonardo_demos.base import RunContext
from leonardo_demos.backend import choose_backend

class TestContract(unittest.TestCase):
    def test_numpy_backend(self):
        xp,name=choose_backend('numpy'); self.assertEqual(name,'numpy'); self.assertTrue(hasattr(xp,'zeros'))
    def test_context_meta(self):
        with tempfile.TemporaryDirectory() as t:
            c=RunContext(Path(t),'x','local',3,{},'numpy'); self.assertTrue((Path(t)/'meta.json').exists()); m=json.loads((Path(t)/'meta.json').read_text()); self.assertEqual(m['status'],'starting')
if __name__=='__main__': unittest.main()
