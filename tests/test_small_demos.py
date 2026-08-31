import tempfile, unittest, json
from pathlib import Path
from leonardo_demos.base import RunContext
from leonardo_demos.demos.reaction_diffusion import ReactionDiffusionDemo
from leonardo_demos.demos.black_hole import BlackHoleDemo
from leonardo_demos.demos.crystal import CrystalDemo
from leonardo_demos.demos.fusion_plasma import FusionPlasmaDemo
from leonardo_demos.demos.weather_ensemble import WeatherEnsembleDemo
from leonardo_demos.demos.molecular_dynamics import MolecularDynamicsDemo

class SmallDemoTests(unittest.TestCase):
    def test_reaction(self):
        with tempfile.TemporaryDirectory() as t:
            c=RunContext(Path(t),'reaction_diffusion','local',2,{'feed':.0367,'kill':.0649},'numpy'); ReactionDiffusionDemo(c,{'n':48,'total_steps':40,'sweep_steps':20,'ensemble':4}).run(); self.assertTrue((Path(t)/'frames/frame_0001.jpg').exists()); self.assertEqual(json.loads((Path(t)/'meta.json').read_text())['status'],'complete')
    def test_blackhole(self):
        with tempfile.TemporaryDirectory() as t:
            c=RunContext(Path(t),'black_hole','local',2,{'mass':1.2,'spin':.2},'numpy'); BlackHoleDemo(c,{'width':120,'height':68,'ensemble':4}).run(); self.assertTrue((Path(t)/'reveal.jpg').exists())
    def test_crystal(self):
        with tempfile.TemporaryDirectory() as t:
            c=RunContext(Path(t),'crystal','local',2,{'undercooling':.75,'anisotropy':.055},'numpy'); CrystalDemo(c,{'depth':3,'ensemble':4,'zoom_levels':1,'zoom_depth':4,'zoom_tile':64}).run(); self.assertTrue((Path(t)/'frames/frame_0001.jpg').exists()); self.assertEqual(json.loads((Path(t)/'meta.json').read_text())['status'],'complete')
    def test_fusion_plasma(self):
        with tempfile.TemporaryDirectory() as t:
            c=RunContext(Path(t),'fusion_plasma','local',2,{'magnetic_field':5.0,'heating':25,'density':1.0},'numpy')
            FusionPlasmaDemo(c,{'n':24,'total_steps':4,'ensemble':4,'sweep_n':20,'sweep_steps':3}).run()
            self.assertTrue((Path(t)/'reveal.jpg').exists())
            self.assertEqual(json.loads((Path(t)/'meta.json').read_text())['status'],'complete')
    def test_weather_ensemble(self):
        with tempfile.TemporaryDirectory() as t:
            c=RunContext(Path(t),'weather_ensemble','local',2,{'warming':1.5,'jet_stream':1.0,'uncertainty':25},'numpy')
            WeatherEnsembleDemo(c,{'n':24,'total_steps':4,'ensemble':4,'sweep_n':24,'sweep_steps':3}).run()
            self.assertTrue((Path(t)/'frames/frame_0001.jpg').exists())
            self.assertEqual(json.loads((Path(t)/'meta.json').read_text())['status'],'complete')
    def test_molecular_dynamics(self):
        with tempfile.TemporaryDirectory() as t:
            c=RunContext(Path(t),'molecular_dynamics','local',2,{'temperature':310,'attraction':1.0,'solvent':.65,'sequence':0},'numpy')
            MolecularDynamicsDemo(c,{'particles':18,'total_steps':4,'ensemble':4,'sweep_particles':14,'sweep_steps':3}).run()
            self.assertTrue((Path(t)/'reveal.jpg').exists())
            self.assertEqual(json.loads((Path(t)/'meta.json').read_text())['status'],'complete')
if __name__=='__main__': unittest.main()
