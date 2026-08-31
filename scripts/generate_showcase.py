from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from run_demo import run
for demo in ['reaction_diffusion','neural_wall','black_hole','pbh','crystal','galaxy_collision','cosmic_web','fluid','fusion_plasma','weather_ensemble','molecular_dynamics']:
    print('\n===',demo,'===')
    run(demo,'local',50)
