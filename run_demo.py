from __future__ import annotations
import argparse, json, time, webbrowser
from pathlib import Path
from leonardo_demos.base import RunContext
from leonardo_demos.registry import DEMOS

ROOT=Path(__file__).resolve().parent

def load_profiles(): return json.loads((ROOT/'config/profiles.json').read_text())
def load_specs(): return json.loads((ROOT/'config/demo_specs.json').read_text())

def run(demo,profile='local',frames=80,params=None,backend='auto',run_dir=None):
    profiles=load_profiles(); specs=load_specs()
    if demo not in DEMOS: raise SystemExit(f"Unknown demo {demo}. Choices: {', '.join(DEMOS)}")
    if profile not in profiles: raise SystemExit(f"Unknown profile {profile}")
    frames=int(frames)
    if frames < 1:
        raise ValueError('frames must be at least 1')
    defaults={k:v.get('value') for k,v in specs[demo]['params'].items()}
    defaults.update(params or {})
    stamp=time.strftime('%Y%m%d_%H%M%S'); run_dir=Path(run_dir or ROOT/'runs'/f'{demo}_{stamp}')
    ctx=RunContext(run_dir,demo,profile,frames,defaults,backend)
    try:
        DEMOS[demo](ctx,profiles[profile][demo]).run()
    except Exception as e:
        ctx.fail(e); raise
    print(run_dir)
    return run_dir

if __name__=='__main__':
    ap=argparse.ArgumentParser(description='Generate a visual HPC demo run')
    ap.add_argument('demo',choices=sorted(DEMOS)); ap.add_argument('--profile',choices=['local','desktop','leonardo'],default='local'); ap.add_argument('--frames',type=int,default=80); ap.add_argument('--backend',default='auto'); ap.add_argument('--param',action='append',default=[],help='key=value, repeatable'); ap.add_argument('--run-dir'); ap.add_argument('--open',action='store_true')
    a=ap.parse_args(); params={}
    for kv in a.param:
        k,v=kv.split('=',1)
        try: v=float(v)
        except ValueError: pass
        params[k]=v
    rd=run(a.demo,a.profile,a.frames,params,a.backend,a.run_dir)
    if a.open:
        webbrowser.open(f'file://{(rd/"frames"/"frame_0000.jpg").resolve()}')
