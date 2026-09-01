#!/usr/bin/env python3
"""Merge CPU and Booster benchmark JSON files into benchmarks/latest.json."""
from __future__ import annotations
import argparse, json, shutil
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('inputs',nargs='+',type=Path)
    parser.add_argument('--output',type=Path,default=Path('benchmarks/latest.json'))
    args=parser.parse_args()
    payloads=[json.loads(path.read_text(encoding='utf-8')) for path in args.inputs]
    profiles={p['profile'] for p in payloads}; frames={p['frames'] for p in payloads}
    if len(profiles)!=1 or len(frames)!=1:
        parser.error('all inputs must use the same profile and frame count')
    rows={}
    for payload in payloads:
        for row in payload['results']:
            key=(row['demo'],row['method'],row['backend_requested'])
            # A measured row always replaces another job's unsupported marker.
            if key not in rows or row['status']!='unsupported': rows[key]=row
    merged=dict(payloads[-1])
    merged['results']=list(rows.values())
    merged['environments']=[p['environment'] for p in payloads]
    merged['notes']=list(dict.fromkeys(note for p in payloads for note in p.get('notes',[])))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(merged,indent=2),encoding='utf-8')
    print(args.output)


if __name__=='__main__': main()
