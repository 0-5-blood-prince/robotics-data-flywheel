"""Development-only data-scaling runs. Never inspect the final test seeds here."""
from dataclasses import replace
from pathlib import Path
from .baseline import BaselineConfig,train_baseline,load_policy
from .rollout import evaluate
from .artifacts import write_json

def run_pilot(episodes,source,compatibility,output,sizes=(25,50),updates=5000,device=None):
    out=Path(output);rows=[]
    for size in dict.fromkeys([*sizes,len(episodes)]):
        cfg=BaselineConfig(initial_episodes=size,updates=updates)
        directory=out/f'episodes_{size}'
        path=train_baseline(episodes,source,cfg,directory,device,compatibility)
        model,on,an,sha=load_policy(path,device)
        summary=evaluate(model,on,an,list(range(1000,1010)),directory/f'development_{sha[:12]}',sha,video_count=1)
        rows.append(dict(episodes=size,updates=updates,success_rate=summary['success_rate'],mean_max_coverage=summary['mean_max_coverage'],checkpoint=str(path)))
        write_json(out/'pilot.json',rows)
    return rows
