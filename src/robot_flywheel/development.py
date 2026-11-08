"""Development-only checkpoint and execution-horizon comparison."""
import csv
import json
from pathlib import Path
import numpy as np
import torch
from .baseline import load_policy
from .rollout import evaluate
from .artifacts import write_json


def trace_metrics(directory, summary):
    metrics=[]
    for row in summary['rows']:
        with np.load(Path(directory)/f"episode_{row['seed']}.npz") as d:
            cov=d['coverage'];actions=d['actions'];states=d['states']
            metrics.append(dict(seed=row['seed'],success=row['success'],max_coverage=row['max_coverage'],
                final_coverage=float(cov[-1]),
                near_success_failure=bool(not row['success'] and row['max_coverage']>=.90),
                action_boundary_fraction=float(np.mean((actions<=.001)|(actions>=511.999))),
                mean_action_change=float(np.linalg.norm(np.diff(actions,axis=0),axis=1).mean()) if len(actions)>1 else 0.,
                object_path_length=float(np.linalg.norm(np.diff(states[:,2:4],axis=0),axis=1).sum())))
    return metrics


def compare_checkpoints(checkpoints, output, seeds=range(1000,1020), prefixes=(2,4,8), max_steps=300, device=None, video_count=1,inference_steps=None):
    seeds=list(seeds)
    if not seeds or any(s<1000 or s>=2000 for s in seeds):raise ValueError('Use development seeds 1000..1999 only')
    out=Path(output);out.mkdir(parents=True,exist_ok=True);records=[]
    for label,path in checkpoints.items():
        model,on,an,sha=load_policy(path,device)
        saved=torch.load(path,map_location='cpu',weights_only=False)
        for prefix in prefixes:
            sampler='ddpm' if inference_steps is None else f'ddim{inference_steps}'
            directory=out/f'{sha[:12]}_execute{prefix}_{sampler}_horizon{max_steps}'
            summary=evaluate(model,on,an,seeds,directory,sha,execute=prefix,max_steps=max_steps,video_count=video_count,inference_steps=inference_steps)
            traces=trace_metrics(directory,summary);write_json(directory/'trace_metrics.json',traces)
            records.append(dict(label=label,checkpoint=str(path),checkpoint_hash=sha,updates=saved['update'],
                sampler=sampler,expert_episodes=len(saved['manifest']['episode_ids']),execute=prefix,
                success_rate=summary['success_rate'],mean_max_coverage=summary['mean_max_coverage'],
                mean_final_coverage=float(np.mean([x['final_coverage'] for x in traces])),
                near_success_failures=sum(x['near_success_failure'] for x in traces),
                action_boundary_fraction=float(np.mean([x['action_boundary_fraction'] for x in traces])),
                evaluation_dir=str(directory)))
            write_json(out/'development_results.json',records)
    with (out/'development_results.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    return records


def plot_traces(records,seed=1000):
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(16,4))
    for record in records:
        with np.load(Path(record['evaluation_dir'])/f'episode_{seed}.npz') as d:
            label=f"{record['label']}, execute {record['execute']}"
            axes[0].plot(np.arange(len(d['coverage']))/10,d['coverage'],label=label)
            axes[1].plot(d['states'][:,2],d['states'][:,3],label=label)
            axes[2].plot(np.arange(1,len(d['actions']))/10,np.linalg.norm(np.diff(d['actions'],axis=0),axis=1),label=label)
    axes[0].axhline(.95,color='black',linestyle='--');axes[0].set(xlabel='Seconds',ylabel='Raw coverage',ylim=(0,1))
    axes[1].set(xlabel='Object x',ylabel='Object y',aspect='equal')
    axes[2].set(xlabel='Seconds',ylabel='Action target change')
    axes[0].legend(fontsize=7);fig.tight_layout();return fig


def side_by_side(records,output,seed=1000):
    """Same simulated time in each panel; hold final frame after termination."""
    import imageio.v2 as imageio
    from PIL import Image,ImageDraw
    paths=[Path(r['evaluation_dir'])/f'episode_{seed}.mp4' for r in records]
    if not paths or not all(p.exists() for p in paths):raise ValueError('Choose a recorded video seed')
    frames=[]
    for path in paths:
        reader=imageio.get_reader(path)
        try:frames.append([x for x in reader])
        finally:reader.close()
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    writer=imageio.get_writer(output,fps=10)
    try:
        for t in range(max(map(len,frames))):
            canvas=Image.new('RGB',(384*len(records),416),'white');draw=ImageDraw.Draw(canvas)
            for i,(record,clip) in enumerate(zip(records,frames)):
                panel=Image.fromarray(clip[min(t,len(clip)-1)]).resize((384,384))
                canvas.paste(panel,(384*i,32))
                draw.text((384*i+8,8),f"{record['label']} | execute {record['execute']}",fill='black')
            writer.append_data(np.array(canvas))
    finally:writer.close()
    return output
