"""Fresh-seed baseline and random control for a matched failure-start comparison."""
from dataclasses import replace
from pathlib import Path
import torch
from .artifacts import write_json
from .baseline import BaselineConfig,train_baseline,load_policy
from .experiment import freeze_protocol
from .selection import select
from .rollout import evaluate
from .initial_state import read,sha


def run_random_reference(episodes,source,protocol,output,compatibility,device=None,final_evaluation=False):
    """Train only a new baseline and sequential random acquisitions; resumable."""
    if not compatibility.get('passed'):raise ValueError('Compatibility must pass')
    p=protocol;out=Path(output)
    if len(p.training_seeds)!=1:raise ValueError('Expected one replication training seed')
    frozen=freeze_protocol(p,source,episodes,out);seed=p.training_seeds[0]
    cfg=BaselineConfig(seed=seed,initial_episodes=p.initial_episodes,updates=p.baseline_updates,
        batch_size=p.batch_size,checkpoint_every=p.checkpoint_every,width=p.width,
        diffusion_steps=p.diffusion_steps,history=p.history,horizon=p.horizon,
        learning_rate=p.learning_rate,ema_decay=p.ema_decay)
    parent=train_baseline(episodes,source,cfg,out/f'seed_{seed}'/'baseline',device,compatibility)
    saved=torch.load(parent,map_location='cpu',weights_only=False)
    acquired=list(saved['manifest']['episode_ids'])
    records=[dict(seed=seed,arm='baseline',round=0,checkpoint=str(parent),episodes=len(acquired),transitions=saved['manifest']['transitions'])]
    for r in range(1,p.rounds+1):
        stage=out/f'seed_{seed}'/'random'/f'round_{r}'
        ids,explanations=select(episodes,acquired,p.acquire,'random',seed=seed+r*1000)
        selection=dict(parent_checkpoint=sha(parent),added_ids=ids,cumulative_ids=acquired+ids,explanations=explanations)
        path=stage/'selection.json'
        if path.exists() and read(path)!=selection:raise ValueError('Random acquisition changed')
        write_json(path,selection);acquired=selection['cumulative_ids']
        parent=train_baseline(episodes,source,replace(cfg,initial_episodes=len(acquired),updates=p.baseline_updates+r*p.round_updates),
            stage,device,compatibility,selected_ids=acquired,parent_checkpoint=parent)
        saved=torch.load(parent,map_location='cpu',weights_only=False)
        records.append(dict(seed=seed,arm='random',round=r,checkpoint=str(parent),episodes=len(acquired),transitions=saved['manifest']['transitions']))
        write_json(out/'checkpoints.json',records)
    if final_evaluation:
        for record in records:
            model,on,an,checkpoint_hash=load_policy(record['checkpoint'],device)
            directory=Path(record['checkpoint']).parent/'final_evaluation'
            summary=evaluate(model,on,an,range(p.evaluation_start,p.evaluation_start+p.evaluation_count),directory,checkpoint_hash,
                execute=p.execute,max_steps=p.max_steps,video_count=p.video_count)
            record.update(success_rate=summary['success_rate'],mean_max_coverage=summary['mean_max_coverage'],evaluation=str(directory/'summary.json'))
        write_json(out/'results.json',dict(protocol_hash=frozen['sha256'],mode=p.mode,records=records))
    return records
