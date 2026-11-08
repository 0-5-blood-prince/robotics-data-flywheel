"""Three-round matched acquisition study with immutable protocol and resume."""
from dataclasses import dataclass,asdict,field,replace
from pathlib import Path
import json
import hashlib
import importlib.metadata
import subprocess
import sys
import numpy as np
import torch
from .baseline import BaselineConfig,train_baseline,load_policy
from .selection import select,geometry,representatives
from .artifacts import write_json,digest,source_fingerprint
from .rollout import evaluate

ARMS=("continued","random","diversity","targeted")

@dataclass
class Protocol:
    training_seeds:list=field(default_factory=lambda:[11,22,33,44,55])
    rounds:int=3
    acquire:int=20
    initial_episodes:int=50
    baseline_updates:int=20000
    round_updates:int=10000
    batch_size:int=128
    checkpoint_every:int=1000
    width:int=64
    diffusion_steps:int=100
    learning_rate:float=.0001
    ema_decay:float=.995
    history:int=2
    horizon:int=16
    execute:int=8
    max_steps:int=300
    diagnostic_count:int=20
    evaluation_count:int=200
    diagnostic_start:int=10000
    evaluation_start:int=100000
    video_count:int=1
    failure_samples_per_episode:int=8
    stalled_window:int=10
    minimum_progress:float=.02
    unsupported_distance:float=.5
    mode:str="full"
    development_success_rate:float=0.
    minimum_development_success:float=.2

    @classmethod
    def quick(cls):
        return cls(training_seeds=[11],initial_episodes=4,acquire=1,baseline_updates=4,
                   round_updates=2,batch_size=2,checkpoint_every=2,width=16,diffusion_steps=4,
                   diagnostic_count=2,evaluation_count=2,max_steps=8,video_count=0,mode="quick")

    def validate(self,total):
        if self.mode=="full" and self.development_success_rate<self.minimum_development_success:
            raise ValueError("Learning gate failed: improve the development baseline before launching the full study")
        if self.mode not in ("quick","full"): raise ValueError("Unknown mode")
        positive=(self.rounds,self.acquire,self.initial_episodes,self.baseline_updates,self.round_updates,
                  self.batch_size,self.checkpoint_every,self.diagnostic_count,self.evaluation_count,
                  self.max_steps,self.failure_samples_per_episode,self.stalled_window)
        if any(x<1 for x in positive): raise ValueError("Budgets must be positive")
        if not self.training_seeds or len(set(self.training_seeds))!=len(self.training_seeds):raise ValueError("Training seeds must be unique")
        if self.initial_episodes+self.rounds*self.acquire>total: raise ValueError("Not enough expert episodes")
        if not 1<=self.execute<=self.horizon:raise ValueError("Invalid action execution horizon")
        diagnostic=set(range(self.diagnostic_start,self.diagnostic_start+self.rounds*self.diagnostic_count))
        evaluation=set(range(self.evaluation_start,self.evaluation_start+self.evaluation_count))
        if diagnostic & evaluation or diagnostic & set(range(1000,2000)) or evaluation & set(range(1000,2000)):
            raise ValueError("Development/diagnostic/evaluation seeds overlap")


def freeze_protocol(protocol,source,episodes,output):
    protocol.validate(len(episodes));out=Path(output);out.mkdir(parents=True,exist_ok=True)
    # Hash actual tensors, not only source URL, to prevent accidental data substitution.
    h=hashlib.sha256()
    for ep in episodes:
        h.update(str(ep.id).encode());h.update(ep.states.tobytes());h.update(ep.actions.tobytes())
    try: commit=subprocess.check_output(['git','-C',str(Path(__file__).resolve().parents[2]),'rev-parse','HEAD'],text=True,stderr=subprocess.DEVNULL).strip()
    except (OSError,subprocess.CalledProcessError):commit="unknown"
    versions={}
    for name in ('numpy','torch','gym-pusht','pymunk','zarr'):
        try:versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:versions[name]="unavailable"
    content=dict(protocol=asdict(protocol),source_sha256=source['archive_sha256'],data_sha256=h.hexdigest(),code_commit=commit,versions=versions,source_fingerprint=source_fingerprint())
    content['sha256']=digest(content)
    path=out/'frozen_protocol.json'
    if path.exists() and json.loads(path.read_text())!=content:
        raise ValueError("Frozen protocol, code, dependencies or data changed: use a new experiment directory")
    write_json(path,content);return content


def failures_from_rollouts(summary,directory,protocol):
    features=[];references=[]
    for row in summary['rows']:
        if row['success']:continue
        path=Path(directory)/f"episode_{row['seed']}.npz"
        with np.load(path) as data:
            states=data['next_states'] if 'next_states' in data else np.vstack((data['states'][1:],data['final_state']));coverage=data['coverage'];n=len(states)
            if not n:continue
            w=protocol.stalled_window
            candidates=[i for i in range(w,n) if coverage[i]-coverage[i-w]<protocol.minimum_progress]
            if not candidates:candidates=list(range(n))
            indices=np.asarray(candidates)[np.linspace(0,len(candidates)-1,min(len(candidates),protocol.failure_samples_per_episode),dtype=int)]
            for i in indices:
                features.append(geometry(states[int(i)]));references.append(dict(seed=row['seed'],step=int(i)))
    return np.asarray(features,dtype=np.float32).reshape(-1,7),references


def run_study(episodes,source,protocol,output,compatibility,device=None,final_evaluation=True):
    """Resumes completed training stages and cached rollouts; never chooses from final data."""
    if not compatibility.get('passed'):raise ValueError('Compatibility must pass')
    out=Path(output);frozen=freeze_protocol(protocol,source,episodes,out)
    records=[];p=protocol
    for seed in p.training_seeds:
        cfg=BaselineConfig(seed=seed,initial_episodes=p.initial_episodes,updates=p.baseline_updates,
                           batch_size=p.batch_size,checkpoint_every=p.checkpoint_every,width=p.width,
                           diffusion_steps=p.diffusion_steps,history=p.history,horizon=p.horizon,
                           learning_rate=p.learning_rate,ema_decay=p.ema_decay)
        base_dir=out/f'seed_{seed}'/'baseline'
        base=train_baseline(episodes,source,cfg,base_dir,device,compatibility)
        saved=torch.load(base,map_location='cpu',weights_only=False)
        initial=saved['manifest']['episode_ids']
        records.append(dict(seed=seed,arm='baseline',round=0,checkpoint=str(base),episodes=len(initial),transitions=saved['manifest']['transitions']))
        for arm in ARMS:
            parent=base;acquired=list(initial)
            for round_index in range(1,p.rounds+1):
                stage=out/f'seed_{seed}'/arm/f'round_{round_index}'
                selection_path=stage/'selection.json'
                parent_hash=hashlib.sha256(Path(parent).read_bytes()).hexdigest()
                if selection_path.exists():
                    selection=json.loads(selection_path.read_text())
                    if selection['parent_checkpoint']!=parent_hash:raise ValueError('Selection parent changed')
                    acquired=selection['cumulative_ids']
                else:
                    model,on,an,checkpoint_hash=load_policy(parent,device)
                    start=p.diagnostic_start+(round_index-1)*p.diagnostic_count
                    diag_dir=stage/'diagnostics'
                    diagnostic=evaluate(model,on,an,list(range(start,start+p.diagnostic_count)),diag_dir,checkpoint_hash,
                                        execute=p.execute,max_steps=p.max_steps,video_count=0)
                    features,refs=failures_from_rollouts(diagnostic,diag_dir,p)
                    added=[];explanations=[];distances=[]
                    if arm!='continued':
                        added,explanations=select(episodes,acquired,p.acquire,arm,seed=seed+round_index*1000,failure_features=features)
                        if len(features):
                            pool=np.concatenate([representatives(e) for e in episodes if e.id not in acquired])
                            distances=np.sqrt(((features[:,None,:]-pool[None,:,:])**2).sum(-1).min(1)).tolist()
                    selection=dict(parent_checkpoint=parent_hash,cumulative_ids=acquired+added,added_ids=added,
                                   explanations=explanations,failure_references=refs,nearest_pool_distances=distances,
                                   unsupported_fraction=float(np.mean(np.array(distances)>p.unsupported_distance)) if distances else None,
                                   fallback=bool(arm=='targeted' and not len(features)))
                    write_json(selection_path,selection);acquired=selection['cumulative_ids']
                stage_cfg=replace(cfg,initial_episodes=len(acquired),updates=p.baseline_updates+round_index*p.round_updates)
                parent=train_baseline(episodes,source,stage_cfg,stage,device,compatibility,selected_ids=acquired,parent_checkpoint=parent)
                saved=torch.load(parent,map_location='cpu',weights_only=False)
                records.append(dict(seed=seed,arm=arm,round=round_index,checkpoint=str(parent),episodes=len(acquired),transitions=saved['manifest']['transitions']))
                write_json(out/'checkpoints.json',records)
    write_json(out/'checkpoints.json',records)
    if final_evaluation:
        for record in records:
            model,on,an,sha=load_policy(record['checkpoint'],device)
            directory=Path(record['checkpoint']).parent/'final_evaluation'
            summary=evaluate(model,on,an,list(range(p.evaluation_start,p.evaluation_start+p.evaluation_count)),directory,sha,
                             execute=p.execute,max_steps=p.max_steps,video_count=p.video_count)
            record.update(success_rate=summary['success_rate'],mean_max_coverage=summary['mean_max_coverage'],evaluation=str(directory/'summary.json'))
        write_json(out/'results.json',dict(protocol_hash=frozen['sha256'],mode=p.mode,records=records))
    return records
