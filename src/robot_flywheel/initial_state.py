"""Fixed-pool adaptation of failure promotion plus initial-state coverage fill.

Inspired by Mulligan (arXiv:2610.05882), not a reproduction: retrieve existing
expert episodes near failed starts rather than collect new supervision there.
"""
from dataclasses import dataclass, asdict, replace
from pathlib import Path
import hashlib
import json
import numpy as np
import torch
from .artifacts import write_json
from .baseline import BaselineConfig, load_policy, train_baseline
from .experiment import Protocol, freeze_protocol
from .rollout import evaluate
from .report import paired_interval

ARM = 'failure_start'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@dataclass
class StartConfig:
    failure_fraction: float = .5


def initial_features(states):
    """Object XY, relative agent XY, periodic orientation; fixed physical scales."""
    s = np.asarray(states, dtype=np.float64)
    if s.shape[-1] != 5 or not np.isfinite(s).all():
        raise ValueError('Expected finite five-dimensional states')
    return np.concatenate((s[..., 2:4]/512., (s[..., :2]-s[..., 2:4])/1024.,
                           np.sin(s[..., 4:5])/2., np.cos(s[..., 4:5])/2.), axis=-1)


def select_starts(episodes, acquired, k, diagnostic, config=None):
    c = config or StartConfig()
    if not 0 <= c.failure_fraction <= 1: raise ValueError('Failure fraction must be in [0,1]')
    by_id = {e.id:e for e in episodes}
    if len(by_id) != len(episodes) or len(set(acquired)) != len(acquired) or not set(acquired) <= by_id.keys():
        raise ValueError('Invalid expert/acquired IDs')
    candidates = set(by_id)-set(acquired)
    if not 0 <= k <= len(candidates): raise ValueError('Acquisition budget exceeds pool')
    features = {i:initial_features(e.states[0]) for i,e in by_id.items()}
    rows = diagnostic['rows']
    if len({r['seed'] for r in rows}) != len(rows): raise ValueError('Duplicate diagnostic starts')
    # No discrete task stages in PushT: lower maximum coverage is a severity proxy.
    failures = sorted((r for r in rows if not r['success']), key=lambda r:(r['max_coverage'],r['seed']))
    chosen=[]; explanations=[]
    for row in failures[:int(np.floor(k*c.failure_fraction))]:
        query=initial_features(row['initial_state'])
        distances={i:float(np.linalg.norm(features[i]-query)) for i in candidates}
        best=min(distances,key=lambda i:(distances[i],i))
        chosen.append(best);candidates.remove(best)
        explanations.append(dict(episode_id=best,reason='failure_start_nearest',
            failure_seed=int(row['seed']),failure_max_coverage=float(row['max_coverage']),
            failure_initial_state=row['initial_state'],match_distance=distances[best]))
    while len(chosen)<k:
        reference=[features[i] for i in list(acquired)+chosen]
        if reference:
            distances={i:float(np.linalg.norm(np.asarray(reference)-features[i],axis=1).min()) for i in candidates}
        else:
            distances={i:0. for i in candidates}
        best=min(distances,key=lambda i:(-distances[i],i))
        chosen.append(best);candidates.remove(best)
        explanations.append(dict(episode_id=best,reason='initial_coverage_fill',nearest_selected_distance=distances[best]))
    return chosen,explanations


def run_failure_start(episodes,source,original,output,compatibility,config=None,rounds_to_run=1,
                      device=None,final_evaluation=False):
    """Run only this arm; default one-round screen, extend unchanged up to p.rounds."""
    c=config or StartConfig();original=Path(original);out=Path(output)
    if not compatibility.get('passed'):raise ValueError('Compatibility must pass')
    if original.resolve()==out.resolve() or original.resolve() in out.resolve().parents:
        raise ValueError('Use a separate output directory')
    if not 0<=c.failure_fraction<=1:raise ValueError('Invalid failure fraction')
    old=read(original/'frozen_protocol.json');p=Protocol(**old['protocol'])
    if len(p.training_seeds)!=1:raise ValueError('Extension expects one training seed')
    if not 1<=rounds_to_run<=p.rounds:raise ValueError('Invalid number of rounds')
    if source['archive_sha256']!=old['source_sha256']:raise ValueError('Expert source changed')
    frozen=freeze_protocol(p,source,episodes,out)
    if frozen['data_sha256']!=old['data_sha256']:raise ValueError('Expert arrays changed')
    seed=p.training_seeds[0];parent=original/f'seed_{seed}'/'baseline/latest.pt'
    saved=torch.load(parent,map_location='cpu',weights_only=False)
    if saved['update']!=p.baseline_updates:raise ValueError('Original baseline update count differs')
    cfg=BaselineConfig(**saved['config']);acquired=list(saved['manifest']['episode_ids'])
    contract=dict(config=asdict(c),baseline_checkpoint=sha(parent),original_protocol=old['sha256'],protocol=frozen['sha256'])
    path=out/'selector_contract.json'
    if path.exists() and read(path)!=contract:raise ValueError('Selector inputs changed; use a new output directory')
    write_json(path,contract)
    records=[]
    for r in range(1,rounds_to_run+1):
        stage=out/f'seed_{seed}'/ARM/f'round_{r}'
        selection_path=stage/'selection.json';parent_hash=sha(parent)
        if selection_path.exists():
            selection=read(selection_path)
            if selection['parent_checkpoint']!=parent_hash:raise ValueError('Selection parent changed')
            added=selection['added_ids']
            if len(added)!=p.acquire or len(set(added))!=len(added) or set(added)&set(acquired):
                raise ValueError('Invalid saved acquisition')
            if selection['cumulative_ids']!=acquired+added:raise ValueError('Saved cumulative IDs differ')
        else:
            print(f'Round {r}: collect {p.diagnostic_count} diagnostic rollouts from parent policy',flush=True)
            model,on,an,checkpoint_hash=load_policy(parent,device)
            start=p.diagnostic_start+(r-1)*p.diagnostic_count
            summary=evaluate(model,on,an,range(start,start+p.diagnostic_count),stage/'diagnostics',checkpoint_hash,
                             execute=p.execute,max_steps=p.max_steps,video_count=0)
            added,explanations=select_starts(episodes,acquired,p.acquire,summary,c)
            selection=dict(parent_checkpoint=parent_hash,added_ids=added,cumulative_ids=acquired+added,
                           explanations=explanations,diagnostic_seeds=summary.get('seeds',[x['seed'] for x in summary['rows']]),
                           failures=sum(not x['success'] for x in summary['rows']))
            write_json(selection_path,selection)
        acquired=selection['cumulative_ids']
        print(f'{ARM} round {r}: train on {len(acquired)} expert episodes',flush=True)
        parent=train_baseline(episodes,source,replace(cfg,initial_episodes=len(acquired),updates=p.baseline_updates+r*p.round_updates),
            stage,device,compatibility,selected_ids=acquired,parent_checkpoint=parent)
        records.append(dict(seed=seed,arm=ARM,round=r,checkpoint=str(parent),episodes=len(acquired),
                            transitions=sum(len(e.states) for e in episodes if e.id in acquired)))
    # Preserve later stages if the user subsequently requests a smaller prefix.
    previous=read(out/'checkpoints.json') if (out/'checkpoints.json').exists() else []
    combined=sorted(records+[x for x in previous if x['round']>rounds_to_run],key=lambda x:x['round'])
    write_json(out/'checkpoints.json',combined)
    if final_evaluation:
        for record in records:
            model,on,an,checkpoint_hash=load_policy(record['checkpoint'],device)
            directory=Path(record['checkpoint']).parent/'final_evaluation'
            summary=evaluate(model,on,an,range(p.evaluation_start,p.evaluation_start+p.evaluation_count),directory,checkpoint_hash,
                             execute=p.execute,max_steps=p.max_steps,video_count=p.video_count)
            record.update(success_rate=summary['success_rate'],mean_max_coverage=summary['mean_max_coverage'],evaluation=str(directory/'summary.json'))
        previous=read(out/'results.json')['records'] if (out/'results.json').exists() else []
        combined=sorted(records+[x for x in previous if x['round']>rounds_to_run],key=lambda x:x['round'])
        write_json(out/'results.json',dict(protocol_hash=frozen['sha256'],mode=p.mode,records=combined))
    return records


def compare_failure_start(original,output,guided=None):
    """Compare the latest evaluated round with the SAME round of existing controls."""
    import csv
    import matplotlib.pyplot as plt
    original=Path(original);out=Path(output)
    contract=read(out/'selector_contract.json');p=read(out/'frozen_protocol.json')['protocol']
    old=read(original/'results.json');new=read(out/'results.json')
    if old['protocol_hash']!=contract['original_protocol'] or new['protocol_hash']!=contract['protocol']:
        raise ValueError('Results/protocol mismatch')
    records=old['records']+new['records']
    if guided is not None:
        guided=Path(guided)
        if read(guided/'classifier_contract.json')['original_protocol']!=contract['original_protocol']:
            raise ValueError('Success-guided run has a different original study')
        records+=read(guided/'results.json')['records']
    expected=list(range(p['evaluation_start'],p['evaluation_start']+p['evaluation_count']))
    values={};starts={}
    for r in records:
        if r['seed']!=p['training_seeds'][0]:raise ValueError('Training seed differs')
        summary=read(r['evaluation'])
        if summary['checkpoint']!=sha(r['checkpoint']):raise ValueError('Evaluation checkpoint differs')
        if summary.get('sampler')!='ddpm' or summary.get('execute')!=p['execute'] or summary.get('max_steps')!=p['max_steps']:
            raise ValueError('Evaluation configuration differs')
        rows=sorted(summary['rows'],key=lambda x:x['seed'])
        if [x['seed'] for x in rows]!=expected:raise ValueError('Unpaired evaluation starts')
        for row in rows:
            if row['seed'] in starts and not np.allclose(starts[row['seed']],row['initial_state'],atol=1e-7,rtol=0):
                raise ValueError('Physical initial states differ')
            starts[row['seed']]=row['initial_state']
        values[r['arm'],r['round']]=np.array([[float(x['success']) for x in rows]])
    final_round=max(r['round'] for r in new['records'])
    comparisons={arm:paired_interval(values[ARM,final_round],values[arm,final_round])
                 for arm in ('random','diversity','continued','targeted','success_guided') if (arm,final_round) in values}
    write_json(out/'comparisons.json',dict(round=final_round,comparisons=comparisons,interpretation='Exploratory, one training seed'))
    write_json(out/'combined_results.json',dict(records=records))
    with (out/'combined_results.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    fig,axs=plt.subplots(1,2,figsize=(12,4));base=next(r for r in records if r['arm']=='baseline')
    for arm in sorted({r['arm'] for r in records}-{'baseline'}):
        series=[base]+sorted((r for r in records if r['arm']==arm),key=lambda x:x['round'])
        for ax,metric in zip(axs,('success_rate','mean_max_coverage')):
            ax.plot([x['round'] for x in series],[x[metric] for x in series],marker='o',label=arm)
            ax.set(xlabel='Round',ylabel=metric,ylim=(0,1));ax.legend()
    fig.suptitle(f'Exploratory; paired comparisons at round {final_round}');fig.tight_layout()
    fig.savefig(out/'combined_curves.png',dpi=150)
    return comparisons,fig
