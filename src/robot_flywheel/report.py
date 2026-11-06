"""Paired, crossed uncertainty estimates and complete experiment tables."""
from pathlib import Path
import json
import csv
import numpy as np
from .artifacts import write_json

def paired_interval(a,b,replicates=2000,seed=0):
    a=np.asarray(a,dtype=float);b=np.asarray(b,dtype=float)
    if a.shape!=b.shape or a.ndim!=2 or not a.size:raise ValueError('Expected paired [training seed, initial state] matrices')
    delta=a-b;rng=np.random.default_rng(seed);means=[]
    for _ in range(replicates):
        rows=rng.integers(delta.shape[0],size=delta.shape[0]);cols=rng.integers(delta.shape[1],size=delta.shape[1])
        means.append(delta[np.ix_(rows,cols)].mean())
    return dict(difference_pp=float(delta.mean()*100),interval95_pp=(np.quantile(means,[.025,.975])*100).tolist(),
                training_seeds=delta.shape[0],initializations=delta.shape[1],method='paired crossed bootstrap',
                limitation='Few training seeds limit uncertainty estimation; quick mode is execution validation only.')

def build_report(output):
    out=Path(output);result=json.loads((out/'results.json').read_text());records=result['records']
    protocol=json.loads((out/'frozen_protocol.json').read_text())['protocol']
    seeds=protocol['training_seeds'];rounds=protocol['rounds'];lookup={(r['seed'],r['arm'],r['round']):r for r in records}
    def matrix(arm,round_index):
        values=[]
        for seed in seeds:
            r=lookup[(seed,arm,round_index)];summary=json.loads(Path(r['evaluation']).read_text())
            rows=sorted(summary['rows'],key=lambda x:x['seed'])
            expected=list(range(protocol['evaluation_start'],protocol['evaluation_start']+protocol['evaluation_count']))
            if [x['seed'] for x in rows]!=expected:raise ValueError('Missing or unpaired evaluation starts')
            values.append([float(x['success']) for x in rows])
        return np.asarray(values)
    # Also verify paired physical initial states, not only RNG seed labels.
    starts={}
    for r in records:
        for row in json.loads(Path(r['evaluation']).read_text())['rows']:
            if row['seed'] in starts and not np.allclose(starts[row['seed']],row['initial_state'],atol=1e-7,rtol=0):raise ValueError('Unpaired physical initial states')
            starts[row['seed']]=row['initial_state']
    comparisons={arm:paired_interval(matrix('targeted',rounds),matrix(arm,rounds)) for arm in ('random','diversity','continued')}
    write_json(out/'comparisons.json',dict(mode=result['mode'],primary='targeted minus random after final round',comparisons=comparisons))
    with (out/'results.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    base=matrix('baseline',0).mean(1)
    for arm in ('continued','random','diversity','targeted'):
        points=np.stack([base]+[matrix(arm,i).mean(1) for i in range(1,rounds+1)],axis=1)
        for row in points:axes[0].plot(range(rounds+1),row,alpha=.15)
        axes[0].plot(range(rounds+1),points.mean(0),marker='o',label=arm)
        counts=[protocol['initial_episodes']]+[lookup[(seeds[0],arm,i)]['episodes'] for i in range(1,rounds+1)]
        axes[1].plot(counts,points.mean(0),marker='o',label=arm)
    for ax in axes:ax.set_ylim(0,1);ax.set_ylabel('Task success rate');ax.legend()
    axes[0].set_xlabel('Acquisition round');axes[1].set_xlabel('Cumulative expert episodes')
    fig.suptitle('EXECUTION CHECK — not performance evidence' if result['mode']=='quick' else 'Held-out policy evaluation')
    fig.tight_layout();fig.savefig(out/'success_curves.png',dpi=160)
    return comparisons,fig
