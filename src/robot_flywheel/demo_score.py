"""Demo-SCORE-inspired acquisition; existing studies are read-only inputs.

Unlike the paper's filtering protocol, rank an unacquired demonstration pool
and retain all previously acquired episodes. Freeze the classifier across rounds.
"""
from dataclasses import dataclass, asdict, replace
from pathlib import Path
import hashlib
import json
import numpy as np
import torch
from torch import nn
from .data import encode
from .baseline import BaselineConfig, atomic_save, load_policy, train_baseline
from .artifacts import digest, write_json
from .experiment import Protocol, freeze_protocol
from .rollout import evaluate
from .report import paired_interval


@dataclass
class ScoreConfig:
    rollouts_per_checkpoint: int = 100
    rollout_start: int = 300000
    classifier_steps: int = 1000
    check_every: int = 25
    learning_rate: float = .001
    weight_decay: float = .01
    dropout: float = .2
    min_per_class: int = 5
    seed: int = 11


def classifier(dropout):
    return nn.Sequential(nn.Linear(6, 8), nn.ReLU(), nn.Dropout(dropout),
                         nn.Linear(8, 8), nn.ReLU(), nn.Dropout(dropout), nn.Linear(8, 1))


def rollout_data(directory):
    directory = Path(directory)
    summary = json.loads((directory / 'summary.json').read_text())
    trajectories, labels = [], []
    for row in summary['rows']:
        with np.load(directory / f"episode_{row['seed']}.npz") as trace:
            # Pre-action states only; never supply success/coverage as input.
            x = encode(trace['states']).astype(np.float32)
        if not len(x) or not np.isfinite(x).all():
            raise ValueError('Empty or nonfinite classifier trajectory')
        trajectories.append(x)
        labels.append(float(row['success']))
    return trajectories, np.asarray(labels, dtype=np.float32)


def fit_classifier(train, validation, config):
    """Equal trajectory weighting; whole checkpoints held apart, CPU deterministic."""
    c = config
    if c.classifier_steps < 1 or c.check_every < 1 or c.min_per_class < 1:
        raise ValueError('Classifier budgets must be positive')
    for name, (xs, ys) in [('training', train), ('validation', validation)]:
        if len(xs) != len(ys) or min(int((ys == 0).sum()), int((ys == 1).sum())) < c.min_per_class:
            raise ValueError(f'{name}: need at least {c.min_per_class} successful and failed rollouts; increase rollout budget in a new output directory')
    torch.manual_seed(c.seed)
    all_x = np.concatenate(train[0])
    mean = all_x.mean(0); std = np.maximum(all_x.std(0), 1e-4)
    def tensors(data):
        xs, ys = data
        x = torch.tensor((np.concatenate(xs) - mean) / std)
        y = torch.tensor(np.concatenate([np.full(len(s), y, dtype=np.float32) for s, y in zip(xs, ys)]))
        weights = torch.tensor(np.concatenate([np.full(len(s), 1 / (len(xs) * len(s)), dtype=np.float32) for s in xs]))
        return x, y, weights
    tx, ty, tw = tensors(train); vx, vy, vw = tensors(validation)
    model = classifier(c.dropout)
    optimizer = torch.optim.AdamW(model.parameters(), lr=c.learning_rate, weight_decay=c.weight_decay)
    best = float('inf'); best_state = None; history = []
    for step in range(1, c.classifier_steps + 1):
        model.train(); optimizer.zero_grad()
        loss = (nn.functional.binary_cross_entropy_with_logits(model(tx).squeeze(-1), ty, reduction='none') * tw).sum()
        loss.backward(); optimizer.step()
        if step % c.check_every == 0 or step == c.classifier_steps:
            model.eval()
            with torch.no_grad():
                val = float((nn.functional.binary_cross_entropy_with_logits(model(vx).squeeze(-1), vy, reduction='none') * vw).sum())
            history.append(dict(step=step, train_loss=float(loss.detach()), validation_loss=val))
            if val < best:
                best = val; best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}; best_step = step
    prior = float(train[1].mean()); target_rate = float(validation[1].mean())
    constant_loss = -target_rate * np.log(prior) - (1-target_rate) * np.log(1-prior)
    saved = dict(state_dict=best_state, mean=mean.tolist(), std=std.tolist(), dropout=c.dropout)
    report = dict(best_step=best_step, validation_loss=best, constant_validation_loss=float(constant_loss),
                  beats_constant=bool(best < constant_loss), training_successes=int(train[1].sum()),
                  validation_successes=int(validation[1].sum()), training_episodes=len(train[1]),
                  validation_episodes=len(validation[1]), history=history)
    return saved, report


def rank_episodes(episodes, saved):
    model = classifier(saved['dropout']); model.load_state_dict(saved['state_dict']); model.eval()
    scores = []
    with torch.no_grad():
        for ep in episodes:
            x = (encode(ep.states) - np.asarray(saved['mean'])) / np.asarray(saved['std'])
            score = float(model(torch.tensor(x, dtype=torch.float32)).sigmoid().mean())
            scores.append(dict(episode_id=ep.id, score=score, reason='success_guided'))
    return sorted(scores, key=lambda r: (-r['score'], r['episode_id']))


def prepare_success_guided(episodes, source, original, output, compatibility, config=None, device=None):
    """Fit once using intermediate/final baseline checkpoints; never retrain old arms."""
    c = config or ScoreConfig(); original = Path(original); out = Path(output)
    old = json.loads((original / 'frozen_protocol.json').read_text())
    p = Protocol(**old['protocol'])
    if len(p.training_seeds) != 1: raise ValueError('This extension expects one training seed')
    if not compatibility.get('passed'): raise ValueError('Compatibility must pass')
    if out.resolve() == original.resolve(): raise ValueError('Use a separate extension directory')
    if source['archive_sha256'] != old['source_sha256']: raise ValueError('Source archive differs')
    seed = p.training_seeds[0]
    base = original / f'seed_{seed}' / 'baseline' / 'latest.pt'
    saved = torch.load(base, map_location='cpu', weights_only=False)
    if saved['update'] != p.baseline_updates: raise ValueError('Baseline update count differs')
    early_step = max(p.checkpoint_every, (p.baseline_updates // 2 // p.checkpoint_every) * p.checkpoint_every)
    early = base.parent / f'step_{early_step:07d}.pt'
    if early == base or early_step >= p.baseline_updates: raise ValueError('Need two distinct baseline checkpoints')
    paths = [early, base]
    hashes = [hashlib.sha256(path.read_bytes()).hexdigest() for path in paths]
    if hashes[0] == hashes[1]: raise ValueError('Classifier checkpoints must differ')
    starts = set(range(c.rollout_start, c.rollout_start + 2*c.rollouts_per_checkpoint))
    reserved = set(range(p.evaluation_start, p.evaluation_start+p.evaluation_count)) | set(range(p.diagnostic_start, p.diagnostic_start+p.rounds*p.diagnostic_count)) | set(range(1000,2000))
    if c.rollouts_per_checkpoint < 1 or starts & reserved: raise ValueError('Invalid or overlapping classifier rollout starts')
    frozen = freeze_protocol(p, source, episodes, out)
    if frozen['data_sha256'] != old['data_sha256']: raise ValueError('Expert data differs from original study')
    contract = dict(config=asdict(c), original_protocol=old['sha256'], checkpoints=hashes, protocol=frozen['sha256'])
    contract_path = out / 'classifier_contract.json'
    if contract_path.exists() and json.loads(contract_path.read_text()) != contract:
        raise ValueError('Classifier inputs changed; use a new extension directory')
    write_json(contract_path, contract)
    artifact = out / 'classifier.pt'
    if not artifact.exists():
        datasets = []
        for index, path in enumerate(paths):
            print(f'Classifier rollout checkpoint {index+1}/2: {path.name}', flush=True)
            model, on, an, sha = load_policy(path, device)
            directory = out / f'classifier_rollouts_{index}'
            start = c.rollout_start + index*c.rollouts_per_checkpoint
            evaluate(model, on, an, range(start, start+c.rollouts_per_checkpoint), directory, sha,
                     execute=p.execute, max_steps=p.max_steps, video_count=0)
            datasets.append(rollout_data(directory))
        fitted, report = fit_classifier(*datasets, c)
        fitted.update(report=report, contract_hash=digest(contract))
        atomic_save(artifact, fitted)
    fitted = torch.load(artifact, map_location='cpu', weights_only=False)
    if fitted['contract_hash'] != digest(contract): raise ValueError('Classifier contract differs')
    write_json(out / 'classifier_report.json', fitted['report'])
    write_json(out / 'ranking.json', rank_episodes(episodes, fitted))
    return fitted['report']


def run_success_guided(episodes, source, original, output, compatibility, device=None, final_evaluation=False):
    """Only the new arm; rerun unchanged to resume. Classifier must already pass."""
    original = Path(original); out = Path(output)
    frozen = json.loads((out / 'frozen_protocol.json').read_text()); p = Protocol(**frozen['protocol'])
    freeze_protocol(p, source, episodes, out)
    fitted = torch.load(out / 'classifier.pt', map_location='cpu', weights_only=False)
    contract = json.loads((out / 'classifier_contract.json').read_text())
    if digest(contract) != fitted['contract_hash']: raise ValueError('Classifier changed')
    if not fitted['report']['beats_constant']: raise ValueError('Classifier does not beat constant validation predictor; inspect report before training')
    seed = p.training_seeds[0]; parent = original / f'seed_{seed}' / 'baseline' / 'latest.pt'
    if hashlib.sha256(parent.read_bytes()).hexdigest() != contract['checkpoints'][1]: raise ValueError('Original baseline changed')
    saved = torch.load(parent, map_location='cpu', weights_only=False)
    cfg = BaselineConfig(**saved['config']); acquired = list(saved['manifest']['episode_ids'])
    ranking = rank_episodes(episodes, fitted); records = []
    for r in range(1, p.rounds+1):
        stage = out / f'seed_{seed}' / 'success_guided' / f'round_{r}'
        added = [x for x in ranking if x['episode_id'] not in acquired][:p.acquire]
        if len(added) != p.acquire: raise ValueError('Insufficient unacquired episodes')
        acquired += [x['episode_id'] for x in added]
        selection = dict(cumulative_ids=acquired.copy(), added_ids=[x['episode_id'] for x in added], explanations=added,
                         classifier_hash=hashlib.sha256((out/'classifier.pt').read_bytes()).hexdigest(),
                         parent_checkpoint=hashlib.sha256(parent.read_bytes()).hexdigest())
        path = stage/'selection.json'
        if path.exists() and json.loads(path.read_text()) != selection: raise ValueError('Selection changed')
        write_json(path, selection)
        print(f'Success-guided round {r}/{p.rounds}: {len(acquired)} episodes', flush=True)
        parent = train_baseline(episodes, source, replace(cfg, initial_episodes=len(acquired), updates=p.baseline_updates+r*p.round_updates),
                                stage, device, compatibility, selected_ids=acquired, parent_checkpoint=parent)
        record = dict(seed=seed, arm='success_guided', round=r, checkpoint=str(parent), episodes=len(acquired),
                      transitions=sum(len(e.states) for e in episodes if e.id in acquired))
        records.append(record)
        write_json(out/'checkpoints.json', records)
    if final_evaluation:
        for record in records:
            model, on, an, sha = load_policy(record['checkpoint'], device)
            directory = Path(record['checkpoint']).parent / 'final_evaluation'
            summary = evaluate(model, on, an, range(p.evaluation_start,p.evaluation_start+p.evaluation_count), directory, sha,
                               execute=p.execute, max_steps=p.max_steps, video_count=p.video_count)
            record.update(success_rate=summary['success_rate'], mean_max_coverage=summary['mean_max_coverage'], evaluation=str(directory/'summary.json'))
        write_json(out/'results.json', dict(protocol_hash=frozen['sha256'], mode=p.mode, records=records))
    return records


def compare_success_guided(original, output):
    """Append into a NEW combined table, reading existing controls without running them."""
    import csv
    import matplotlib.pyplot as plt
    original = Path(original); out = Path(output)
    original_result = json.loads((original/'results.json').read_text())
    contract = json.loads((out/'classifier_contract.json').read_text())
    if original_result['protocol_hash'] != contract['original_protocol']:
        raise ValueError('Original results belong to a different study')
    old = original_result['records']
    new = json.loads((out/'results.json').read_text())['records']
    records = old + new
    p = json.loads((out/'frozen_protocol.json').read_text())['protocol']
    if {r['seed'] for r in records} != set(p['training_seeds']):
        raise ValueError('Training seeds differ')
    expected = list(range(p['evaluation_start'], p['evaluation_start']+p['evaluation_count']))
    starts = {}; values = {}
    for record in records:
        summary = json.loads(Path(record['evaluation']).read_text())
        if summary['execute'] != p['execute'] or summary['max_steps'] != p['max_steps'] or summary['sampler'] != 'ddpm':
            raise ValueError('Evaluation protocol differs')
        actual_hash = hashlib.sha256(Path(record['checkpoint']).read_bytes()).hexdigest()
        if summary['checkpoint'] != actual_hash: raise ValueError('Evaluation checkpoint changed')
        rows = sorted(summary['rows'], key=lambda x:x['seed'])
        if [x['seed'] for x in rows] != expected: raise ValueError('Unpaired evaluation starts')
        for row in rows:
            if row['seed'] in starts and not np.allclose(starts[row['seed']],row['initial_state'],atol=1e-7,rtol=0): raise ValueError('Unpaired physical states')
            starts[row['seed']] = row['initial_state']
        values[(record['arm'],record['round'])] = np.array([[float(x['success']) for x in rows]])
    comparisons = {arm:paired_interval(values['success_guided',p['rounds']],values[arm,p['rounds']])
                   for arm in ('random','diversity','continued','targeted')}
    write_json(out/'combined_results.json', dict(records=records, interpretation='Exploratory redesign; one training seed'))
    write_json(out/'comparisons.json', comparisons)
    with (out/'combined_results.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    fig, axes=plt.subplots(1,2,figsize=(12,4))
    base=next(r for r in old if r['arm']=='baseline')
    for arm in ('continued','random','diversity','targeted','success_guided'):
        series=[base]+sorted([r for r in records if r['arm']==arm],key=lambda r:r['round'])
        for ax, metric in zip(axes, ('success_rate','mean_max_coverage')):
            ax.plot([r['round'] for r in series],[r[metric] for r in series],marker='o',label=arm)
            ax.set(xlabel='Acquisition round',ylabel=metric,ylim=(0,1));ax.legend()
    fig.suptitle('Exploratory comparison — one training seed');fig.tight_layout()
    fig.savefig(out/'combined_curves.png',dpi=160)
    return comparisons, fig
