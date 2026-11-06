import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from robot_flywheel.data import Episode
from robot_flywheel.artifacts import write_json
from robot_flywheel.experiment import Protocol,run_study
from robot_flywheel.report import paired_interval

# Deterministic rollout double tests orchestration, not simulator performance.
def fixture_evaluate(policy,on,an,seeds,output,checkpoint_id,**kwargs):
    out=Path(output);out.mkdir(parents=True,exist_ok=True);rows=[]
    for seed in seeds:
        states=np.tile([100.,100.,200.,200.,0.],(4,1))
        np.savez(out/f'episode_{seed}.npz',states=states,coverage=np.zeros(4))
        rows.append(dict(seed=seed,success=False,max_coverage=0.,initial_state=states[0].tolist()))
    summary=dict(rows=rows,success_rate=0.,mean_max_coverage=0.)
    write_json(out/'summary.json',summary);return summary

class ExperimentTests(unittest.TestCase):
    def test_three_rounds_branches_and_resume(self):
        torch.set_num_threads(1);rng=np.random.default_rng(3)
        episodes=[Episode(i,rng.uniform(0,300,(5,5)),rng.uniform(0,512,(5,2))) for i in range(8)]
        with tempfile.TemporaryDirectory() as directory,patch('robot_flywheel.experiment.evaluate',side_effect=fixture_evaluate):
            p=Protocol.quick();records=run_study(episodes,{'archive_sha256':'fixture'},p,directory,{'passed':True},'cpu')
            self.assertEqual(len(records),13)
            for r in records:
                expected=4 if r['arm'] in ('baseline','continued') else 4+r['round']
                self.assertEqual(r['episodes'],expected)
                state=torch.load(r['checkpoint'],weights_only=False)
                self.assertEqual(state['update'],4+r['round']*2)
            baseline=torch.load(records[0]['checkpoint'],weights_only=False)
            for r in records[1:]:
                stage=torch.load(r['checkpoint'],weights_only=False)
                self.assertEqual(stage['obs_norm'],baseline['obs_norm'])
            with patch('robot_flywheel.baseline.train_updates',side_effect=AssertionError('Resume retrained completed stage')):
                self.assertEqual(len(run_study(episodes,{'archive_sha256':'fixture'},p,directory,{'passed':True},'cpu')),13)
            p.acquire=2
            with self.assertRaises(ValueError):run_study(episodes,{'archive_sha256':'fixture'},p,directory,{'passed':True},'cpu')
    def test_seed_leakage_rejected(self):
        p=Protocol.quick();p.evaluation_start=p.diagnostic_start
        with self.assertRaises(ValueError):p.validate(20)
    def test_crossed_bootstrap_preserves_pairing(self):
        x=np.array([[1,0,1],[0,1,0]])
        self.assertEqual(paired_interval(x,x)['interval95_pp'],[0.,0.])
        self.assertEqual(paired_interval(np.ones((2,3)),np.zeros((2,3)))['difference_pp'],100.)
