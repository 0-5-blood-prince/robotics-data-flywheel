import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from robot_flywheel.data import Episode
from robot_flywheel.artifacts import write_json
from robot_flywheel.experiment import Protocol
from robot_flywheel.replication import run_random_reference
from robot_flywheel.initial_state import run_failure_start,compare_failure_start,read,sha

class ReplicationTests(unittest.TestCase):
    def test_shared_new_seed_baseline_and_two_arm_resume(self):
        def evaluation(model,on,an,seeds,output,checkpoint_id,**kw):
            out=Path(output);out.mkdir(parents=True,exist_ok=True)
            rows=[dict(seed=s,initial_state=[100,100,200,200,0],success=False,max_coverage=0.) for s in seeds]
            summary=dict(rows=rows,seeds=list(seeds),success_rate=0.,mean_max_coverage=0.,checkpoint=checkpoint_id,
                sampler='ddpm',execute=kw['execute'],max_steps=kw['max_steps'])
            write_json(out/'summary.json',summary);return summary
        torch.set_num_threads(1);rng=np.random.default_rng(8)
        episodes=[Episode(i,rng.uniform(0,400,(6,5)),rng.uniform(0,512,(6,2))) for i in range(8)]
        source={'archive_sha256':'fixture'};p=Protocol.quick();p.training_seeds=[22];p.rounds=2;p.acquire=2;p.initial_episodes=2
        with tempfile.TemporaryDirectory() as d:
            ref=Path(d)/'reference';out=Path(d)/'failure'
            with patch('robot_flywheel.replication.evaluate',side_effect=evaluation),patch('robot_flywheel.initial_state.evaluate',side_effect=evaluation):
                reference=run_random_reference(episodes,source,p,ref,{'passed':True},'cpu',True)
                new=run_failure_start(episodes,source,ref,out,{'passed':True},rounds_to_run=2,device='cpu',final_evaluation=True)
            self.assertEqual(len(reference)+len(new),5)
            for arm,home in [('random',ref),('failure_start',out)]:
                self.assertEqual(read(home/'seed_22'/arm/'round_1/selection.json')['parent_checkpoint'],sha(ref/'seed_22/baseline/latest.pt'))
                state=torch.load(home/'seed_22'/arm/'round_2/latest.pt',weights_only=False)
                self.assertEqual(state['config']['seed'],22);self.assertEqual(state['update'],8)
                self.assertEqual(len(state['manifest']['episode_ids']),6)
            comparisons,fig=compare_failure_start(ref,out)
            self.assertEqual(set(comparisons),{'random'})
            self.assertEqual(read(out/'comparisons.json')['round'],2)
            with patch('robot_flywheel.baseline.train_updates',side_effect=AssertionError('Retraining')):
                run_random_reference(episodes,source,p,ref,{'passed':True},'cpu',False)
                run_failure_start(episodes,source,ref,out,{'passed':True},rounds_to_run=2,device='cpu')
