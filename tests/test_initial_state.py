import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from robot_flywheel.data import Episode
from robot_flywheel.initial_state import initial_features,select_starts,StartConfig,run_failure_start,compare_failure_start,sha
from robot_flywheel.experiment import Protocol,run_study
from robot_flywheel.artifacts import write_json

class InitialStateTests(unittest.TestCase):
    def test_periodic_angle(self):
        a=np.array([100,100,200,200,.01])
        b=a.copy();b[-1]+=2*np.pi
        np.testing.assert_allclose(initial_features(a),initial_features(b),atol=1e-7)

    def test_matching_and_coverage(self):
        episodes=[Episode(i,[[x,100,x,100,0]],[[x,100]]) for i,x in enumerate([0,10,200,300,500])]
        diagnostic={'rows':[dict(seed=2,success=False,max_coverage=.1,initial_state=[11,100,11,100,0]),
                            dict(seed=3,success=True,max_coverage=1.,initial_state=[300,100,300,100,0])]}
        ids,log=select_starts(episodes,[0],2,diagnostic)
        self.assertEqual(ids,[1,4]);self.assertEqual(log[0]['failure_seed'],2)
        self.assertEqual(log[1]['reason'],'initial_coverage_fill')
        ids,log=select_starts(episodes,[0],2,{'rows':[]})
        self.assertEqual(ids,[4,2]);self.assertTrue(all(r['reason']=='initial_coverage_fill' for r in log))

    def test_one_round_extension_resume_and_report(self):
        def evaluation(model,on,an,seeds,output,checkpoint_id,**kwargs):
            out=Path(output);out.mkdir(parents=True,exist_ok=True);rows=[]
            for seed in seeds:
                state=[100.,100.,200.,200.,0.]
                np.savez(out/f'episode_{seed}.npz',states=np.tile(state,(4,1)),next_states=np.tile(state,(4,1)),coverage=np.zeros(4))
                rows.append(dict(seed=seed,success=False,max_coverage=0.,initial_state=state))
            s=dict(rows=rows,seeds=list(seeds),success_rate=0.,mean_max_coverage=0.,checkpoint=checkpoint_id,
                   execute=kwargs['execute'],max_steps=kwargs['max_steps'],sampler='ddpm')
            write_json(out/'summary.json',s);return s
        torch.set_num_threads(1);rng=np.random.default_rng(12)
        episodes=[Episode(i,rng.uniform(0,400,(6,5)),rng.uniform(0,512,(6,2))) for i in range(8)]
        source={'archive_sha256':'fixture'}
        with tempfile.TemporaryDirectory() as tmp:
            old=Path(tmp)/'old';out=Path(tmp)/'new'
            with patch('robot_flywheel.experiment.evaluate',side_effect=evaluation):
                run_study(episodes,source,Protocol.quick(),old,{'passed':True},'cpu')
            before={str(p):sha(p) for p in old.rglob('*') if p.is_file()}
            with patch('robot_flywheel.initial_state.evaluate',side_effect=evaluation):
                rows=run_failure_start(episodes,source,old,out,{'passed':True},device='cpu',final_evaluation=True)
                self.assertEqual(len(rows),1)
                comparisons,fig=compare_failure_start(old,out)
                self.assertEqual(json.loads((out/'comparisons.json').read_text())['round'],1)
                selection=json.loads((Path(rows[0]['checkpoint']).parent/'selection.json').read_text())
                self.assertEqual(selection['parent_checkpoint'],before[str(old/'seed_11/baseline/latest.pt')])
                rows=run_failure_start(episodes,source,old,out,{'passed':True},rounds_to_run=3,device='cpu',final_evaluation=True)
                self.assertEqual(len(rows),3)
            with patch('robot_flywheel.initial_state.evaluate',side_effect=AssertionError('Unexpected diagnostics')):
                with patch('robot_flywheel.baseline.train_updates',side_effect=AssertionError('Unexpected training')):
                    run_failure_start(episodes,source,old,out,{'passed':True},rounds_to_run=3,device='cpu')
                    run_failure_start(episodes,source,old,out,{'passed':True},rounds_to_run=1,device='cpu')
            self.assertEqual(len(json.loads((out/'checkpoints.json').read_text())),3)
            self.assertEqual(before,{str(p):sha(p) for p in old.rglob('*') if p.is_file()})
            with self.assertRaisesRegex(ValueError,'inputs changed'):
                run_failure_start(episodes,source,old,out,{'passed':True},StartConfig(.25),device='cpu')
