import unittest
import tempfile
from pathlib import Path
import numpy as np
import torch
from robot_flywheel.demo_score import ScoreConfig, fit_classifier, rank_episodes, rollout_data
from robot_flywheel.data import Episode
from robot_flywheel.artifacts import write_json

class DemoScoreTests(unittest.TestCase):
    def test_outcome_signal_and_episode_ranking(self):
        torch.set_num_threads(1)
        xs = [np.full((i+2,6), -1. if i%2==0 else 1.,dtype=np.float32) for i in range(12)]
        ys = np.array([i%2 for i in range(12)],dtype=np.float32)
        cfg=ScoreConfig(classifier_steps=100,check_every=10,learning_rate=.03)
        fitted, report=fit_classifier((xs,ys),(xs,ys),cfg)
        self.assertTrue(report['beats_constant'])
        episodes=[Episode(i,np.full((4,5),v),np.zeros((4,2))) for i,v in [(0,-1.),(1,1.)]]
        self.assertEqual(rank_episodes(episodes,fitted)[0]['episode_id'],1)
        fitted2, report2=fit_classifier((xs,ys),(xs,ys),cfg)
        self.assertEqual(report,report2)

    def test_rejects_insufficient_classes(self):
        data=([np.ones((4,6),dtype=np.float32)]*10,np.zeros(10,dtype=np.float32))
        with self.assertRaisesRegex(ValueError,'successes and'):
            fit_classifier(data,data,ScoreConfig())

    def test_trajectory_reader_uses_pre_action_states_only(self):
        with tempfile.TemporaryDirectory() as d:
            write_json(Path(d)/'summary.json',dict(rows=[dict(seed=3,success=True)]))
            np.savez(Path(d)/'episode_3.npz',states=np.zeros((2,5)),next_states=np.ones((2,5)),coverage=[.9,1.])
            xs,ys=rollout_data(d)
            self.assertTrue((xs[0][:,:4]==0).all())
            self.assertEqual(ys.tolist(),[1.])

    def test_new_arm_only_resume_and_combined_report(self):
        import hashlib
        import json
        from unittest.mock import patch
        from robot_flywheel.experiment import Protocol, run_study
        from robot_flywheel.demo_score import prepare_success_guided, run_success_guided, compare_success_guided
        # Synthetic outcome fixture exercises orchestration, not robot performance.
        def evaluate(policy,on,an,seeds,output,checkpoint_id,**kwargs):
            out=Path(output);out.mkdir(parents=True,exist_ok=True);rows=[]
            for seed in seeds:
                success=bool(seed%2)
                state=[400.,400.,400.,400.,0.] if success else [100.,100.,100.,100.,0.]
                states=np.tile(state,(4,1))
                np.savez(out/f'episode_{seed}.npz',states=states,next_states=states,coverage=np.zeros(4))
                rows.append(dict(seed=seed,success=success,max_coverage=float(success),initial_state=state))
            summary=dict(rows=rows,success_rate=np.mean([r['success'] for r in rows]),mean_max_coverage=.5,
                         checkpoint=checkpoint_id,execute=kwargs['execute'],max_steps=kwargs['max_steps'],sampler='ddpm')
            write_json(out/'summary.json',summary)
            return summary
        torch.set_num_threads(1)
        rng=np.random.default_rng(1)
        episodes=[Episode(i,rng.uniform(0,400,(6,5)),rng.uniform(0,512,(6,2))) for i in range(8)]
        source={'archive_sha256':'synthetic'}
        with tempfile.TemporaryDirectory() as tmp:
            original=Path(tmp)/'original'; out=Path(tmp)/'extension'
            with patch('robot_flywheel.experiment.evaluate',side_effect=evaluate):
                run_study(episodes,source,Protocol.quick(),original,{'passed':True},'cpu')
            before={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in original.rglob('*') if p.is_file()}
            c=ScoreConfig(rollouts_per_checkpoint=12,classifier_steps=100,learning_rate=.03,check_every=10)
            with patch('robot_flywheel.demo_score.evaluate',side_effect=evaluate):
                report=prepare_success_guided(episodes,source,original,out,{'passed':True},c,'cpu')
                self.assertTrue(report['beats_constant'])
                records=run_success_guided(episodes,source,original,out,{'passed':True},'cpu',True)
            self.assertEqual(len(records),3)
            for r in records:
                self.assertEqual(r['arm'],'success_guided')
                self.assertEqual(r['episodes'],4+r['round'])
                state=torch.load(r['checkpoint'],weights_only=False)
                self.assertEqual(state['update'],4+2*r['round'])
                selection=json.loads((Path(r['checkpoint']).parent/'selection.json').read_text())
                if r['round']==1:
                    self.assertEqual(selection['parent_checkpoint'],before[str(original/'seed_11/baseline/latest.pt')])
            with patch('robot_flywheel.demo_score.evaluate',side_effect=AssertionError('Unexpected rollout')):
                prepare_success_guided(episodes,source,original,out,{'passed':True},c,'cpu')
                with patch('robot_flywheel.baseline.train_updates',side_effect=AssertionError('Unexpected retraining')):
                    run_success_guided(episodes,source,original,out,{'passed':True},'cpu',False)
            import matplotlib
            matplotlib.use('Agg')
            comparisons,fig=compare_success_guided(original,out)
            self.assertEqual(set(comparisons),{'random','continued','diversity','targeted'})
            self.assertEqual(len(json.loads((out/'combined_results.json').read_text())['records']),16)
            after={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in original.rglob('*') if p.is_file()}
            self.assertEqual(before,after)
            # Any protocol change must fail before it can reuse classifier artifacts.
            c.rollout_start+=1
            with self.assertRaisesRegex(ValueError,'inputs changed'):
                prepare_success_guided(episodes,source,original,out,{'passed':True},c,'cpu')
