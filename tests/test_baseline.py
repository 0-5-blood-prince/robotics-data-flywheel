import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from robot_flywheel.data import Episode
from robot_flywheel.baseline import BaselineConfig,train_baseline,load_policy
from robot_flywheel.rollout import pose_error,coverage

class BaselineTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        rng=np.random.default_rng(8)
        self.episodes=[Episode(i,rng.normal(size=(8,5)),rng.normal(size=(8,2))) for i in range(4)]
    def test_resume_matches_uninterrupted(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg=BaselineConfig(initial_episodes=2,updates=4,checkpoint_every=2,batch_size=2,width=16,diffusion_steps=4)
            full=train_baseline(self.episodes,{'archive_sha256':'fixture'},cfg,Path(tmp)/'full',device='cpu',compatibility={'passed':True})
            cfg.updates=2
            train_baseline(self.episodes,{'archive_sha256':'fixture'},cfg,Path(tmp)/'resumed',device='cpu',compatibility={'passed':True})
            cfg.updates=4
            resumed=train_baseline(self.episodes,{'archive_sha256':'fixture'},cfg,Path(tmp)/'resumed',device='cpu',compatibility={'passed':True})
            a=torch.load(full,weights_only=False);b=torch.load(resumed,weights_only=False)
            self.assertEqual(a['losses'],b['losses'])
            for key in a['model']:torch.testing.assert_close(a['model'][key],b['model'][key],rtol=0,atol=0)
            for key in a['ema']:torch.testing.assert_close(a['ema'][key],b['ema'][key],rtol=0,atol=0)
            model,obs,act,sha=load_policy(resumed,device='cpu')
            self.assertEqual(len(sha),64)
            cfg.width=32
            with self.assertRaises(ValueError):train_baseline(self.episodes,{'archive_sha256':'fixture'},cfg,Path(tmp)/'resumed',device='cpu',compatibility={'passed':True})
    def test_compatibility_is_required(self):
        with self.assertRaises(ValueError):train_baseline(self.episodes,{},BaselineConfig())
    def test_wrapped_pose_error_and_coverage(self):
        p,a=pose_error([0,0,0,0,2*np.pi],[0,0,0,0,0])
        self.assertAlmostEqual(float(a),0);self.assertEqual(float(p),0)
        self.assertEqual(coverage(None,{'coverage':.76}),.76)
if __name__=='__main__':unittest.main()
