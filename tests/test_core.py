import unittest
import math
import numpy as np
import torch
from robot_flywheel.data import Episode, window, encode, Normalizer
from robot_flywheel.selection import geometry, select
from robot_flywheel.policy import DiffusionPolicy
from robot_flywheel.artifacts import manifest

class CoreTests(unittest.TestCase):
    def ep(self,i,x=0):
        return Episode(i,np.tile([x,10,x,10,0],(4,1)),np.tile([x,10],(4,1)))
    def test_reject_incomplete_state(self):
        with self.assertRaises(ValueError): Episode(0,np.zeros((4,2)),np.zeros((4,2)))
    def test_boundaries_and_alignment(self):
        e=self.ep(1); e.actions[:,0]=np.arange(4)
        o,a,m=window(e,3,history=2,horizon=4)
        np.testing.assert_array_equal(a[:,0],[3,3,3,3]);np.testing.assert_array_equal(m,[True,False,False,False])
        o,_,_=window(e,0); np.testing.assert_array_equal(o[0],o[1])
    def test_angle_wrap(self):
        a=np.array([[0,0,10,10,0],[0,0,10,10,2*math.pi]])
        np.testing.assert_allclose(geometry(a)[0],geometry(a)[1],atol=1e-6)
    def test_selector_relevance_and_no_reacquisition(self):
        pool=[self.ep(0,0),self.ep(1,250),self.ep(2,500)]
        ids,log=select(pool,[0],1,'targeted',failure_features=geometry(pool[2].states))
        self.assertEqual(ids,[2]); self.assertGreater(log[0]['marginal_gain'],0)
        with self.assertRaises(ValueError): select(pool,[0],3,'random')
    def test_fallback_determinism(self):
        pool=[self.ep(i,i*100) for i in range(4)]
        a,_=select(pool,[0],2,'targeted');b,_=select(pool,[0],2,'diversity')
        self.assertEqual(a,b)
    def test_manifest_integrity(self):
        a=manifest([self.ep(0)]); b=manifest([self.ep(1)])
        self.assertNotEqual(a['sha256'],b['sha256'])
        with self.assertRaises(ValueError):manifest([self.ep(0),self.ep(0)])
    def test_diffusion_gradients_and_sampling(self):
        torch.set_num_threads(1);torch.manual_seed(3)
        m=DiffusionPolicy(width=16,steps=4)
        obs=torch.zeros(2,2,6); act=torch.zeros(2,16,2);mask=torch.ones(2,16)
        loss=m.loss(obs,act,mask);loss.backward()
        self.assertTrue(torch.isfinite(loss));self.assertTrue(any(p.grad is not None for p in m.parameters()))
        a=m.sample(obs,generator=torch.Generator().manual_seed(4))
        b=m.sample(obs,generator=torch.Generator().manual_seed(4))
        self.assertEqual(a.shape,(2,16,2));self.assertTrue(torch.isfinite(a).all());torch.testing.assert_close(a,b)

if __name__=='__main__':unittest.main()
