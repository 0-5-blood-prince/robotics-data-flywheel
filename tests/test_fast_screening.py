import unittest
from unittest.mock import patch
import torch
from robot_flywheel.policy import DiffusionPolicy
from robot_flywheel.experiment import Protocol

class FastScreenTests(unittest.TestCase):
    def test_ddim_forward_budget_and_repeatability(self):
        torch.set_num_threads(1);p=DiffusionPolicy(width=16,steps=100);obs=torch.zeros(1,2,6)
        with patch.object(p,'forward',wraps=p.forward) as forward:
            a=p.sample(obs,generator=torch.Generator().manual_seed(2),inference_steps=10)
            self.assertEqual(forward.call_count,10)
        b=p.sample(obs,generator=torch.Generator().manual_seed(2),inference_steps=10)
        torch.testing.assert_close(a,b)
        self.assertTrue(torch.isfinite(a).all())
    def test_learning_gate(self):
        p=Protocol()
        with self.assertRaisesRegex(ValueError,'Learning gate'):p.validate(206)
        p.development_success_rate=.3;p.validate(206)
        Protocol.quick().validate(206)
