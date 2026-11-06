import numpy as np
import torch
from .data import window

def train_updates(policy, optimizer, episodes, obs_norm, action_norm, updates, batch_size=32, seed=0, rng=None, on_update=None):
    """Episode-uniform then timestep-uniform sampling, shared across arms."""
    rng=np.random.default_rng(seed) if rng is None else rng; device=next(policy.parameters()).device
    policy.train(); losses=[]
    for _ in range(updates):
        batch=[]
        for _ in range(batch_size):
            ep=episodes[int(rng.integers(len(episodes)))]
            batch.append(window(ep,int(rng.integers(len(ep.states))),policy.history,policy.horizon))
        obs,act,mask=map(np.stack,zip(*batch))
        tensors=[torch.as_tensor(x,device=device) for x in (obs_norm.transform(obs),action_norm.transform(act),mask.astype(np.float32))]
        optimizer.zero_grad(set_to_none=True); loss=policy.loss(*tensors)
        if not torch.isfinite(loss): raise FloatingPointError("Nonfinite training loss")
        loss.backward(); torch.nn.utils.clip_grad_norm_(policy.parameters(),1.0); optimizer.step()
        if on_update is not None: on_update(policy)
        losses.append(float(loss.detach()))
    return losses
