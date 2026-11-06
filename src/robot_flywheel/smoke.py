"""Synthetic execution check only. Does not measure robotics performance."""
import argparse
import numpy as np
import torch
from .data import Episode, Normalizer, encode
from .policy import DiffusionPolicy
from .train import train_updates
from .selection import select
from .artifacts import manifest, write_json

def main():
    p=argparse.ArgumentParser(); p.add_argument("--output",default="runs/smoke.json"); args=p.parse_args()
    torch.manual_seed(7); torch.set_num_threads(1); rng=np.random.default_rng(7)
    eps=[Episode(i,np.column_stack((rng.uniform(50,450,(24,4)),rng.uniform(-3,3,24))),rng.uniform(0,512,(24,2))) for i in range(8)]
    obs=Normalizer.fit([encode(e.states) for e in eps[:2]]); act=Normalizer.fit([e.actions for e in eps[:2]])
    model=DiffusionPolicy(width=16,steps=8); opt=torch.optim.AdamW(model.parameters(),lr=1e-3)
    losses=train_updates(model,opt,eps[:2],obs,act,3,batch_size=4)
    ids,log=select(eps,[0,1],2,"targeted")
    model.eval(); sample=model.sample(torch.zeros(1,2,6))
    assert sample.shape==(1,16,2) and torch.isfinite(sample).all()
    write_json(args.output,dict(status="synthetic smoke passed",not_a_performance_result=True,losses=losses,selection=log,manifest=manifest([eps[i] for i in [0,1]+ids])))
    print(args.output)
if __name__=="__main__": main()
