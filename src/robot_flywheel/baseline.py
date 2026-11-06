"""Resumable real-data baseline. Run only after compatibility gates pass."""
from dataclasses import dataclass,asdict
from pathlib import Path
import copy
import hashlib
import os
import tempfile
import time
import numpy as np
import torch
from .data import Normalizer,encode
from .policy import DiffusionPolicy
from .train import train_updates
from .artifacts import digest,manifest,write_json

@dataclass
class BaselineConfig:
    seed:int=42
    initial_episodes:int=50
    updates:int=20000
    batch_size:int=128
    checkpoint_every:int=1000
    learning_rate:float=0.0001
    width:int=64
    diffusion_steps:int=100
    history:int=2
    horizon:int=16
    ema_decay:float=.995

def atomic_save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(dir=path.parent);os.close(fd)
    try:
        torch.save(value,tmp);os.replace(tmp,path)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)

def model_config(c):
    return dict(history=c.history,horizon=c.horizon,width=c.width,steps=c.diffusion_steps)

def train_baseline(episodes,source,config,output="runs/baseline",device=None,compatibility=None):
    if not compatibility or not compatibility.get("passed"):
        raise ValueError("Real-data baseline requires a passed compatibility report")
    c=config;out=Path(output);out.mkdir(parents=True,exist_ok=True)
    if not 1<=c.initial_episodes<=len(episodes) or c.updates<1 or c.checkpoint_every<1:
        raise ValueError("Invalid data or training budget")
    device=device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(c.seed);rng=np.random.default_rng(c.seed)
    indices=rng.permutation(len(episodes))[:c.initial_episodes]
    selected=[episodes[int(i)] for i in indices]
    data_manifest=manifest(selected,source_sha256=source["archive_sha256"])
    # updates may be extended, but every other configuration field must match.
    contract=asdict(c);contract.pop("updates")
    run_id=digest(dict(config=contract,manifest=data_manifest))
    obs=Normalizer.fit([encode(e.states) for e in selected]);act=Normalizer.fit([e.actions for e in selected])
    policy=DiffusionPolicy(**model_config(c)).to(device);ema=copy.deepcopy(policy)
    optimizer=torch.optim.AdamW(policy.parameters(),lr=c.learning_rate)
    checkpoint=out/"latest.pt";start=0;losses=[];elapsed=0.
    if checkpoint.exists():
        saved=torch.load(checkpoint,map_location=device,weights_only=False)
        if saved["run_id"]!=run_id:raise ValueError("Checkpoint config or data differs; use a new output directory")
        policy.load_state_dict(saved["model"]);ema.load_state_dict(saved["ema"]);optimizer.load_state_dict(saved["optimizer"])
        start=saved["update"];losses=saved["losses"];elapsed=saved["elapsed_seconds"]
        rng.bit_generator.state=saved["numpy_rng"]
        torch.set_rng_state(saved["torch_rng"].cpu())
        if device.startswith("cuda") and saved["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all([x.cpu() for x in saved["cuda_rng"]])
        obs=Normalizer(**saved["obs_norm"]);act=Normalizer(**saved["action_norm"])
    write_json(out/"dataset.json",data_manifest)
    write_json(out/"config.json",dict(config=asdict(c),device=device,run_id=run_id,model_parameters=sum(p.numel() for p in policy.parameters())))
    begin=time.monotonic()
    def update_ema(model):
        with torch.no_grad():
            for target,current in zip(ema.parameters(),model.parameters()):target.lerp_(current,1-c.ema_decay)
    for block in range(start,c.updates,c.checkpoint_every):
        n=min(c.checkpoint_every,c.updates-block)
        values=train_updates(policy,optimizer,selected,obs,act,n,c.batch_size,rng=rng,on_update=update_ema)
        losses.extend(values);step=block+n
        saved=dict(run_id=run_id,model_config=model_config(c),model=policy.state_dict(),ema=ema.state_dict(),optimizer=optimizer.state_dict(),
                   obs_norm=obs.to_dict(),action_norm=act.to_dict(),update=step,losses=losses,
                   elapsed_seconds=elapsed+time.monotonic()-begin,numpy_rng=rng.bit_generator.state,
                   torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if device.startswith("cuda") else None,
                   manifest=data_manifest,config=asdict(c))
        atomic_save(checkpoint,saved);atomic_save(out/f"step_{step:07d}.pt",saved)
        write_json(out/"training.json",dict(update=step,losses=losses,elapsed_seconds=saved["elapsed_seconds"]))
        print(f"Update {step}/{c.updates}: mean loss {np.mean(values):.4f}, elapsed {saved['elapsed_seconds']:.0f}s",flush=True)
    return checkpoint

def load_policy(path,device=None):
    device=device or ("cuda" if torch.cuda.is_available() else "cpu")
    saved=torch.load(path,map_location=device,weights_only=False)
    model=DiffusionPolicy(**saved["model_config"]).to(device)
    model.load_state_dict(saved["ema"]);model.eval()
    return model,Normalizer(**saved["obs_norm"]),Normalizer(**saved["action_norm"]),hashlib.sha256(Path(path).read_bytes()).hexdigest()
