"""Simulator compatibility checks and state-policy evaluation."""
from collections import deque
from pathlib import Path
import os
import numpy as np
import torch
from .data import encode
from .artifacts import write_json, digest

def make_env(max_steps=300):
    os.environ.setdefault("SDL_VIDEODRIVER","dummy")
    import gymnasium as gym
    import gym_pusht  # registers the environment
    return gym.make("gym_pusht/PushT-v0",obs_type="state",render_mode="rgb_array",max_episode_steps=max_steps)

def coverage(env,info):
    # Reward is coverage / success threshold, clipped. It is not raw overlap.
    return float(info["coverage"]) if "coverage" in info else float(env.unwrapped._get_coverage())

def pose_error(actual,expected):
    d=np.asarray(actual)-np.asarray(expected)
    angle=np.arctan2(np.sin(d[...,4]),np.cos(d[...,4]))
    return np.linalg.norm(d[...,:4],axis=-1),np.abs(angle)

def reset_recorded(env,state):
    """Restore observed pose, compensating the legacy reset position shift.

    Not a full physics snapshot: hidden velocities/contact caches are unavailable.
    Intended only for the expert replay diagnostic, not random evaluation resets.
    """
    raw,_=env.reset(seed=0,options={"reset_to_state":state.tolist()})
    request=np.asarray(state,dtype=float).copy()
    corrected=raw
    for _ in range(5):
        pos,ang=pose_error(corrected,state)
        if pos<1e-4 and ang<1e-4: break
        request[:4]+=state[:4]-corrected[:4]
        corrected,_=env.reset(seed=0,options={"reset_to_state":request.tolist()})
    return corrected,raw

def check_replay(episodes,output="runs/compatibility",count=3,position_tolerance=5.,angle_tolerance=.1):
    out=Path(output);out.mkdir(parents=True,exist_ok=True);rows=[]
    for episode in episodes[:count]:
        env=make_env(max_steps=max(300,len(episode.states)+1))
        try:
            state,raw=reset_recorded(env,episode.states[0]);initial=state.copy()
            recorded=[];simulated=[];cov=[]
            for t in range(len(episode.states)-1):
                recorded.append(episode.states[t]); simulated.append(state.copy())
                state,_,done,truncated,info=env.step(episode.actions[t])
                cov.append(coverage(env,info))
                if done or truncated: break
            simulated.append(state.copy());recorded.append(episode.states[len(simulated)-1])
            positions,angles=pose_error(simulated,recorded)
            reset_position,reset_angle=pose_error(initial,episode.states[0])
            row=dict(episode_id=episode.id,steps=len(cov),raw_reset_state=raw.tolist(),corrected_reset_state=initial.tolist(),
                     reset_position_error=float(reset_position),reset_angle_error=float(reset_angle),
                     position_rmse=float(np.sqrt(np.mean(positions**2))),angle_rmse=float(np.sqrt(np.mean(angles**2))),
                     max_coverage=max(cov,default=0.))
            row["passed"]=bool(reset_position<1e-3 and reset_angle<1e-3 and row["position_rmse"]<=position_tolerance and row["angle_rmse"]<=angle_tolerance)
            rows.append(row)
            np.savez_compressed(out/f"replay_{episode.id}.npz",recorded=recorded,simulated=simulated,coverage=cov)
        finally: env.close()
    result=dict(passed=bool(rows) and all(x["passed"] for x in rows),episodes=rows,
                position_tolerance=position_tolerance,angle_tolerance=angle_tolerance,
                limitation="Pose restoration lacks recorded velocities; divergence requires diagnosis, not threshold relaxation.")
    write_json(out/"report.json",result);return result

def evaluate(policy,obs_norm,action_norm,seeds,output,checkpoint_id,execute=8,max_steps=300,video_count=3,inference_steps=None):
    if not 1<=execute<=policy.horizon: raise ValueError("Invalid execution horizon")
    output=Path(output);output.mkdir(parents=True,exist_ok=True);policy.eval()
    device=next(policy.parameters()).device;rows=[]
    seeds=list(seeds)
    if not seeds or len(set(seeds))!=len(seeds): raise ValueError("Nonempty unique seeds required")
    identity=dict(schema_version=2,checkpoint=checkpoint_id,execute=execute,max_steps=max_steps,
                  sampler="ddpm" if inference_steps is None else "ddim",inference_steps=inference_steps)
    signature=digest(identity)
    for index,seed in enumerate(seeds):
        path=output/f"episode_{seed}.json"
        if path.exists():
            import json
            row=json.loads(path.read_text())
            if row["signature"]!=signature: raise ValueError("Evaluation directory belongs to a different configuration; use a new directory")
            trace_exists=False
            try:
                with np.load(output/f"episode_{seed}.npz") as trace:
                    trace_exists=len(trace['actions'])==row['steps'] and len(trace['next_states'])==row['steps']
            except (OSError,ValueError,KeyError): pass
            video_exists=index>=video_count or (output/f"episode_{seed}.mp4").exists()
            if trace_exists and video_exists:
                rows.append(row);continue
        env=make_env(max_steps);writer=None
        try:
            state,info=env.reset(seed=int(seed));initial=state.copy()
            history=deque([state.copy() for _ in range(policy.history)],maxlen=policy.history)
            generator=torch.Generator(device=device).manual_seed(int(seed)+1_000_000)
            states=[];next_states=[];actions=[];coverages=[];rewards=[];success=False;done=False;truncated=False
            if index<video_count:
                import imageio.v2 as imageio
                writer=imageio.get_writer(str(output/f"episode_{seed}.mp4"),fps=10)
                writer.append_data(env.render())
            while len(actions)<max_steps and not (done or truncated):
                obs=obs_norm.transform(encode(np.asarray(history)))
                with torch.no_grad():
                    predicted=policy.sample(torch.as_tensor(obs[None],device=device),generator=generator,inference_steps=inference_steps)[0].cpu().numpy()
                planned=action_norm.inverse(predicted)
                if not np.isfinite(planned).all(): raise FloatingPointError("Nonfinite policy actions")
                for action in planned[:execute]:
                    action=np.clip(action,0,512).astype(np.float32)
                    states.append(state.copy());actions.append(action)
                    state,reward,done,truncated,info=env.step(action)
                    next_states.append(state.copy());history.append(state.copy());coverages.append(coverage(env,info));rewards.append(float(reward))
                    success=success or bool(info.get("is_success",done))
                    if writer: writer.append_data(env.render())
                    if done or truncated or len(actions)>=max_steps: break
            np.savez_compressed(output/f"episode_{seed}.npz",states=states,next_states=next_states,actions=actions,coverage=coverages,reward=rewards,final_state=state)
            row=dict(signature=signature,seed=int(seed),initial_state=initial.tolist(),inference_seed=int(seed)+1_000_000,
                     success=success,max_coverage=max(coverages,default=0.),steps=len(actions),terminated=bool(done),truncated=bool(truncated))
            if writer:
                writer.close();writer=None
            write_json(path,row);rows.append(row)
            print(f"Evaluation {index+1}/{len(seeds)}: success={success}, max coverage={row['max_coverage']:.3f}",flush=True)
        finally:
            if writer: writer.close()
            env.close()
    summary=dict(**identity,seeds=seeds,episodes=len(rows),success_rate=float(np.mean([x["success"] for x in rows])),
                 mean_max_coverage=float(np.mean([x["max_coverage"] for x in rows])),rows=rows)
    write_json(output/"summary.json",summary);return summary
