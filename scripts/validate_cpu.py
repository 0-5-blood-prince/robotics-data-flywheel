"""Run the actual data/simulator/training/report path at tiny CPU budgets."""
import argparse
import json
from pathlib import Path
import torch
from robot_flywheel.prepare import prepare
from robot_flywheel.rollout import check_replay,evaluate
from robot_flywheel.experiment import Protocol,run_study
from robot_flywheel.report import build_report
from robot_flywheel.baseline import load_policy
from robot_flywheel.artifacts import write_json

p=argparse.ArgumentParser();p.add_argument('--output',default='runs/cpu-validation');p.add_argument('--data',default='data/pusht')
args=p.parse_args();out=Path(args.output);torch.set_num_threads(1)
episodes,source=prepare(args.data)
compat=check_replay(episodes,out/'compatibility')
assert compat['passed'],compat
protocol=Protocol.quick()
records=run_study(episodes,source,protocol,out/'study',compat,device='cpu')
assert len(records)==13
comparison,figure=build_report(out/'study')
# Re-running must reuse completed training and cached evaluations.
from unittest.mock import patch
with patch('robot_flywheel.baseline.train_updates',side_effect=AssertionError('Completed training unexpectedly rerun')):
    resumed=run_study(episodes,source,protocol,out/'study',compat,device='cpu')
assert len(resumed)==13
model,on,an,sha=load_policy(records[-1]['checkpoint'],'cpu')
evaluate(model,on,an,[2000],out/'video',sha,max_steps=8,video_count=1)
import imageio.v2 as imageio
reader=imageio.get_reader(out/'video/episode_2000.mp4')
try:frame=reader.get_data(0)
finally:reader.close()
assert frame.ndim==3
write_json(out/'validation.json',dict(status='passed',performance_evidence=False,episodes=len(episodes),
    checkpoint_evaluations=len(records),replay=compat,video_shape=list(frame.shape),
    verified=['data load','expert replay','training','three acquisition rounds','all controls','resume','paired statistics','report','video decode']))
print('CPU VALIDATION PASSED:',out/'validation.json')
