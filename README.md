# Robotics Data Flywheel

State-based PushT diffusion-policy learning with controlled demonstration acquisition.

## Run the real-data baseline in Colab

[Open in Colab](https://colab.research.google.com/github/0-5-blood-prince/robotics-data-flywheel/blob/main/notebooks/flywheel.ipynb)

1. Open the link above and select a GPU runtime.
2. Run setup: it fetches source directly from GitHub and prints the exact commit.
3. Optionally enable Google Drive persistence for checkpoints.
4. Run dataset inspection and the expert replay check, then training and development evaluation.

Keep this notebook link. Future code updates require restarting the runtime and rerunning setup, not uploading files. Set `REVISION` to a commit SHA for reproducible runs. A saved personal notebook copy retains its own cells; reopen the link if notebook cells themselves change.

Training defaults to a provisional 50-demonstration, 20,000-update pilot. This is not a guaranteed convergence budget. Latest and periodic checkpoints include optimizer, EMA model, normalization, RNG state, and data provenance. Rerunning resumes; only increasing the update target is permitted in the same run directory. Only load trusted checkpoint files.

**Status:** real-data preparation, replay diagnostic, resumable training, and rollout/video evaluation are implemented. Ten local unit tests pass, including exact CPU interrupted-versus-uninterrupted training. Real dataset/simulator execution and fresh-Colab installation have NOT been validated here: local network access cannot resolve the download host and simulator dependencies are unavailable. No robot-training results are claimed.

The compatibility check deliberately stops training on replay divergence. Its report records raw/corrected reset poses, position/angle errors, and trajectory traces. Diagnose failures before changing thresholds. Coverage is recorded separately from scaled simulator reward.

## Local commands

```bash
pip install -e '.[simulation]'
python -m unittest discover -s tests -v
python -m robot_flywheel.smoke
```

The smoke command remains a synthetic execution check, not a performance benchmark.

## Dataset and scope

The current LeRobot export lacks full object pose. Preparation downloads the original Diffusion Policy archive from https://diffusion-policy.cs.columbia.edu/data/training/pusht.zip, records its checksum, safely extracts it, and reads full state/action arrays. The computed checksum is provenance, not an upstream signature. Dependencies are provisional until a fresh Colab run is verified.

Random/diversity/targeted selector primitives are implemented. Full three-round orchestration, multi-seed statistics, and final test evaluation remain subsequent work after baseline competence. See SPEC.md and REVIEW.md.
