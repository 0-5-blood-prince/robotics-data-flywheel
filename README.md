# Robotics Data Flywheel

A Colab-first experiment testing whether failure-guided expert-data acquisition improves state-based PushT policy learning over random and diversity controls.

[Open the complete notebook in Colab](https://colab.research.google.com/github/0-5-blood-prince/robotics-data-flywheel/blob/main/notebooks/flywheel.ipynb)

## Development diagnostic notebook

[Open development experiment in Colab](https://colab.research.google.com/github/0-5-blood-prince/robotics-data-flywheel/blob/main/notebooks/development.ipynb)

Development defaults now use five starts and 10-step DDIM screening, with original-DDPM confirmation available. See [review and measured speed](docs/REVIEW_FAST_FEEDBACK.md). Use new run directories after this update; old checkpoints remain evaluable.

Use this before another full study: paired checkpoint evaluation on fixed development starts; execution prefixes 2/4/8; 50-episode and all-data training references; coverage/action diagnostics; synchronized side-by-side videos. Drive storage is enabled by default. Quick mode validates execution; development mode measures learning. Existing checkpoints can be supplied to skip training. This does not yet incorporate an independently verified upstream policy implementation.

The original flywheel notebook and all experiment code remain available unchanged.

## One seed first

[Open one-seed exploratory study](https://colab.research.google.com/github/0-5-blood-prince/robotics-data-flywheel/blob/main/notebooks/single_seed.ipynb)

Runs all four arms and three rounds for seed 11 using DDPM/execute 8. Uses the recorded development evidence instead of retraining the pilot. Change `TRAINING_SEED` to run later repeats in separate directories without overwriting previous results. One seed means 13 training stages and 2,600 evaluation rollouts. Drive persistence is enabled. Keep settings and code version fixed across repeats; results are exploratory until replicated.

## Add success-guided selection to an existing run

[Open success-guided extension in Colab](https://colab.research.google.com/github/0-5-blood-prince/robotics-data-flywheel/blob/main/notebooks/success_guided.ipynb)

Runs only the new `success_guided` arm from your existing seed-11 baseline. A small success/failure classifier ranks demonstrations, inspired by [Demo-SCORE](https://arxiv.org/html/2503.03707v2). This is an acquisition adaptation, not a reproduction. Defaults use 200 separate classifier rollouts, three 10k-update acquisition stages, and 600 evaluation rollouts. The original controls are read from Drive and appended into new combined reports without retraining or reevaluation. See [method and validation](docs/SUCCESS_GUIDED.md).

## Notebook workflow

1. Setup fetches this repository, checks imports, and resets/steps the simulator.
2. Download real demonstrations and verify expert replay.
3. Choose `quick` (CPU execution validation) or `full` (GPU experiment).
4. Full mode runs a development data-scaling pilot; review the resulting learning regime before freezing settings.
5. Freeze data/model/budget/seed configuration and source provenance.
6. Train baselines and three rounds of continued-training, random, diversity, and failure-guided branches.
7. Inspect acquired examples, failure features, and unsupported-state distances.
8. Evaluate saved checkpoints on paired held-out initializations.
9. Generate curves, paired uncertainty intervals, CSV results, and illustrative videos.

Full defaults: five training seeds, 50 initial demonstrations, 20 acquisitions per round, 20,000 baseline updates, 10,000 updates per round, and 200 final initializations per checkpoint. These are provisional development settings; they do not guarantee policy convergence. Full evaluation alone comprises 13,000 rollouts.

Enable Google Drive persistence for long runs. Pin `REVISION` to the printed code commit and retain dependency versions when resuming a frozen experiment. Changes to code/config/data require a new study directory. Training checkpoints include optimizer, EMA, normalization, RNG state, and dataset provenance. Only load trusted checkpoint files.

## Validation

The real-data quick study completed on CPU through all arms and rounds, rollout evaluation, report generation, and video rendering. Fourteen focused tests pass. See [validation details](docs/VALIDATION.md).

**No performance improvement is claimed.** Quick mode deliberately undertrains policies. Full GPU convergence and multi-seed results remain to be measured. Local CPU execution does not establish fresh-Colab compatibility for every runtime version.

```bash
pip install -e '.[simulation]'
python -m unittest discover -s tests -v
python scripts/validate_cpu.py
```

The notebook fetches code from GitHub; no ZIP uploads are needed. A personal notebook copy retains its own cells, so reopen the stable link above for notebook changes. Setup explicitly activates the editable source path in the running kernel and checks native simulator API compatibility.

## Scientific scope

Original full-state PushT demonstrations, a compact conditional temporal U-Net diffusion policy, and offline demonstration acquisition guided by simulated failures. It does not collect new expert corrections or demonstrate real-robot/perception performance. Acquisition budgets match episode counts; transition counts are reported separately. Final evaluation never feeds selection. See [specification](SPEC.md) and [review](REVIEW.md).
