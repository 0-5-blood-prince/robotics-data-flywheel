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

Runs only the new `success_guided` arm from your existing seed-11 baseline. A small success/failure classifier ranks demonstrations, inspired by [Demo-SCORE](https://arxiv.org/html/2503.03707v2). This is an acquisition adaptation, not a reproduction. Defaults use 400 separate classifier rollouts from saved continued-training checkpoints, three 10k-update acquisition stages, and 600 evaluation rollouts. The original controls are read from Drive and appended into new combined reports without retraining or reevaluation. See [method and validation](docs/SUCCESS_GUIDED.md).

## Diagnose selection using saved evidence

[Open the evidence audit in Colab](https://colab.research.google.com/github/0-5-blood-prince/robotics-data-flywheel/blob/main/notebooks/evidence_audit.ipynb)

CPU-only analysis of the existing original and success-guided runs: trajectory-level ROC-AUC/AP with bootstrap intervals, prefix-versus-tail score comparisons, acquisition geometry and novelty, paired outcomes, coverage loss, and deterministic example traces. It reads saved artifacts without training, running policy inference, or simulating new episodes. Writes CSVs, figures, and `evidence_report.md` to `success_guided_v3_evidence`. Classifier validation and reused evaluation results are explicitly exploratory. All analysis cells were executed locally against synthetic saved-run artifacts; Drive mounting and your actual artifact results require the Colab run.

## Failure-start acquisition

[Open the failure-start notebook in Colab](https://colab.research.google.com/github/0-5-blood-prince/robotics-data-flywheel/blob/main/notebooks/failure_start.ipynb)

Runs only a new `failure_start` arm from the existing 20k baseline. A [Mulligan-inspired](https://arxiv.org/html/2610.05882v1) fixed-pool adaptation assigns half the acquisition slots to nearest demonstrations at failed starting states and fills the rest with initial-state diversity. Unlike the paper, it does not collect new expert supervision at exact failed starts. It uses no classifier.

Defaults to one round (20 diagnostic rollouts, 20 acquired episodes, 10k updates, 200 evaluations), comparing with the same round of existing controls. Set `ROUNDS_TO_RUN=3` later to extend without repeating completed training. New outputs go to `failure_start_v1_seed11`; original and success-guided artifacts are retained. Severity is approximated by low maximum coverage; matches use object position, relative agent position, and periodic orientation. Distances and selection reasons are logged. This is exploratory and does not establish performance gains.

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
