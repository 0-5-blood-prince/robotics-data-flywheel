# Success-guided acquisition v2

Open `notebooks/success_guided.ipynb` in a fresh Colab runtime. Existing Drive run defaults to `exploratory_ddpm_v1_seed11`; new output defaults to `success_guided_v3_seed11`. No all-arms calls are needed. The original directory is read-only to the extension. The notebook setup fetches code, so use a fresh runtime to avoid stale Python imports. Pin the printed commit when resuming.

## Method

Inspired by [Demo-SCORE](https://arxiv.org/html/2503.03707v2), train a small classifier on states from successful versus failed policy rollouts. Use saved continued-training round 2 (40k updates) for training rollouts and round 3 (50k updates) for validation rollouts, with disjoint starts. Default: 200 rollouts per checkpoint starting at 300000. This is separate from original diagnostic, development, and final evaluation starts.

Features: four positions plus sine/cosine of object angle. Standardize using training states only. Architecture: 6→8→8→1, ReLU and dropout 0.2. AdamW with lr .001 and weight decay .01; 1,000 full-batch updates, validate every 25. Binary cross entropy averages over states within each trajectory then over trajectories. Outcomes are labels only; coverage and time are never model inputs. Select the classifier checkpoint with lowest validation loss. Require five success and five failure trajectories in both splits, and lower validation loss than a constant predictor using training success prevalence. These checks do not establish useful acquisition or independent generalization.

Rank each candidate expert demonstration by mean classifier probability over its states. Add the top 20 unacquired episodes in each of three rounds, with episode ID tie-breaking. Keep ranking fixed, retain all prior episodes, and inherit baseline policy weights, optimizer, EMA, normalization, and RNG. Train another 10k updates per round with original DDPM model settings. Scores are heuristic; they are not calibrated estimates of demonstration utility.

This differs from the paper: acquisition rather than filtering; two continued-training checkpoints; fixed ranking across rounds; no successful policy rollouts added to expert training. No classifier uses final-evaluation trajectories. No claim of paper reproduction or guaranteed performance gain.

## Artifacts and comparison

Frozen protocol, classifier contract/checkpoint, training/validation curves and class counts, full expert ranking, per-round selected IDs and scores, checkpoint hashes, policies, per-start evaluations, new combined CSV/JSON, paired bootstrap intervals, and curves. Hashes and frozen contracts reject mismatched resumes. Existing controls and their final summaries are read, not rerun. The extension preserves the old files.

Defaults: 400 classifier rollouts + 30k additional policy updates + 600 evaluation rollouts. Resume skips completed classifier fitting and policy training and reuses cached final rollouts. Changing classifier settings requires a new output directory. If either class is too sparse or validation does not beat the constant predictor, policy training stops; inspect diagnostics rather than presenting a failed gate as successful learning.

The original final suite is now exploratory because its outcomes informed this redesign. Fresh evaluation starts and independent training seeds are required for confirmation. A classifier may merely recognize easy or near-goal states. Inspect high/low-ranked demonstrations and report random-controlled success, coverage, and transition budgets, including negative results.

## Validation

CPU tests cover deterministic fitting on a synthetic outcome signal, insufficient class rejection, pre-action-only feature ingestion, original baseline branching, three acquisition rounds, completed-stage resume without training, unchanged original artifacts, and appended comparisons. Synthetic rollout fixtures test orchestration, not robot learning. Separate real-data CPU simulation checks verify replay, rollout feature ingestion and the insufficient-success gate. Full GPU performance remains unmeasured for this new method.

The original 10k checkpoint produced 0/100 successes and the 20k checkpoint 4/100, insufficient for classifier fitting. Version 3 uses later checkpoints but verifies they contain exactly the original baseline episode IDs. The acquired policy still branches from the original 20k baseline. Later teacher checkpoints add computation to the selection method; equal policy-training budgets do not imply equal total compute. The new output directory preserves earlier failed-gate diagnostics.
