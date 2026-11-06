# Failure-Guided Robotics Data Flywheel

Status: implementation specification; experiment design agreed with user on 2026-11-06.

## Objective

Build a Colab-first, state-based PushT experiment connecting rollout failure analysis, geometric demonstration acquisition, dataset versioning, diffusion-policy training, and reproducible evaluation. Test whether failure-guided selection improves learning relative to random and diversity-based selection under matched acquisition and optimization budgets.

Deliver a competent policy and an honest controlled experiment. Positive gains are a hypothesis, not an acceptance requirement. This is simulated 2D manipulation and acquisition from an existing expert demonstration pool, not real-robot deployment, newly collected corrections, or a multimodal perception model.

## Deliverables

- One executable Colab notebook backed by reusable Python modules.
- Pinned dependencies, dataset revision, explicit configuration, and documented runtime setup.
- Checkpoints, immutable selection manifests, rollout-level metrics, and resumable experiment state.
- Saved example notebook outputs, plots, and paired rollout videos.
- README explaining the question, methods, measured results, limitations, and reproduction commands.
- Machine-readable result tables and a generated factual summary suitable for subsequent resume wording.

Publishing and paid resource provisioning are separate actions; the project must not require credentials or auto-publish artifacts. Colab GPU use is the intended execution path. No paid resource is provisioned by this specification.

## Primary hypothesis and outcomes

Primary contrast: targeted acquisition versus random acquisition after round three, measured by paired task-success difference in percentage points on held-out simulator initializations. Diversity acquisition is a stronger secondary comparator.

Secondary outcomes: success at every round, mean maximum coverage, area under the success-versus-acquired-demonstrations curve, transition counts, training wall time, and selection overhead. Report original-baseline and no-new-data controls. Never replace task success with training loss as the main evidence.

Success follows the installed PushT environment's documented criterion, expected to be at least 95% target coverage. Verify reward and termination behavior in the pinned implementation. Respect episode truncation and record it separately from success.

## Architecture

```text
Expert demonstrations -> verified adapter -> initial set + reserved pool
                                             |
                                      baseline training
                                             |
                                 fork experiment branches
                                             |
                diagnostic rollouts -> failure segments -> selection
                                             |
                           versioned cumulative dataset -> training
                                             |
                                 repeat for three rounds
                                             |
                      frozen checkpoints -> final evaluation -> report
```

Suggested package: `src/robot_flywheel/` with `data.py`, `policy.py`, `train.py`, `rollout.py`, `features.py`, `selection.py`, `experiment.py`, `artifacts.py`, `report.py`; configuration files in `configs/`; entry notebook in `notebooks/`; focused tests in `tests/`.

## Stage 0: compatibility and learning gates

Review finding: the current official LeRobot PushT export exposes only a two-dimensional agent state. Use the original Diffusion Policy full-state Zarr archive as the adapter target, with gym-pusht as the simulator candidate. Archive inspection and replay validation remain required before real training. Pin verified versions rather than blindly combining current library code with historical checkpoint formats.

Before model work, inspect an actual episode and establish:

1. Availability of object pose, agent position, expert actions, timestamps/cadence, and episode boundaries. Observation field names and shapes must be discovered, not assumed.
2. Coordinate units, orientation convention, action meaning, control frequency, simulator initial-state restoration, and dataset/environment consistency.
3. A bounded expert-action replay check from a known compatible initial state. Explain residual mismatch; do not silently treat incompatible demonstrations as valid supervision.
4. Deterministic reset and reproducible rollout recording.
5. A minibatch overfit check, followed by pilot training that produces meaningful rollout competence.

If the proposed dataset lacks required state, inspect its authoritative upstream state archive and document conversion. Do not infer unavailable object poses from arbitrary metadata. Dataset incompatibility is a blocking issue for dependent experiments.

## Policy

State-conditioned action-sequence diffusion policy with a compact temporal 1D U-Net. No image encoder. Observation features are agent x/y, object x/y, and sine/cosine of object orientation. Two observations, 16 predicted actions, and an eight-action execution prefix are initial pilot settings, not presumed optimal values.

Train by adding scheduled Gaussian noise to expert action sequences and predicting that noise, conditioned on observation history and diffusion timestep. Replan after executing a short action prefix. Keep observation/action alignment explicit and tested. Sequence construction must never cross episode boundaries; pad boundary observations deterministically and mask padded targets in the loss.

Prefer an established diffusion implementation where state conditioning is supported. Verify its conditioning path and tensor shapes. Record network configuration, parameter count, noise schedule, training/inference diffusion steps, optimizer, learning rate, batch size, gradient clipping, and action bounds.

Compute and freeze normalization from each seed's initial training subset so later acquisition arms share the same coordinates and checkpoint semantics. Use a documented physical scaling for geometric retrieval features, independent of final evaluation data. Do not change normalization silently after acquisition.

Inference stochasticity must be seeded and recorded. Use a fixed inference procedure across arms. A pretrained policy may serve as an optional sanity reference, but must not be the experimental baseline if it has already trained on the acquisition pool.

## Development pilot and freeze

Use development simulator seeds only. Run a small, declared pilot grid for initial dataset size, training duration, learning rate, model width, and execution horizon. Use successive screening rather than a large Cartesian grid. Log every tried configuration and decision.

Train an all-pool reference to estimate achievable performance; it is not a formal upper bound. Prefer a starting data regime with meaningful baseline competence and room to improve, selected from the baseline data-scaling curve, not targeted-selection wins.

After the pilot, write `frozen_protocol.json` containing dataset revision, split algorithm, initial set size N0, per-round acquisition size K, three rounds, model configuration, training update budgets, seed schedules, selector constants, and evaluation procedure. Require N0 + 3K <= total eligible episodes and 3K <= the reserved acquisition pool size. Update counts and N0/K are derived from measured learning curves and dataset availability, not invented in advance.

Method refinement is allowed on development results. Final evaluation remains unopened until the protocol is frozen. If final results motivate a redesign, label that analysis exploratory and use a new untouched evaluation suite for a new confirmatory experiment.

## Experimental arms

For each of five independent training seeds, create one shared initial expert subset and one reserved pool. Train a baseline, then fork its exact weights and optimizer state into four branches:

| Arm | Data each round | Training each round |
| --- | --- | --- |
| Continued training | Initial subset only | U optimizer updates |
| Random | Add K unseen random expert episodes | U updates |
| Diversity | Add K unseen geometry-diverse expert episodes | U updates |
| Failure-guided + diversity | Add K unseen failure-relevant, diverse expert episodes | U updates |

Retain all acquired episodes in subsequent rounds. Use equal batch sizes, update counts, architecture, and schedules across updated arms. Sampling should use the same documented episode/window policy; retain old data rather than exclusively replay new selections. Fork optimizer state identically; define learning-rate progression globally across rounds.

Acquisition budget is K complete episodes per round. Episode lengths differ, so equal episode count does not mean equal transition count. Report cumulative unique transitions and sampled training transitions. If selected lengths differ substantially, add a transition-budget sensitivity analysis before claiming superior data efficiency per transition.

Baseline is a round-zero checkpoint, not a branch that receives extra training. An optional all-pool reference is displayed separately with its own larger data budget.

## Geometric retrieval and selection

Represent goal-relative object position, object orientation as sine/cosine, agent-to-object offset, and goal distance. Validate the goal pose and any symmetries against the actual simulator geometry. Standardize feature-group scales to prevent position units dominating orientation. Document weights and fix them before final runs.

Extract diagnostic windows from unsuccessful episodes, especially sustained low coverage progress. Choose window duration in physical time and convert using verified cadence. Cap windows per rollout and cluster/subsample them to avoid one long stall dominating selection. Do not label every pause as a failure.

For each reserved expert episode, derive bounded representative state features. Retrieve states geometrically close to diagnostic windows and retain the episode provenance. Selection must score relevant expert states; it does not treat the failed policy's actions as expert labels.

Proposed targeted objective: greedily choose episodes that improve coverage of diagnostic failure features, penalizing redundant selections. Implement using cached distances with explicit tie-breaking. Record per-selection marginal coverage gain, nearest failure examples, distance, and novelty. Freeze exact distance aggregation and weights after development.

Diversity control uses the same feature representation and greedy coverage machinery but covers the expert pool distribution instead of diagnostic failures. Random samples without replacement. Selected episode IDs cannot re-enter an arm's reserved pool.

Measure failure-to-pool nearest distances and report unsupported failures. When diagnostic rollouts have no failures, use a prespecified diversity fallback; record its activation. Do not keep inventing failure labels to force a targeted advantage. Thresholds for unsupported states and empty-window behavior must be configured and tested.

Each adaptive branch diagnoses its own current policy using the same scheduled diagnostic initializations. Use matched diagnostic counts across rounds and branches; selection can use only permitted diagnostic outcomes. No final evaluation information reaches selectors.

## Evaluation and statistics

Three disjoint simulator-seed namespaces: development, acquisition diagnostics, and final evaluation. Record actual initial states as well as RNG seeds. Where reset generation can duplicate states, audit overlap beyond seed identity.

Target five independent training seeds and 200 held-out initializations per trained policy. Pair initializations across arms, rounds, and training seeds. Record inference RNG seeds. Run all saved round checkpoints only after protocol freeze; save resumable rollout results to avoid rerunning completed evaluations.

Report per-training-seed scores and the aggregate. Primary uncertainty should account for shared evaluation starts and training-seed variation: paired crossed bootstrap independently resampling training-seed IDs and shared initial-state IDs, evaluating their Cartesian product and preserving arm pairing; explain the limited precision of five training seeds. Do not treat all rollouts as independent model-training replicates. Report 95% intervals, raw counts, and percentage-point differences without promising significance.

One prespecified primary contrast avoids selecting the best-looking round or metric. Other comparisons are descriptive unless separately declared. Show all three rounds and all seeds, including regressions. Show representative paired videos selected by a declared rule, plus both successes and failures; do not present selected videos as aggregate evidence.

## Artifact contracts

- Episode index: source revision, episode ID, length, cadence, state/action schema, checksum.
- Dataset manifest: run/seed/arm/round, parent manifest hash, ordered episode IDs, acquisition reasons, selector configuration, unique transition count.
- Checkpoint: weights, optimizer/scheduler/RNG states, normalization, model config, dataset manifest hash, update count.
- Rollout: policy checkpoint hash, evaluation namespace, initial state/seed, inference seed, step, timestamp, state, action, coverage/reward, termination/truncation flags; optional video path.
- Selection log: candidate/selected episode IDs, failure-window references, distances, novelty, marginal score, fallback reason.
- Results: per-rollout records, per-training-seed summaries, paired contrasts, uncertainty method, runtime/hardware metadata.

Persist metadata/indexes as Parquet where appropriate, configurations/manifests as JSON, checkpoints in the chosen framework format, videos as MP4. Use atomic checkpoint writes and resumable run IDs; never silently mix artifacts from changed configurations.

## Notebook narrative

1. Question, scope, hypothesis, and expected outputs.
2. Runtime setup, GPU detection, version/configuration display, artifact directory selection.
3. Dataset inspection and action/state compatibility evidence.
4. Policy architecture explanation and pilot/baseline curves.
5. Baseline rollout viewer: frame, trajectory, coverage, action trace.
6. Failure windows and nearest expert examples with selection explanations.
7. Three-round experiment progress and dataset manifests.
8. Training/coverage/success curves and paired before/after rollout videos.
9. Final comparison table with uncertainty, controls, compute/data budgets, and negative outcomes.
10. Limitations and export/download links.

Provide `smoke`, `pilot`, and `full` configurations. Smoke mode verifies execution and artifacts, not scientific performance. Full mode can resume across Colab sessions and write to a user-selected persistent location. Saved outputs permit review without running training. Do not require a cloud account beyond the user's Colab session or an experiment-tracking service.

## Verification and acceptance

Meaningful checks cover episode-boundary handling, state/action alignment, angle wraparound, reset/replay compatibility, split disjointness, acquisition uniqueness, fixed budgets, checkpoint branching/resume, normalization consistency, and metric agreement with simulator outcomes. Include small deterministic selector fixtures and artifact provenance round trips.

Implementation acceptance: a fresh Colab environment completes smoke mode; pilot can train and evaluate a policy; full configuration implements all agreed arms/rounds; resume is demonstrated; generated reports trace each result to its data and checkpoint.

Scientific acceptance: frozen protocol, complete planned comparisons or explicit incomplete-run disclosure, raw metrics, five-seed variability where completed, honest confidence intervals, and no final-test-driven tuning. Improved targeted performance is not required for experiment validity. A policy that never learns requires diagnosis before drawing conclusions about acquisition quality.

## Execution milestones

M1: Dataset/environment compatibility, one rendered expert replay, smoke artifact contracts.
M2: State-conditioned diffusion baseline, training-time measurements, pilot data-scaling curve.
M3: Freeze protocol and implement auditable random/diversity/targeted selectors.
M4: Complete a full three-round development run with matched controls; validate resume and reporting.
M5: Execute the frozen multi-seed experiment and final evaluation; generate aggregate figures and paired videos.
M6: Fresh-Colab verification, saved notebook output, README, limitations, and measured application-ready project summary.

## Risks and responses

- Simulator or library incompatibility: resolve at M1, pin versions, record conversions.
- Weak baseline: inspect alignment, replay, normalization, temporal horizons, and training convergence before acquisition claims.
- Pool cannot cover failure states: quantify the coverage gap; fresh expert corrections are a future extension.
- Targeted overspecialization: retain cumulative data and include diversity; show broad evaluation regressions.
- Extra compute explains gains: compare continued training and equal-update acquisition controls.
- Data budget imbalance: report lengths and transition counts; qualify episode-level efficiency claims.
- Expensive evaluation or Colab interruption: cache rollouts, checkpoint atomically, resume; disclose reduced runs rather than silently changing the protocol.

## Future extensions

New corrective demonstrations, image-conditioned policies, real-robot episode adapters, richer annotation UI, query/storage benchmarks, and distributed orchestration. These are outside the initial experiment.

## Primary references

- PushT environment and observation/action semantics: https://github.com/huggingface/gym-pusht
- LeRobot training example: https://github.com/huggingface/lerobot/blob/main/examples/training/train_policy.py
- LeRobot rollout evaluation: https://github.com/huggingface/lerobot/blob/main/src/lerobot/scripts/lerobot_eval.py
- Official PushT diffusion checkpoint/model card (reference, not experimental initialization): https://huggingface.co/lerobot/diffusion_pusht

These references informed the design; exact installed API and dataset schema remain M1 verification requirements.

## Implementation review addendum

See REVIEW.md for compatibility findings and corrections. The first implementation uses a custom compact conditional temporal U-Net; establishing learning competence is mandatory before acquisition experiments. The current notebook is an explicit synthetic smoke preview, not the completed experiment. Full final evaluation entails 13,000 rollouts before diagnostic and pilot overhead.
