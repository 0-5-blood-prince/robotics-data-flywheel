# Review: fast failure detection

Reviewed complete project through 665662f against SPEC.md and documented resume behavior. Separate read-only standards and spec reviews were performed.

## Standards

Fixed: standalone training resume now hashes actual selected arrays, package source, and NumPy/PyTorch versions; notebook test failures now raise; evaluation requires trace files and requested videos before using cached rows; video writer closes before completion JSON; basic model/optimizer validation occurs before training; frozen source provenance derives from package location and contents.

Residual: cache checks verify trace readability and video existence, not full video integrity. Source changes intentionally invalidate training resume and require new output directories. Existing checkpoint weights remain usable for development evaluation. Full environment recreation is not yet a lockfile-based guarantee.

## Spec

Fixed: full study rejects development success below a configurable 20% gate (quick mode exempt). The notebook records the observed best pilot success, rather than silently proceeding with weak pilot output. Failure features now use post-action states aligned with coverage. Evaluation identity no longer depends on the requested seed list, allowing incremental expansion.

Residual: complete acquisition/compute-efficiency reporting (curve area, selection overhead, sampled-transition accounting) remains incomplete. Full comparison against an independently verified upstream model remains pending. The learning gate is a development heuristic, not statistical proof of competence.

## Faster development

Default 5,000-update runs are screened before any larger budgets. A zero-success screen with mean maximum coverage below 0.6 stops extending that dataset's run. Users can inspect traces and adjust development settings; no checkpoints are deleted.

Development uses 10-step deterministic DDIM after seeded initial noise, reducing forward calls from 100 to 10 per action plan. This changes inference behavior, so it is labeled in results/cache identity and has an explicit optional confirmation cell using original DDPM on 20 development starts. The full study still defaults to original DDPM. Do not compare DDIM and DDPM numbers as though they used the same policy configuration.

Measured CPU width-64 plan latency: 16.76 ms DDPM versus 1.63 ms DDIM (10 repeated samples, one thread). This is an inference microbenchmark, not a GPU or whole-study speedup claim.

Validation: 16 tests; actual simulator incremental cache expansion and aligned post-state traces; notebook syntax. Earlier full CPU validation applies to the pre-review revision; the full GPU study remains unmeasured.

## Resume guidance

New source/data contracts and evaluation schema deliberately reject old run directories rather than mix results. Preserve all old outputs and use new directory names for new runs. To resume an old full study exactly, pin its original commit and dependency versions. Existing checkpoints can be supplied to the development notebook for fresh evaluation without retraining.
