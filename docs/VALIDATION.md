# Validation status

CPU validation on macOS/Python 3.12, 2026-11-06:

- 14 focused tests passed, including exact interrupted/uninterrupted CPU training, retained normalization, acquired-episode budgets, three-round branching, completed-stage resume, disjoint seed ranges, and paired bootstrap.
- Original public PushT archive loaded: 206 episodes, 25,650 transitions.
- Archive SHA-256: `63d52a114a3f010861f0181309d165b7d69133ccae426ece2fc94caed147bdf9` (locally observed provenance).
- First three expert episodes passed corrected-reset replay; position RMSE below 0.00003 simulation units.
- Actual simulator quick study completed: one baseline plus four arms times three rounds, 13 evaluated checkpoints, two evaluation starts each.
- CSV, comparison JSON, success plots, and a rollout MP4 were generated. Completed-stage resume passed without retraining, and the MP4 decoded successfully.

Quick mode uses intentionally insufficient training and short rollouts. Its zero-success results are execution checks, not evidence for or against acquisition quality.

Reproduce with `python scripts/validate_cpu.py` after `pip install -e '.[simulation]'`. This also verifies cached resume and video decoding. Use a new output directory when changing frozen code/configuration.

Not established: convergence, full five-seed results, GPU/Colab throughput, or a fresh Python 3.13 Colab execution of every cell. Simulator dependencies are pinned for the legacy collision API. A development pilot is required before treating the full defaults as an adequate training budget.
