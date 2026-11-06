# Spec review and implementation decisions

1. **Blocking dataset mismatch:** current official LeRobot metadata has observation.state shape [2], not full pose. Original Diffusion Policy image dataset source documents raw state as agent position plus object pose. Implement an adapter for the original Zarr schema; actual archive/replay validation remains outstanding.
2. **Reset semantics:** gym-pusht warns reset_to_state may shift the block due to legacy behavior. Record returned initial state, verify replay, and never assume input pose equals output pose.
3. **Metric distinction:** inspect raw geometric coverage separately from possibly normalized reward. Never label max reward as max overlap without checking implementation.
4. **Matched budget:** episode counts do not equal transitions. Report both; claims are episode-budget comparisons unless transition sensitivity supports stronger statements.
5. **Statistics:** initial states are shared across training seeds, so bootstrap is crossed: resample seed IDs and initial-state IDs independently, use their Cartesian product, and preserve arm pairing. Do not resample different initial states independently per training seed.
6. **Protocol ambiguity:** N0 + 3K <= total eligible episodes, not the reserved-pool size; separately require 3K <= reserved-pool size.
7. **Resource accounting:** complete primary evaluation requires 5 * (1 + 4*3) * 200 = 13,000 rollouts, excluding diagnostic/development runs. Cache results and measure rollout cost before full execution.
8. **Implementation scope:** lightweight custom conditional U-Net avoids coupling the state-only project to full LeRobot hardware/video dependencies. This departure from preferring an established policy implementation increases model-validation responsibility. Unit/smoke checks establish execution, not learning competence.

Sources checked 2026-11-06:
- https://huggingface.co/datasets/lerobot/pusht/raw/main/meta/info.json
- https://raw.githubusercontent.com/huggingface/gym-pusht/main/gym_pusht/envs/pusht.py
- https://raw.githubusercontent.com/real-stanford/diffusion_policy/main/diffusion_policy/dataset/pusht_image_dataset.py
- https://diffusion-policy.cs.columbia.edu/data/training/
