#!/usr/bin/env bash
set -euo pipefail

# Standard CLI mode (threshold)
deal --file "../data/traj.xyz" --threshold 0.5 --mask false
python -m deal.chemiscope_cli --trajectory "deal_selected.xyz" --output "deal_cli_chemiscope.json.gz"

test -s "deal_cli_chemiscope.json.gz"

# Incremental CLI mode (max-selected) must work without a deal_mask array.
rm -f deal_selected.xyz
deal --file "../data/traj.xyz" --max-selected 5

# Incremental mode with save_full_trajectory="final" must re-evaluate every
# input frame with the final GP, not only the frames left in the last iteration.
rm -f deal_incremental_trajectory_uncertainty.xyz
deal -c incremental_full_traj.yaml
python - <<'PY'
from ase.io import read

n_input = len(read("../data/traj.xyz", index=":"))
frames = read("deal_incremental_trajectory_uncertainty.xyz", index=":")
assert len(frames) == n_input, f"expected {n_input} frames, got {len(frames)}"
assert all("atomic_uncertainty" in f.arrays for f in frames)
PY

# With save_full_trajectory=true, on-the-fly frames are appended across
# iterations instead of being overwritten by the last one.
rm -f deal_incremental_trajectory_uncertainty.xyz
sed 's/save_full_trajectory: final/save_full_trajectory: true/' incremental_full_traj.yaml > incremental_full_traj_onthefly.yaml
deal -c incremental_full_traj_onthefly.yaml
rm -f incremental_full_traj_onthefly.yaml
python - <<'PY'
from ase.io import read

n_input = len(read("../data/traj.xyz", index=":"))
frames = read("deal_incremental_trajectory_uncertainty.xyz", index=":")
assert len(frames) > n_input, f"expected more than {n_input} frames, got {len(frames)}"
assert len({f.info["threshold"] for f in frames}) > 1
PY

# deal-evaluate with the saved GP must reproduce save_full_trajectory="final".
rm -f deal_evaluate_*
sed -e 's/save_full_trajectory: final/save_full_trajectory: final\n  save_gp: true/' -e 's/output_prefix: deal_incremental/output_prefix: deal_evaluate/' incremental_full_traj.yaml > deal_evaluate.yaml
deal -c deal_evaluate.yaml
python -m deal.evaluate_cli -c deal_evaluate.yaml -m deal_evaluate_sgp.json -o deal_evaluate_out.xyz
rm -f deal_evaluate.yaml
python - <<'PY'
import numpy as np
from ase.io import read

ref = {a.info["original_frame"]: a for a in read("deal_evaluate_trajectory_uncertainty.xyz", index=":")}
out = read("deal_evaluate_out.xyz", index=":")
assert len(out) == len(ref)
for atoms in out:
    np.testing.assert_allclose(
        atoms.arrays["atomic_uncertainty"],
        ref[atoms.info["original_frame"]].arrays["atomic_uncertainty"],
        atol=1e-9,
    )
PY
