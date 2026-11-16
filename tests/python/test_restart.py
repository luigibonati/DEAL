import os
import tempfile

import numpy as np
from ase.io import read

from deal import DataConfig, SGPConfig, DEAL
from deal.cli import _run_incremental_cli
from deal.restart import read_restart


def _selection(prefix):
    return [
        (
            int(atoms.info["original_frame"]),
            float(atoms.info["threshold"]),
            frozenset(np.atleast_1d(atoms.info["target_atoms"]).tolist()),
        )
        for atoms in read(f"{prefix}_selected.xyz", index=":")
    ]


def _run(prefix, stop_at=(), resume=None):
    """Incremental run that pretends wall time is reached at the given frames."""
    frames_seen = [0]

    def wall_time_reached(self, now):
        frames_seen[0] += 1
        return frames_seen[0] in stop_at

    original = DEAL._wall_time_reached
    DEAL._wall_time_reached = wall_time_reached
    try:
        deal_dict = {
            "max_selected": 6,
            "max_iterations": 4,
            "mask": False,
            "output_prefix": prefix,
            "restart": True,
            "wall_time": 1e6,
            "save_full_trajectory": "final",
        }
        config = {"data": {}, "deal": deal_dict, "sgp": {"cutoff": 4.5}}
        _run_incremental_cli(
            DataConfig(files="../data/traj.xyz", shuffle=True, seed=1),
            deal_dict,
            SGPConfig(cutoff=4.5),
            config=config,
            resume=resume,
        )
    finally:
        DEAL._wall_time_reached = original


def main():
    with tempfile.TemporaryDirectory() as tmp:
        reference = os.path.join(tmp, "reference")
        _run(reference)
        assert read_restart(f"{reference}.restart")["finished"]

        # Stop in the middle of an iteration, then again after resuming.
        interrupted = os.path.join(tmp, "interrupted")
        _run(interrupted, stop_at=(4,))
        state = read_restart(f"{interrupted}.restart")
        assert not state["finished"] and not state["run"]["completed"]
        _run(interrupted, stop_at=(5,), resume=state)
        state = read_restart(f"{interrupted}.restart")
        assert not state["finished"]
        _run(interrupted, resume=state)
        assert read_restart(f"{interrupted}.restart")["finished"]

        assert _selection(interrupted) == _selection(reference)
        final_ref = read(f"{reference}_trajectory_uncertainty.xyz", index=":")
        final_int = read(f"{interrupted}_trajectory_uncertainty.xyz", index=":")
        assert len(final_int) == len(final_ref)
        for a, b in zip(final_ref, final_int):
            np.testing.assert_allclose(
                a.arrays["atomic_uncertainty"], b.arrays["atomic_uncertainty"], atol=1e-6
            )


if __name__ == "__main__":
    main()
