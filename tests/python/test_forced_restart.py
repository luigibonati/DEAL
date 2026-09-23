from types import SimpleNamespace
from tempfile import TemporaryDirectory

import numpy as np
import pytest
from ase import Atoms

from deal.core import DEAL


class _RecordingModel:
    def __init__(self):
        self.updates = []

    @staticmethod
    def to_model_atoms(atoms):
        return atoms

    def update(self, *, train_atoms, **kwargs):
        self.updates.append(list(train_atoms))


def _selector_for(frame, output_prefix):
    selector = object.__new__(DEAL)
    selector.data_cfg = SimpleNamespace(images=[frame])
    selector.deal_cfg = SimpleNamespace(
        mask=False,
        force_only=True,
        train_hyps=False,
        save_gp=False,
        save_full_trajectory=False,
        verbose=False,
        debug=False,
        threshold=100.0,
        min_steps_with_model=999,
        output_prefix=output_prefix,
    )
    selector.sgp_cfg = SimpleNamespace(variance_type="local")
    selector.model = _RecordingModel()
    selector.timers = {
        "start": 0.0,
        "total": 0.0,
        "extract_dft": 0.0,
        "predict": 0.0,
        "update": 0.0,
        "io_write": 0.0,
        "other": 0.0,
        "frames": 0,
    }
    selector.last_dft_step = -(10**9)
    selector.selected_frames = []
    selector.dft_count = 0
    return selector


def test_forced_restart_updates_saved_target_atoms_before_threshold_check():
    frame = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])
    frame.info["deal_force_update"] = True
    frame.info["target_atoms"] = np.array([1], dtype=int)

    with TemporaryDirectory() as tmpdir:
        selector = _selector_for(frame, f"{tmpdir}/deal")
        selector.run()

    assert selector.model.updates == [[1]]
    assert selector.dft_count == 1
    assert selector.selected_frames[0].info["target_atoms"].tolist() == [1]
    assert selector.last_dft_step == 0


def test_forced_restart_requires_saved_target_atoms():
    frame = Atoms("H", positions=[[0, 0, 0]])
    frame.info["deal_force_update"] = True

    with pytest.raises(ValueError, match="no 'target_atoms'"):
        DEAL._forced_target_atoms(frame, candidate_mask=None)
