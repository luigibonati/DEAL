from types import SimpleNamespace
from tempfile import TemporaryDirectory

import numpy as np
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import read

from deal.core import DEAL


def test_selected_xyz_preserves_original_energy_and_forces():
    """Selected extxyz frames retain calculator-backed reference labels."""
    forces = np.array([[0.1, -0.2, 0.3], [-0.4, 0.5, -0.6]])
    frame = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]])
    frame.calc = SinglePointCalculator(frame, energy=-1.25, forces=forces)

    with TemporaryDirectory() as tmpdir:
        selector = object.__new__(DEAL)
        selector.deal_cfg = SimpleNamespace(
            output_prefix=f"{tmpdir}/deal", threshold=0.1
        )
        selector.timers = {"io_write": 0.0}
        selector.dft_count = 0
        selector.selected_frames = []

        selector._store_selected_frame(step=7, ase_frame=frame, target_atoms=[1])

        selected = read(f"{tmpdir}/deal_selected.xyz", format="extxyz")
        assert selected.get_potential_energy() == -1.25
        np.testing.assert_allclose(selected.get_forces(), forces)


if __name__ == "__main__":
    test_selected_xyz_preserves_original_energy_and_forces()
