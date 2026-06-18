from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from ase import Atoms
from ase.io import read, write

from deal import TrajectoryMasker


with TemporaryDirectory() as tmpdir:
    workdir = Path(tmpdir)
    input_file = workdir / "mask_input.xyz"
    output_file = workdir / "mask_output.xyz"

    atoms = Atoms("H4", positions=np.zeros((4, 3)))
    atoms.arrays["force_std_comp_max"] = np.array([0.01, 0.07, 0.03, 0.12])
    write(input_file, atoms, format="extxyz")

    summary = TrajectoryMasker(
        key="force_std_comp_max",
        threshold=0.05,
        mask_key="deal_mask",
    ).run(str(input_file), str(output_file), file_format="extxyz")

    masked = read(output_file, format="extxyz")
    assert summary.n_frames == 1
    assert summary.n_atoms == 4
    assert summary.n_selected_atoms == 2
    assert summary.n_frames_with_selection == 1
    np.testing.assert_array_equal(masked.arrays["deal_mask"], np.array([0, 1, 0, 1]))
    assert masked.info["deal_mask_count"] == 2
    assert masked.info["deal_mask_source"] == "force_std_comp_max"
    assert masked.info["deal_mask_threshold"] == 0.05

matrix_atoms = Atoms("H2", positions=np.zeros((2, 3)))
matrix_atoms.arrays["unc_components"] = np.array([[0.01, 0.02, 0.03], [0.0, 0.2, 0.01]])
masker = TrajectoryMasker(key="unc_components", threshold=0.05)
masker.apply_to_atoms(matrix_atoms)
np.testing.assert_array_equal(matrix_atoms.arrays["deal_mask"], np.array([0, 1]))
