from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import numpy as np
import yaml
from ase import Atoms
from ase.io import read, write

from deal import TrajectoryMasker, write_preprocessed_trajectory
from deal.mask_cli import main as mask_main


with TemporaryDirectory() as tmpdir:
    workdir = Path(tmpdir)
    input_file = workdir / "mask_input.xyz"
    output_file = workdir / "mask_output.xyz"

    atoms = Atoms("H4", positions=np.zeros((4, 3)))
    atoms.arrays["force_std_comp_max"] = np.array([0.01, 0.07, 0.03, 0.12])
    write(input_file, atoms, format="extxyz")

    summary = TrajectoryMasker(
        key="force_std_comp_max",
        mask_threshold=0.05,
        mask_key="deal_mask",
        plot=False,
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
masker = TrajectoryMasker(
    key="unc_components", mask_threshold=0.05, plot=False
)
masker.apply_to_atoms(matrix_atoms)
np.testing.assert_array_equal(matrix_atoms.arrays["deal_mask"], np.array([0, 1]))

trajectory = [
    Atoms("H2", positions=np.zeros((2, 3))),
    Atoms("H2", positions=np.zeros((2, 3))),
]
trajectory[0].arrays["score"] = np.array([0.1, 0.9])
trajectory[1].arrays["score"] = np.array([0.2, 0.3])
summary = TrajectoryMasker(
    key="score", mask_threshold=0.5, plot=False
).apply_to_trajectory(trajectory)
assert summary.n_frames == 2
assert summary.n_atoms == 4
assert summary.n_selected_atoms == 1
assert summary.n_frames_with_selection == 1
np.testing.assert_array_equal(trajectory[0].arrays["deal_mask"], [0, 1])
np.testing.assert_array_equal(trajectory[1].arrays["deal_mask"], [0, 0])

try:
    TrajectoryMasker(
        key="score",
        mask_threshold=0.5,
        mode="between",
        mask_upper_threshold=0.4,
    )
except ValueError as error:
    assert "greater" in str(error)
else:
    raise AssertionError("Invalid preprocessing interval was accepted")

automatic = [
    Atoms("H2", positions=np.zeros((2, 3))) for _ in range(3)
]
automatic[0].arrays["score"] = np.array([0.01, 0.02])
automatic[1].arrays["score"] = np.array([0.02, 0.08])
automatic[2].arrays["score"] = np.array([0.1, 1.0])
with TemporaryDirectory() as tmpdir:
    plot_file = Path(tmpdir) / "selection.png"
    summary = TrajectoryMasker(
        key="score", plot=str(plot_file)
    ).apply_to_trajectory(automatic)
    mean_max = np.mean([0.02, 0.08, 1.0])
    assert np.isclose(summary.lower_threshold, 1.1 * mean_max)
    assert np.isclose(summary.upper_threshold, 4.0 * mean_max)
    np.testing.assert_array_equal(automatic[0].arrays["deal_mask"], [0, 0])
    np.testing.assert_array_equal(automatic[1].arrays["deal_mask"], [0, 0])
    np.testing.assert_array_equal(automatic[2].arrays["deal_mask"], [0, 1])
    assert plot_file.is_file()

with TemporaryDirectory() as tmpdir:
    output = Path(tmpdir) / "cached.xyz"
    assert write_preprocessed_trajectory(trajectory, str(output))
    original = output.read_bytes()
    trajectory[0].positions[0, 0] = 99
    assert not write_preprocessed_trajectory(trajectory, str(output))
    assert output.read_bytes() == original
    assert write_preprocessed_trajectory(trajectory, str(output), overwrite=True)
    assert output.read_bytes() != original

with TemporaryDirectory() as tmpdir:
    workdir = Path(tmpdir)
    source = workdir / "source.xyz"
    output = workdir / "from_yaml.xyz"
    config = workdir / "input.yaml"
    write(source, trajectory, format="extxyz")
    config.write_text(
        yaml.safe_dump(
            {
                "data": {"files": [str(source)], "format": "extxyz"},
                "preprocessing": {
                    "key": "score",
                    "mask_threshold": 0.5,
                    "plot": False,
                    "output": str(output),
                },
            }
        )
    )
    old_argv = sys.argv
    try:
        sys.argv = ["deal-mask", "-c", str(config)]
        mask_main()
    finally:
        sys.argv = old_argv
    assert output.is_file()
    masked = read(output, index=":", format="extxyz")
    np.testing.assert_array_equal(masked[0].arrays["deal_mask"], [0, 1])
