import numpy as np
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator

from deal.core import DEAL


def test_extract_dft_accepts_unlabeled_frame():
    atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]])

    forces, energy, stress = DEAL._extract_dft(None, atoms)

    assert forces is None
    assert energy is None
    assert stress is None


def test_extract_dft_accepts_energy_only_frame():
    atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]])
    atoms.calc = SinglePointCalculator(atoms, energy=-1.25)

    forces, energy, stress = DEAL._extract_dft(None, atoms)

    assert forces is None
    assert energy == -1.25
    assert stress is None


def test_extract_dft_keeps_available_forces_without_stress():
    atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]])
    expected_forces = np.array([[0.1, 0.2, 0.3]])
    atoms.calc = SinglePointCalculator(
        atoms, energy=-1.25, forces=expected_forces
    )

    forces, energy, stress = DEAL._extract_dft(None, atoms)

    np.testing.assert_array_equal(forces, expected_forces)
    assert energy == -1.25
    assert stress is None
