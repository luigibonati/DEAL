import numpy as np
from ase import Atoms

from deal import SGPConfig
from deal.model import DealActiveLearningModel


def test_nonperiodic_atoms_without_a_cell_support_local_uncertainty():
    atoms = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]])
    assert not atoms.pbc.any()
    assert np.allclose(atoms.cell, 0.0)

    model_atoms = DealActiveLearningModel.to_model_atoms(atoms)
    assert not model_atoms.pbc.any()

    model = DealActiveLearningModel(SGPConfig(cutoff=3.0, species=[1]))
    model.update(atoms, train_atoms=[0], dft_forces=None, local_uncertainty_only=True)
    uncertainty = model.predict_uncertainty(atoms)

    assert uncertainty.shape == (len(atoms),)
    assert np.all(np.isfinite(uncertainty))
