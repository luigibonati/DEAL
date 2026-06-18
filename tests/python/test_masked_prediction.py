import numpy as np
from ase.io import read

from deal import SGPConfig
from deal.model import DealActiveLearningModel


def main():
    atoms = read("../data/traj.xyz", index=0)
    species = sorted(set(int(z) for z in atoms.numbers))
    train_atoms = [
        int(np.flatnonzero(atoms.numbers == atomic_number)[0])
        for atomic_number in species
    ]
    candidate_atoms = train_atoms[:2]

    model = DealActiveLearningModel(SGPConfig(cutoff=5.0, species=species))
    model.update(
        atoms,
        train_atoms=train_atoms,
        dft_forces=atoms.get_forces(),
        dft_energy=atoms.get_potential_energy(),
    )

    full_uncertainty = model.predict_uncertainty(atoms)
    masked_uncertainty = model.predict_uncertainty(
        atoms, candidate_atoms=candidate_atoms
    )

    np.testing.assert_allclose(
        masked_uncertainty[candidate_atoms],
        full_uncertainty[candidate_atoms],
        rtol=1e-8,
        atol=1e-12,
    )

    raw_stds = model.calculator.results["stds"][:, 0]
    non_candidates = np.setdiff1d(np.arange(len(atoms)), candidate_atoms)
    assert np.all(raw_stds[non_candidates] < 0.0)
    assert np.all(raw_stds[candidate_atoms] >= 0.0)


if __name__ == "__main__":
    main()
