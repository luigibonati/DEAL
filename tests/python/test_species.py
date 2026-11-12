import pytest
from ase import Atoms

from deal import DataConfig, DEALConfig, DEAL, SGPConfig


def _frame(formula):
    atoms = Atoms(formula, cell=[10.0, 10.0, 10.0], pbc=True)
    atoms.positions = [[1.5 * i, 0.0, 0.0] for i in range(len(atoms))]
    return atoms


def _deal(images, species=None):
    return DEAL(
        DataConfig(images=images),
        DEALConfig(threshold=0.5, verbose=False),
        SGPConfig(cutoff=3.0, species=species),
    )


def test_species_detected_from_all_frames():
    # the first frame does not contain every species of the dataset
    deal = _deal([_frame("H2"), _frame("HAl"), _frame("HO")])

    assert deal.sgp_cfg.species == [1, 8, 13]


def test_new_data_with_subset_of_species_does_not_require_rebuild():
    deal = _deal([_frame("H2"), _frame("HAl")], species=[1, 6, 13])

    deal.configure_run(data_cfg=DataConfig(images=[_frame("H2")]))


def test_new_data_with_uncovered_species_requires_rebuild():
    deal = _deal([_frame("H2"), _frame("HAl")])

    with pytest.raises(ValueError, match=r"\[8\]"):
        deal.configure_run(data_cfg=DataConfig(images=[_frame("HO")]))
