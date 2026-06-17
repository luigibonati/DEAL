from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import asdict
from typing import Optional, Sequence

import numpy as np
from ase.calculators.singlepoint import SinglePointCalculator

from .sgp import FLARE_Atoms, SGP_Calculator, SGP_Wrapper
from .sgp.utils import is_std_in_bound

from .config import FlareConfig


class DealActiveLearningModel:
    """Narrow model surface used by DEAL active learning.

    This is intentionally smaller than FLARE's public API.  It isolates the
    SGP local-uncertainty workflow that DEAL needs today, and gives us a
    stable place to replace FLARE internals with a DEAL-specific engine later.
    """

    def __init__(self, config: FlareConfig):
        self.config = config
        self.calculator, self.kernels = self._build_sgp_calculator(asdict(config))
        self.gp = self.calculator.gp_model

    @property
    def training_size(self) -> int:
        return len(self.gp.training_data)

    @property
    def force_noise(self) -> float:
        return self.gp.force_noise

    @staticmethod
    def to_model_atoms(ase_atoms):
        return FLARE_Atoms.from_ase_atoms(ase_atoms)

    @staticmethod
    def extract_atomic_uncertainty(atoms, n_atoms: int) -> np.ndarray:
        candidates = [
            getattr(atoms, "stds", None),
            getattr(atoms, "local_uncertainties", None),
            getattr(atoms, "uncertainties", None),
        ]
        if getattr(atoms, "calc", None) is not None:
            results = getattr(atoms.calc, "results", {}) or {}
            candidates.extend(
                [
                    results.get("stds"),
                    results.get("local_uncertainties"),
                    results.get("uncertainties"),
                ]
            )

        raw = next((c for c in candidates if c is not None), None)
        if raw is None:
            return np.full(n_atoms, np.nan, dtype=float)

        stds = np.asarray(raw, dtype=float)
        if stds.ndim == 1:
            if stds.shape[0] == n_atoms:
                return stds.copy()
            if stds.shape[0] == 3 * n_atoms:
                return np.linalg.norm(stds.reshape(n_atoms, 3), axis=1)
            if stds.size == n_atoms:
                return stds.reshape(n_atoms)
        elif stds.ndim == 2 and stds.shape[0] == n_atoms:
            if stds.shape[1] == 1:
                return stds[:, 0].copy()
            return np.linalg.norm(stds, axis=1)

        return np.full(n_atoms, np.nan, dtype=float)

    def predict_uncertainty(self, atoms) -> np.ndarray:
        atoms.calc = self.calculator
        _ = atoms.get_forces()
        return self.extract_atomic_uncertainty(atoms, len(atoms))

    def select_atoms_by_uncertainty(
        self,
        atoms,
        threshold: float,
        update_threshold: float,
        preselection_mask: Optional[np.ndarray] = None,
        use_preselected_targets: bool = True,
    ) -> tuple[bool, list[int]]:
        if preselection_mask is not None:
            std_in_bound, target_atoms = self._select_preselected_target_atoms(
                atoms, threshold, update_threshold, preselection_mask
            )
            if use_preselected_targets or std_in_bound:
                return std_in_bound, target_atoms

        std_in_bound, target_atoms = is_std_in_bound(
            threshold * -1,
            self.force_noise,
            atoms,
            update_style="threshold",
            update_threshold=update_threshold,
        )
        if preselection_mask is not None:
            target_atoms = [
                idx for idx in target_atoms if 0 <= idx < len(preselection_mask)
            ]
        return std_in_bound, target_atoms

    @staticmethod
    def _select_preselected_target_atoms(
        atoms,
        threshold: float,
        update_threshold: float,
        mask: np.ndarray,
    ) -> tuple[bool, list[int]]:
        candidate_atoms = np.flatnonzero(mask)
        if len(candidate_atoms) == 0:
            return True, []

        max_stds = {}
        for atom_index in candidate_atoms:
            max_stds[int(atom_index)] = float(np.max(atoms.stds[atom_index]))

        sorted_atoms = sorted(candidate_atoms.tolist(), key=lambda idx: max_stds[idx])
        target_atoms = [
            atom_index
            for atom_index in sorted_atoms
            if max_stds[atom_index] > update_threshold
        ]
        std_in_bound = max_stds[sorted_atoms[-1]] <= threshold
        if std_in_bound:
            target_atoms = []
        return std_in_bound, target_atoms

    def update(
        self,
        atoms,
        train_atoms: Sequence[int],
        dft_forces: np.ndarray,
        dft_energy: float | None = None,
        dft_stress: np.ndarray | None = None,
        force_only: bool = True,
        train_hyperparameters: bool = False,
    ) -> None:
        flare_stress = self._to_flare_stress(dft_stress)
        if force_only:
            dft_energy = None
            flare_stress = None

        structure_to_add = deepcopy(atoms)
        sp_results = {"forces": dft_forces}
        if dft_energy is not None:
            sp_results["energy"] = dft_energy
        if flare_stress is not None:
            sp_results["stress"] = flare_stress
        structure_to_add.calc = SinglePointCalculator(structure_to_add, **sp_results)

        self.gp.update_db(
            structure_to_add,
            dft_forces,
            custom_range=list(train_atoms),
            energy=dft_energy,
            stress=np.zeros(6) if flare_stress is None else flare_stress,
        )
        self.gp.set_L_alpha()

        if train_hyperparameters:
            self.gp.train(logger_name=None)

    def write(self, filename: str) -> None:
        self.calculator.write_model(filename)

    @staticmethod
    def _to_flare_stress(dft_stress: np.ndarray | None) -> np.ndarray | None:
        if dft_stress is None:
            return None
        dft_stress = np.asarray(dft_stress)
        if dft_stress.shape == (3, 3):
            xx, yy, zz = dft_stress[0, 0], dft_stress[1, 1], dft_stress[2, 2]
            yz, xz, xy = dft_stress[1, 2], dft_stress[0, 2], dft_stress[0, 1]
            dft_stress_voigt = np.array([xx, yy, zz, yz, xz, xy])
        else:
            dft_stress_voigt = dft_stress

        return -np.array(
            [
                dft_stress_voigt[0],
                dft_stress_voigt[5],
                dft_stress_voigt[4],
                dft_stress_voigt[1],
                dft_stress_voigt[3],
                dft_stress_voigt[2],
            ]
        )

    @staticmethod
    def _build_sgp_calculator(flare_config):
        from .sgp._C_flare import B2, B3, FourBody, ThreeBody, TwoBody
        from .sgp._C_flare import NormalizedDotProduct, SquaredExponential

        sgp_file = flare_config.get("file", None)

        if sgp_file is not None:
            with open(sgp_file, "r") as f:
                gp_dct = json.loads(f.readline())
                if gp_dct.get("class", None) == "SGP_Calculator":
                    return SGP_Calculator.from_file(sgp_file)
                sgp, kernels = SGP_Wrapper.from_file(sgp_file)
                return SGP_Calculator(sgp), kernels

        opt_algorithm = flare_config.get("opt_algorithm", "BFGS")
        max_iterations = flare_config.get("max_iterations", 20)
        bounds = flare_config.get("bounds", None)
        use_mapping = flare_config.get("use_mapping", False)

        kernels = []
        for kernel in flare_config["kernels"]:
            if kernel["name"] == "NormalizedDotProduct":
                kernels.append(NormalizedDotProduct(kernel["sigma"], kernel["power"]))
            elif kernel["name"] == "SquaredExponential":
                kernels.append(SquaredExponential(kernel["sigma"], kernel["ls"]))
            else:
                raise NotImplementedError(f"{kernel['name']} kernel is not implemented")

        n_species = len(flare_config["species"])
        cutoff = flare_config["cutoff"]
        descriptors = []
        for descriptor in flare_config["descriptors"]:
            cutoff_hyps = []
            if "cutoff_matrix" in descriptor:
                assert np.allclose(
                    np.array(descriptor["cutoff_matrix"]).shape,
                    (n_species, n_species),
                ), "cutoff_matrix needs to be of shape (n_species, n_species)"

            if descriptor["name"] == "B2":
                radial_hyps = [0.0, cutoff]
                descriptor_settings = [
                    n_species,
                    descriptor["nmax"],
                    descriptor["lmax"],
                ]
                if "cutoff_matrix" in descriptor:
                    desc_calc = B2(
                        descriptor["radial_basis"],
                        descriptor["cutoff_function"],
                        radial_hyps,
                        cutoff_hyps,
                        descriptor_settings,
                        descriptor["cutoff_matrix"],
                    )
                else:
                    desc_calc = B2(
                        descriptor["radial_basis"],
                        descriptor["cutoff_function"],
                        radial_hyps,
                        cutoff_hyps,
                        descriptor_settings,
                    )
            elif descriptor["name"] == "B3":
                radial_hyps = [0.0, cutoff]
                descriptor_settings = [
                    n_species,
                    descriptor["nmax"],
                    descriptor["lmax"],
                ]
                desc_calc = B3(
                    descriptor["radial_basis"],
                    descriptor["cutoff_function"],
                    radial_hyps,
                    cutoff_hyps,
                    descriptor_settings,
                )
            elif descriptor["name"] == "TwoBody":
                desc_calc = TwoBody(
                    cutoff, n_species, descriptor["cutoff_function"], cutoff_hyps
                )
            elif descriptor["name"] == "ThreeBody":
                desc_calc = ThreeBody(
                    cutoff, n_species, descriptor["cutoff_function"], cutoff_hyps
                )
            elif descriptor["name"] == "FourBody":
                desc_calc = FourBody(
                    cutoff, n_species, descriptor["cutoff_function"], cutoff_hyps
                )
            else:
                raise NotImplementedError(
                    f"{descriptor['name']} descriptor is not supported"
                )
            descriptors.append(desc_calc)

        species_map = {flare_config.get("species")[i]: i for i in range(n_species)}
        sae_dct = flare_config.get("single_atom_energies", None)
        if sae_dct is not None:
            assert n_species == len(
                sae_dct
            ), "'single_atom_energies' should be the same length as 'species'"
            single_atom_energies = {i: sae_dct[i] for i in range(n_species)}
        else:
            single_atom_energies = {i: 0 for i in range(n_species)}

        sgp = SGP_Wrapper(
            kernels=kernels,
            descriptor_calculators=descriptors,
            cutoff=cutoff,
            sigma_e=flare_config.get("energy_noise", 0.1),
            sigma_f=flare_config.get("forces_noise", 0.05),
            sigma_s=flare_config.get("stress_noise", 0.1),
            species_map=species_map,
            variance_type=flare_config.get("variance_type", "local"),
            single_atom_energies=single_atom_energies,
            energy_training=flare_config.get("energy_training", True),
            force_training=flare_config.get("force_training", True),
            stress_training=flare_config.get("stress_training", True),
            max_iterations=max_iterations,
            opt_method=opt_algorithm,
            bounds=bounds,
        )

        return SGP_Calculator(sgp, use_mapping), kernels
