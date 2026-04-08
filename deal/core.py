from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import List, Optional, Sequence, Dict, Any

import numpy as np
import sys
from pprint import pformat
import json
from copy import deepcopy
import time

from ase import Atoms
from ase.io import iread, write
from ase.calculators.singlepoint import SinglePointCalculator

from flare.learners.utils import is_std_in_bound
from flare.atoms import FLARE_Atoms


@dataclass
class DataConfig:
    # --- data / trajectory ---
    files: Optional[str | List[str]] = None
    atoms_list: Optional[List[Atoms]] = None
    format: Optional[str] = None
    index: str = ":"  # ASE selection string
    colvar: Optional[List[str]] = None
    shuffle: bool = False
    seed: int = 24

    def __post_init__(self):
        if isinstance(self.files, str):
            self.files = [self.files]

        if self.files is not None and len(self.files) == 0:
            self.files = None
        if self.atoms_list is not None and len(self.atoms_list) == 0:
            self.atoms_list = None

        if self.files is not None and self.atoms_list is not None:
            raise ValueError(
                "Provide exactly one of 'files' or 'atoms_list' in DataConfig."
            )
        if self.files is None and self.atoms_list is None:
            raise ValueError(
                "Provide exactly one of 'files' or 'atoms_list' in DataConfig."
            )

        if self.files is not None:
            loaded_atoms = []
            for fname in self.files:
                loaded_atoms.extend(
                    list(iread(fname, index=self.index, format=self.format))
                )
            if len(loaded_atoms) == 0:
                raise ValueError("DataConfig does not contain any frames.")
            self.atoms_list = loaded_atoms
            self.files = None

        if self.atoms_list is not None:
            for i, atoms in enumerate(self.atoms_list):
                if not isinstance(atoms, Atoms):
                    raise TypeError(
                        f"DataConfig.atoms_list[{i}] is not an ASE Atoms object."
                    )


@dataclass
class DEALConfig:
    # --- selection parameters ---
    threshold: float = 1.0
    update_threshold: Optional[float | List[float]] = None

    max_atoms_added: Optional[float | int] = 0.2
    # max_atoms_added can be:
    #  - int >= 1 : explicit number of atoms to add
    #  - float in (0,1) : fraction of atoms to add (relative to system size)
    #  - -1 : no limit
    #
    min_steps_with_model: int = 0  # frames between two selections

    initial_atoms: Optional[List[int] | float] = None
    # atoms to use for initial training. Allowed values:
    #   - list of atom indices
    #   - float in (0,1) : fraction of atoms (computed per species)
    #   - None (use 1 atom per species)

    # --- GP training options ---
    force_only: bool = True  # ignore energies/stress if True
    train_hyps: bool = False  # train hyperparams after each update

    # --- output ---
    output_prefix: str = "deal"
    verbose: bool | str = True  # allowed values: true/false/"debug" (default: false)
    save_gp: bool = False
    save_full_trajectory: bool = False
    debug: bool = False  # internal debug flag

    # --- Validation of parameters ---
    def __post_init__(self):
        # --- Default update_threshold ---
        if self.update_threshold is None:
            if isinstance(self.threshold, list):
                # If threshold is a list, compute update_threshold as list too
                self.update_threshold = [0.8 * t for t in self.threshold]
            else:
                self.update_threshold = 0.8 * self.threshold

        # --- Check that threshold and update_threshold lists match in length ---
        if isinstance(self.threshold, list) and isinstance(self.update_threshold, list):
            if len(self.threshold) != len(self.update_threshold):
                raise ValueError(
                    f"Length of 'threshold' list ({len(self.threshold)}) must match "
                    f"length of 'update_threshold' list ({len(self.update_threshold)})"
                )
        if isinstance(self.threshold, list) and isinstance(
            self.update_threshold, float
        ):
            self.update_threshold = [self.update_threshold for _ in self.threshold]
            print(
                f"[WARNING] 'update_threshold' was a float while 'threshold' was a list. "
                f"Converted 'update_threshold' to list: {self.update_threshold}"
            )

        # --- Check max_atoms_added ---
        if isinstance(self.max_atoms_added, int):
            if self.max_atoms_added == 0 or self.max_atoms_added < -1:
                print(
                    f"[WARNING] Invalid value for max_atoms_added "
                    f"'{self.max_atoms_added}'. Resetting to '-1' (no limit)."
                )
                self.max_atoms_added = -1

        elif isinstance(self.max_atoms_added, float):
            if not (0 < self.max_atoms_added < 1):
                # cast to int
                self.max_atoms_added = int(self.max_atoms_added)
                if self.max_atoms_added <= 0:
                    print(
                        f"[WARNING] Invalid max_atoms_added fraction. Resetting to '-1' (no limit)."
                    )
                    self.max_atoms_added = -1
                else:
                    print(
                        f"[WARNING] Invalid fraction of max_atoms_added. Casting to int '{self.max_atoms_added}'."
                    )

        # --- Check initial_atoms validity ---
        if isinstance(self.initial_atoms, float):
            if not (0 < self.initial_atoms < 1):
                print(
                    f"[WARNING] Invalid initial_atoms fraction '{self.initial_atoms}'. Resetting to None (use 1 per species)."
                )
                self.initial_atoms = None

        # --- Handle verbose/debug logic ---
        if isinstance(self.verbose, str):
            if self.verbose.lower() == "debug":
                self.verbose = True
                self.debug = True
            else:
                print(
                    f"[WARNING] Invalid verbose option '{self.verbose}'. Setting verbose to False."
                )
                self.verbose = False


@dataclass
class FlareConfig:
    # --- gp ---
    gp: str = "SGP_Wrapper"

    # --- kernel ---
    kernels: List[Dict[str, Any]] = field(
        default_factory=lambda: [
            {"name": "NormalizedDotProduct", "sigma": 2.0, "power": 2}
        ]
    )
    # --- descriptor ---
    descriptors: List[Dict[str, Any]] = field(
        default_factory=lambda: [
            {
                "name": "B2",
                "nmax": 8,
                "lmax": 3,
                "cutoff_function": "cosine",
                "radial_basis": "chebyshev",
            }
        ]
    )
    # --- species ---
    species: list[int] = None
    # --- parameters ---
    cutoff: float = 4.5
    variance_type: str = "local"
    max_iterations: int = 20


class DEAL:
    """
    Minimal DEAL-style selector:

    - iterate over a trajectory (ASE files)
    - for each frame:
        * read energy/forces/stress from the file
        * evaluate SGP and its uncertainties
        * if uncertainty is above threshold -> "select" frame and update SGP

    Outputs:
        <output_prefix>_selected.xyz
        <output_prefix>_flare.json
    """

    def __init__(
        self, data_cfg: DataConfig, deal_cfg: DEALConfig, flare_cfg: FlareConfig
    ):

        # Configure using provided config objects
        self.configure_run(data_cfg, deal_cfg, flare_cfg)

        if self.deal_cfg.verbose:
            print("[INFO] Configurations:")
            print("-", pformat(self.data_cfg))
            print("-", pformat(self.deal_cfg))
            print("-", pformat(self.flare_cfg),"\n")

        # Timing accumulation
        self.timers = {
            "start": time.perf_counter(),
            "total": 0.0,
            "extract_dft": 0.0,
            "predict": 0.0,
            "update": 0.0,
            "io_write": 0.0,
            "other": 0.0,
        }

    def configure_run(
        self,
        data_cfg: Optional[DataConfig] = None,
        deal_cfg: Optional[DEALConfig] = None,
        flare_cfg: Optional[FlareConfig] = None,
    ) -> None:
        """
        Setup configuration objects when provided.
        SGP is rebuilt only when flare_cfg is provided.
        """
        data_changed = data_cfg is not None
        deal_changed = deal_cfg is not None
        flare_changed = flare_cfg is not None

        if data_changed:
            if not isinstance(data_cfg, DataConfig):
                raise TypeError(
                    f"Expected DataConfig for data_cfg, got {type(data_cfg)}."
                )
            self.data_cfg = data_cfg
            self.data_cfg.__post_init__()
            self.rng = np.random.default_rng(self.data_cfg.seed)

        if deal_changed:
            if not isinstance(deal_cfg, DEALConfig):
                raise TypeError(
                    f"Expected DEALConfig for deal_cfg, got {type(deal_cfg)}."
                )
            self.deal_cfg = deal_cfg
            self.deal_cfg.__post_init__()

        if flare_changed:
            if not isinstance(flare_cfg, FlareConfig):
                raise TypeError(
                    f"Expected FlareConfig for flare_cfg, got {type(flare_cfg)}."
                )
            self.flare_cfg = flare_cfg

        if data_changed and not flare_changed and self.flare_cfg.species is not None:
            data_species = self._get_species()
            if data_species != sorted(self.flare_cfg.species):
                raise ValueError(
                    "Input species changed but FLARE model was not rebuilt. "
                    "Pass flare_cfg to configure_run to rebuild SGP."
                )

        if flare_changed:
            if self.flare_cfg.species is None:
                self.flare_cfg.species = self._get_species()
            self.flare_calc, self.kernels = self._get_sgp_calc(asdict(self.flare_cfg))
            self.gp = self.flare_calc.gp_model
            self.selected_frames = []
            self.dft_count = 0

        self.last_dft_step = -(10**9)

    # ------------------------------------------------------------------
    # basic helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _copy_atoms_with_results(atoms: Atoms) -> Atoms:
        copied = atoms.copy()
        if atoms.calc is not None:
            copied.calc = SinglePointCalculator(copied)
            copied.calc.results = deepcopy(getattr(atoms.calc, "results", {}) or {})
        return copied

    def _frames(self):
        """Generator over all frames, optionally shuffled, with
        atoms.info['frame'] containing the original global index."""

        if not self.data_cfg.shuffle:
            # Streaming, non-shuffled mode
            global_idx = 0
            for atoms in self.data_cfg.atoms_list or []:
                at = self._copy_atoms_with_results(atoms)
                at.info["frame"] = global_idx
                global_idx += 1
                yield at
            return

        # --- Shuffling mode: load all frames first ---
        frames = []
        global_idx = 0

        for atoms in self.data_cfg.atoms_list or []:
            at = self._copy_atoms_with_results(atoms)
            at.info["frame"] = global_idx
            frames.append(at)
            global_idx += 1

        self.rng.shuffle(frames)

        for at in frames:
            yield at

    def _get_species(self):
        """
        Detect species automatically using the DataConfig instance.
        Reads only the first available frame.
        """
        for atoms in self.data_cfg.atoms_list or []:
            return sorted(set(atoms.get_atomic_numbers().tolist()))
        raise ValueError("Could not detect species: no input frames available.")

    def _extract_dft(self, ase_atoms):
        """
        Extract DFT forces / energy / stress from a frame.

        Assumes the extxyz was written with energies and forces and that
        ASE has attached a SinglePointCalculator to atoms.calc.
        """
        if ase_atoms.calc is None:
            raise RuntimeError(
                "Frame has no calculator attached. Make sure your extxyz "
                "contains energies/forces so ASE builds a SinglePointCalculator."
            )

        res = ase_atoms.calc.results
        if "forces" not in res:
            raise RuntimeError(
                "Frame is missing 'forces' in calculator results. "
                "Input data must include force labels."
            )
        if "energy" not in res and "energy" not in ase_atoms.info:
            raise RuntimeError(
                "Frame is missing 'energy' in calculator results/info. "
                "Input data must include energy labels."
            )
        forces = np.array(res["forces"])
        energy = float(res["energy"]) if "energy" in res else float(ase_atoms.info["energy"])
        stress = res.get("stress", None)

        return forces, energy, stress

    def _print_progress(self, step: int, elapsed: float, step_time: float):
        msg = f"[DEAL] Examined: {step+1:>5} | Selected: {self.dft_count:>5} | Speed: {step_time:6.2f} s/step | Elapsed: {elapsed:8.2f} s"
        sys.stdout.write("\r" + msg)
        sys.stdout.flush()

    def _extract_atomic_uncertainty(self, atoms, n_atoms: int) -> np.ndarray:
        """
        Extract per-atom uncertainty values from FLARE_Atoms after prediction.
        Returns a (n_atoms,) array, using NaN values when unavailable.
        """
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

    def _store_full_trajectory_frame(
        self, step: int, ase_frame, atomic_uncertainty: np.ndarray
    ) -> None:
        """Store one frame of the full trajectory with atomic uncertainty array."""
        if not self.deal_cfg.save_full_trajectory:
            return
        frame = ase_frame.copy()
        frame.info["step"] = step
        frame.arrays["atomic_uncertainty"] = np.asarray(atomic_uncertainty, dtype=float)
        frame.info["max_atomic_uncertainty"] = (
            float(np.nanmax(atomic_uncertainty))
            if np.any(np.isfinite(atomic_uncertainty))
            else float("nan")
        )
        write(f"{self.deal_cfg.output_prefix}_trajectory_uncertainty.xyz", frame, append=(step != 0))

    # ------------------------------------------------------------------
    # main loop
    # ------------------------------------------------------------------
    def run(self) -> None:
        for step, ase_frame in enumerate(self._frames()):
            step_start = time.perf_counter()
            init_frame = False
            # 1) DFT labels from original ASE frame
            t0 = time.perf_counter()
            dft_forces, dft_energy, dft_stress = self._extract_dft(ase_frame)
            dt = time.perf_counter() - t0
            self.timers["extract_dft"] += dt

            # 2) Convert to FLARE_Atoms for SGP calculations & uncertainties
            atoms = FLARE_Atoms.from_ase_atoms(ase_frame)

            # 2a) INITIALIZATION: if GP has no training data, use first frame
            #     to bootstrap the model (no uncertainty check).
            if len(self.gp.training_data) == 0:
                t_up0 = time.perf_counter()
                init_frame = True
                if self.deal_cfg.debug:
                    sys.stdout.write(
                        "\r"
                        + f"[DEBUG] : step {step+1} : Initializing GP with first frame\n"
                    )
                if isinstance(self.deal_cfg.initial_atoms, list):
                    init_atoms = self.deal_cfg.initial_atoms
                else:
                    unique_species = sorted(set(atoms.get_atomic_numbers().tolist()))
                    idx_species = {sp: [] for sp in unique_species}
                    for sp in unique_species:
                        idx_species[sp] = [
                            i for i, at in enumerate(atoms) if at.number == sp
                        ]
                        self.rng.shuffle(
                            idx_species[sp]
                        )  # indices of this species randomly shuffle

                    if self.deal_cfg.initial_atoms is None:  # use 1 atom per species
                        init_atoms = [idx_species[sp][0] for sp in unique_species]
                    else:  # use fraction of atoms per species
                        init_atoms = []
                        for sp in unique_species:
                            init_atoms += idx_species[sp][
                                : int(
                                    np.ceil(
                                        self.deal_cfg.initial_atoms
                                        * len(idx_species[sp])
                                    )
                                )
                            ]
                if self.deal_cfg.debug:
                    sys.stdout.write(
                        "\r"
                        + f"[DEBUG] : step {step+1} : Initial atoms selected : {init_atoms}"
                    )
                self._update_gp(
                    atoms=atoms,
                    train_atoms=init_atoms,
                    dft_frcs=dft_forces,
                    dft_energy=dft_energy,
                    dft_stress=dft_stress,
                )
                self.timers["update"] += time.perf_counter() - t_up0

            # 3) Predict with SGP and compute uncertainties
            t_pred0 = time.perf_counter()
            atoms.calc = self.flare_calc
            _ = atoms.get_forces()  # triggers GP eval and stores stds internally
            atomic_uncertainty = self._extract_atomic_uncertainty(atoms, len(atoms))

            if self.deal_cfg.save_full_trajectory:
                t_io0 = time.perf_counter()
                self._store_full_trajectory_frame(step, ase_frame, atomic_uncertainty)
                self.timers["io_write"] += time.perf_counter() - t_io0

            max_atom_added = (
                self.deal_cfg.max_atoms_added
                if isinstance(self.deal_cfg.max_atoms_added, int)
                else int(np.ceil(self.deal_cfg.max_atoms_added * len(atoms)))
            )
            std_in_bound, target_atoms = is_std_in_bound(
                self.deal_cfg.threshold * -1,  # threshold = - std_tolerance_factor
                self.gp.force_noise,
                atoms,
                update_style="threshold",
                update_threshold=self.deal_cfg.update_threshold,
            )
            if (
                0 < max_atom_added < len(target_atoms)
            ):  # only keep up to max_atoms_added atoms
                target_atoms = target_atoms[-max_atom_added:]
            self.timers["predict"] += time.perf_counter() - t_pred0
            if self.deal_cfg.debug:
                sys.stdout.write(
                    "\r"
                    + f"[DEBUG] : step {step+1} : {std_in_bound} : {target_atoms} : {self.deal_cfg.max_atoms_added if isinstance(self.deal_cfg.max_atoms_added, int) else int(np.ceil(self.deal_cfg.max_atoms_added * len(atoms)))}"
                )

            steps_since_last = step - self.last_dft_step

            if (not std_in_bound) and (
                steps_since_last >= self.deal_cfg.min_steps_with_model
            ):
                # Select this frame & update GP
                t_up0 = time.perf_counter()
                self.last_dft_step = step
                self._store_selected_frame(
                    step,
                    ase_frame,
                    target_atoms + init_atoms if init_frame else target_atoms,
                )
                if self.deal_cfg.debug:
                    sys.stdout.write(
                        "\r"
                        + f"[DEBUG] : step {step+1} : Atoms selected : {target_atoms+init_atoms if init_frame else target_atoms}"
                    )
                self._update_gp(
                    atoms=atoms,
                    train_atoms=list(target_atoms),
                    dft_frcs=dft_forces,
                    dft_energy=dft_energy,
                    dft_stress=dft_stress,
                )
                self.timers["update"] += time.perf_counter() - t_up0
            elif (
                std_in_bound and init_frame
            ):  # In initial frame case, store selected frame even if stds are okay
                self.last_dft_step = step
                self._store_selected_frame(step, ase_frame, target_atoms=init_atoms)
            # ========== print progress ==========
            elapsed = time.perf_counter() - self.timers["start"]
            step_time = time.perf_counter() - step_start
            self._print_progress(step, elapsed, step_time)

        # newline so terminal prompt doesn't collide with progress line
        print("")

        # ------------------------------------------------------------------
        # outputs
        # ------------------------------------------------------------------
        if self.selected_frames and self.deal_cfg.verbose:
            print(f"[OUTPUT] Saved selected frames to: {self.deal_cfg.output_prefix}_selected.xyz")

        if self.selected_frames and self.deal_cfg.save_gp:
            # Save final SGP model
            t_io0 = time.perf_counter()
            self.flare_calc.write_model(f"{self.deal_cfg.output_prefix}_flare.json")
            self.timers["io_write"] += time.perf_counter() - t_io0
            if self.deal_cfg.verbose:
                print(
                    f"[OUTPUT] Saved GP model to {self.deal_cfg.output_prefix}_flare.json"
                )

        if self.deal_cfg.save_full_trajectory and self.deal_cfg.verbose:
            print(
                f"[OUTPUT] Saved full trajectory with atomic uncertainty {self.deal_cfg.output_prefix}_trajectory_uncertainty.xyz to:"
            )

        # final total time

        self.timers["total"] = time.perf_counter() 

        if self.deal_cfg.verbose:
            total = self.timers["total"] - self.timers["start"]
            extract = self.timers["extract_dft"]
            pred = self.timers["predict"]
            upd = self.timers["update"]
            iow = self.timers["io_write"]
            oth = max(0.0, total - (extract + pred + upd + iow))
            self.timers["other"] = oth
            print("\n[TIMING] Summary (s):")
            print(f"  total           : {total:8.2f}")
            print(f"    extract_dft   : {extract:8.2f} ({extract/total*100:4.1f}%)")
            print(f"    predict       : {pred:8.2f} ({pred/total*100:4.1f}%)")
            print(f"    update        : {upd:8.2f} ({upd/total*100:4.1f}%)")
            print(f"    io_write      : {iow:8.2f} ({iow/total*100:4.1f}%)")
            print(f"    other         : {oth:8.2f} ({oth/total*100:4.1f}%)")

    # ------------------------------------------------------------------
    # GP creation and update
    # ------------------------------------------------------------------

    def _get_sgp_calc(self, flare_config):
        """
        Return a SGP_Calculator with sgp from SparseGP
        source: https://github.com/mir-group/flare/blob/master/flare/scripts/otf_train.py
        """
        from flare.bffs.sgp._C_flare import NormalizedDotProduct, SquaredExponential
        from flare.bffs.sgp._C_flare import B2, B3, TwoBody, ThreeBody, FourBody
        from flare.bffs.sgp import SGP_Wrapper
        from flare.bffs.sgp.calculator import SGP_Calculator

        sgp_file = flare_config.get("file", None)

        # Load sparse GP from file
        if sgp_file is not None:
            with open(sgp_file, "r") as f:
                gp_dct = json.loads(f.readline())
                if gp_dct.get("class", None) == "SGP_Calculator":
                    flare_calc, kernels = SGP_Calculator.from_file(sgp_file)
                else:
                    sgp, kernels = SGP_Wrapper.from_file(sgp_file)
                    flare_calc = SGP_Calculator(sgp)
            return flare_calc, kernels

        kernels = flare_config.get("kernels")
        opt_algorithm = flare_config.get("opt_algorithm", "BFGS")
        max_iterations = flare_config.get("max_iterations", 20)
        bounds = flare_config.get("bounds", None)
        use_mapping = flare_config.get("use_mapping", False)

        # Define kernels.
        kernels = []
        for k in flare_config["kernels"]:
            if k["name"] == "NormalizedDotProduct":
                kernels.append(NormalizedDotProduct(k["sigma"], k["power"]))
            elif k["name"] == "SquaredExponential":
                kernels.append(SquaredExponential(k["sigma"], k["ls"]))
            else:
                raise NotImplementedError(f"{k['name']} kernel is not implemented")

        # Define descriptor calculators.
        n_species = len(flare_config["species"])
        cutoff = flare_config["cutoff"]
        descriptors = []
        for d in flare_config["descriptors"]:
            if "cutoff_matrix" in d:  # multiple cutoffs
                assert np.allclose(
                    np.array(d["cutoff_matrix"]).shape, (n_species, n_species)
                ), "cutoff_matrix needs to be of shape (n_species, n_species)"

            if d["name"] == "B2":
                radial_hyps = [0.0, cutoff]
                cutoff_hyps = []
                descriptor_settings = [n_species, d["nmax"], d["lmax"]]
                if "cutoff_matrix" in d:  # multiple cutoffs
                    desc_calc = B2(
                        d["radial_basis"],
                        d["cutoff_function"],
                        radial_hyps,
                        cutoff_hyps,
                        descriptor_settings,
                        d["cutoff_matrix"],
                    )
                else:
                    desc_calc = B2(
                        d["radial_basis"],
                        d["cutoff_function"],
                        radial_hyps,
                        cutoff_hyps,
                        descriptor_settings,
                    )

            elif d["name"] == "B3":
                radial_hyps = [0.0, cutoff]
                cutoff_hyps = []
                descriptor_settings = [n_species, d["nmax"], d["lmax"]]
                desc_calc = B3(
                    d["radial_basis"],
                    d["cutoff_function"],
                    radial_hyps,
                    cutoff_hyps,
                    descriptor_settings,
                )

            elif d["name"] == "TwoBody":
                desc_calc = TwoBody(
                    cutoff, n_species, d["cutoff_function"], cutoff_hyps
                )

            elif d["name"] == "ThreeBody":
                desc_calc = ThreeBody(
                    cutoff, n_species, d["cutoff_function"], cutoff_hyps
                )

            elif d["name"] == "FourBody":
                desc_calc = FourBody(
                    cutoff, n_species, d["cutoff_function"], cutoff_hyps
                )

            else:
                raise NotImplementedError(f"{d['name']} descriptor is not supported")

            descriptors.append(desc_calc)

        # Define remaining parameters for the SGP wrapper.
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

        flare_calc = SGP_Calculator(sgp, use_mapping)
        return flare_calc, kernels

    def _store_selected_frame(self, step: int, ase_frame, target_atoms: Sequence[int]):
        """Keep a copy of the selected ASE frame for writing to XYZ."""
        sel = ase_frame.copy()
        sel.info["step"] = step
        if "frame" not in sel.info:
            sel.info["frame"] = step
        sel.info["threshold"] = self.deal_cfg.threshold
        sel.info["target_atoms"] = np.array(target_atoms, dtype=int)
        t_io0 = time.perf_counter()
        write(f"{self.deal_cfg.output_prefix}_selected.xyz", sel, append=(self.dft_count != 0))
        self.timers["io_write"] += time.perf_counter() - t_io0
        self.selected_frames.append(sel)
        self.dft_count += 1

    def _update_gp(
        self,
        atoms,
        train_atoms: Sequence[int],
        dft_frcs: np.ndarray,
        dft_energy: float | None = None,
        dft_stress: np.ndarray | None = None,
    ) -> None:
        """
        Update the SGP using DFT forces (and optionally energies/stress)
        on the current FLARE_Atoms structure.

        This mirrors original update gp from FLARE OTF, but without
        wall-time logging or mapping.
        """
        # Convert stress into FLARE convention if present
        flare_stress = None
        if dft_stress is not None:
            dft_stress = np.asarray(dft_stress)
            # allow either 3x3 tensor or 6-vector from ASE
            if dft_stress.shape == (3, 3):
                xx, yy, zz = dft_stress[0, 0], dft_stress[1, 1], dft_stress[2, 2]
                yz, xz, xy = dft_stress[1, 2], dft_stress[0, 2], dft_stress[0, 1]
                dft_stress_voigt = np.array([xx, yy, zz, yz, xz, xy])
            else:
                dft_stress_voigt = dft_stress

            # ASE uses +sigma; FLARE uses -sigma in this convention
            flare_stress = -np.array(
                [
                    dft_stress_voigt[0],
                    dft_stress_voigt[5],
                    dft_stress_voigt[4],
                    dft_stress_voigt[1],
                    dft_stress_voigt[3],
                    dft_stress_voigt[2],
                ]
            )

        if self.deal_cfg.force_only:
            dft_energy = None
            flare_stress = None

        # Store a copy of the structure, attach DFT labels via SinglePointCalculator
        struc_to_add = deepcopy(atoms)

        sp_results = {"forces": dft_frcs}
        if dft_energy is not None:
            sp_results["energy"] = dft_energy
        if flare_stress is not None:
            sp_results["stress"] = flare_stress

        struc_to_add.calc = SinglePointCalculator(struc_to_add, **sp_results)

        # Update GP database
        self.gp.update_db(
            struc_to_add,
            dft_frcs,
            custom_range=list(train_atoms),
            energy=dft_energy,
            stress=np.zeros(6) if flare_stress is None else flare_stress,
        )

        # Update internal L and alpha
        self.gp.set_L_alpha()

        # Train hyperparameters
        if self.deal_cfg.train_hyps:
            self.gp.train(logger_name=None)
