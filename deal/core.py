from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
import sys
from pprint import pformat
from copy import deepcopy
import time

from ase import Atoms
from ase.io import write
from ase.calculators.singlepoint import SinglePointCalculator

from .config import DataConfig, DEALConfig, SGPConfig
from .model import DealActiveLearningModel
from .runtime import format_runtime_info


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
        <output_prefix>_sgp.json
    """

    def __init__(
        self, data_cfg: DataConfig, deal_cfg: DEALConfig, sgp_cfg: SGPConfig
    ):

        # Configure using provided config objects
        self.configure_run(data_cfg, deal_cfg, sgp_cfg)

        if self.deal_cfg.verbose:
            print(format_runtime_info())
            if self._use_local_uncertainty_fast_path():
                print(
                    "[INFO] Optimization: local uncertainty fast path enabled "
                    "(training labels, Kuf, posterior mean, and QR are skipped)."
                )
            print("[INFO] Configurations:")
            print("-", pformat(self.data_cfg))
            print("-", pformat(self.deal_cfg))
            print("-", pformat(self.sgp_cfg),"\n")

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
        sgp_cfg: Optional[SGPConfig] = None,
    ) -> None:
        """
        Setup configuration objects when provided.
        SGP is rebuilt only when sgp_cfg is provided.
        """
        data_changed = data_cfg is not None
        deal_changed = deal_cfg is not None
        sgp_changed = sgp_cfg is not None

        if data_changed:
            if not isinstance(data_cfg, DataConfig):
                raise TypeError(
                    f"Expected DataConfig for data_cfg, got {type(data_cfg)}."
                )
            self.data_cfg = data_cfg
            self.rng = np.random.default_rng(self.data_cfg.seed)

        if deal_changed:
            if not isinstance(deal_cfg, DEALConfig):
                raise TypeError(
                    f"Expected DEALConfig for deal_cfg, got {type(deal_cfg)}."
                )
            self.deal_cfg = deal_cfg

        if sgp_changed:
            if not isinstance(sgp_cfg, SGPConfig):
                raise TypeError(
                    f"Expected SGPConfig for sgp_cfg, got {type(sgp_cfg)}."
                )
            self.sgp_cfg = sgp_cfg

        if data_changed and not sgp_changed and self.sgp_cfg.species is not None:
            data_species = self._get_species()
            if data_species != sorted(self.sgp_cfg.species):
                raise ValueError(
                    "Input species changed but the SGP model was not rebuilt. "
                    "Pass sgp_cfg to configure_run to rebuild SGP."
                )

        if sgp_changed:
            if self.sgp_cfg.species is None:
                self.sgp_cfg.species = self._get_species()
            self.model = DealActiveLearningModel(self.sgp_cfg)
            self.sgp_calc = self.model.calculator
            self.kernels = self.model.kernels
            self.gp = self.model.gp
            self.selected_frames = []
            self.dft_count = 0

        self.last_dft_step = -(10**9)

    # ------------------------------------------------------------------
    # basic helpers
    # ------------------------------------------------------------------

    def _use_local_uncertainty_fast_path(self) -> bool:
        return (
            self.sgp_cfg.variance_type == "local"
            and not self.deal_cfg.train_hyps
            and not self.deal_cfg.save_gp
        )

    @staticmethod
    def _copy_atoms_with_results(atoms: Atoms) -> Atoms:
        copied = atoms.copy()
        if atoms.calc is not None:
            copied.calc = SinglePointCalculator(copied)
            copied.calc.results = deepcopy(getattr(atoms.calc, "results", {}) or {})
        return copied

    def _frames(self):
        """Generator over frames prepared in DataConfig."""
        for atoms in self.data_cfg.images or []:
            yield self._copy_atoms_with_results(atoms)

    def _get_species(self):
        """
        Detect species automatically using the DataConfig instance.
        Reads only the first available frame.
        """
        for atoms in self.data_cfg.images or []:
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

    @staticmethod
    def _format_atom_indices(atom_indices: Sequence[int], limit: int = 20) -> str:
        """Format atom indices without letting debug lines grow without bound."""
        indices = [int(index) for index in atom_indices]
        shown = ",".join(str(index) for index in indices[:limit])
        if len(indices) > limit:
            shown += f",...(+{len(indices) - limit})"
        return f"[{shown}]"

    def _debug(self, step: int, **fields) -> None:
        """Write a labelled, grep-friendly debug record for one frame event."""
        if not self.deal_cfg.debug:
            return
        details = " | ".join(f"{key}={value}" for key, value in fields.items())
        sys.stdout.write(f"\r[DEBUG] step={step + 1} | {details}\n")

    def _get_candidate_mask(self, ase_frame) -> Optional[np.ndarray]:
        """Return a boolean candidate mask from the configured frame array."""
        if self.deal_cfg.mask is False:
            return None

        key = "deal_mask" if self.deal_cfg.mask is True else self.deal_cfg.mask
        if key not in ase_frame.arrays:
            raise RuntimeError(
                f"Frame is missing per-atom mask array '{key}'. "
                "Set 'mask: false' to use all atoms or prepare/provide the mask array."
            )

        values = np.asarray(ase_frame.arrays[key])
        n_atoms = len(ase_frame)
        if values.shape[0] != n_atoms:
            raise RuntimeError(
                f"Per-atom mask array '{key}' has incompatible shape "
                f"{values.shape}; expected first dimension {n_atoms}."
            )
        if values.ndim > 1:
            values = np.nanmax(values.reshape(n_atoms, -1).astype(float), axis=1)

        return values.astype(bool)

    def _apply_candidate_mask(
        self, atomic_uncertainty: np.ndarray, mask: Optional[np.ndarray]
    ) -> np.ndarray:
        """Mark atoms excluded by a candidate mask as ineligible."""
        if mask is None:
            return atomic_uncertainty
        filtered = atomic_uncertainty.copy()
        filtered[~mask] = -1.0
        return filtered

    @staticmethod
    def _filter_atoms_by_mask(
        atom_indices: Sequence[int], mask: Optional[np.ndarray]
    ) -> List[int]:
        if mask is None:
            return list(atom_indices)
        return [idx for idx in atom_indices if 0 <= idx < len(mask) and mask[idx]]

    def _select_masked_target_atoms(self, atoms, mask: np.ndarray):
        """Select target atoms using only atoms that pass the candidate mask."""
        threshold = self.deal_cfg.threshold
        update_threshold = self.deal_cfg.update_threshold
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

    def _store_full_trajectory_frame(
        self, step: int, ase_frame
    ) -> None:
        """Store one frame of the full trajectory with atomic uncertainty array."""
        if not self.deal_cfg.save_full_trajectory:
            return
        frame = ase_frame.copy()
        frame.info["step"] = step
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

            candidate_mask = self._get_candidate_mask(ase_frame)
            if (
                candidate_mask is not None
                and not np.any(candidate_mask)
            ):
                ase_frame.arrays["atomic_uncertainty"] = np.full(
                    len(ase_frame),
                    -1.0,
                    dtype=float,
                )
                ase_frame.info["max_atomic_uncertainty"] = -1.0
                if self.deal_cfg.save_full_trajectory:
                    t_io0 = time.perf_counter()
                    self._store_full_trajectory_frame(step, ase_frame)
                    self.timers["io_write"] += time.perf_counter() - t_io0
                self._debug(
                    step,
                    decision="skip",
                    reason="empty_candidate_mask",
                    candidates=f"0/{len(ase_frame)}",
                )
                elapsed = time.perf_counter() - self.timers["start"]
                step_time = time.perf_counter() - step_start
                self._print_progress(step, elapsed, step_time)
                continue

            # 2) Convert to SGP atoms for calculations & uncertainties
            atoms = self.model.to_model_atoms(ase_frame)

            # 2a) INITIALIZATION: if GP has no training data, use first frame
            #     to bootstrap the model (no uncertainty check).
            if self.model.training_size == 0:
                t_up0 = time.perf_counter()
                init_frame = True
                init_candidate_mask = candidate_mask
                if isinstance(self.deal_cfg.initial_atoms, list):
                    init_atoms = self._filter_atoms_by_mask(
                        self.deal_cfg.initial_atoms, init_candidate_mask
                    )
                else:
                    unique_species = sorted(set(atoms.get_atomic_numbers().tolist()))
                    idx_species = {sp: [] for sp in unique_species}
                    for sp in unique_species:
                        idx_species[sp] = [
                            i for i, at in enumerate(atoms) if at.number == sp
                        ]
                        idx_species[sp] = self._filter_atoms_by_mask(
                            idx_species[sp], init_candidate_mask
                        )
                        self.rng.shuffle(
                            idx_species[sp]
                        )  # indices of this species randomly shuffle
                    unique_species = [
                        sp for sp in unique_species if len(idx_species[sp]) > 0
                    ]

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
                if init_candidate_mask is not None and len(init_atoms) == 0:
                    ase_frame.arrays["atomic_uncertainty"] = np.full(
                        len(ase_frame),
                        -1.0,
                        dtype=float,
                    )
                    ase_frame.info["max_atomic_uncertainty"] = -1.0
                    if self.deal_cfg.save_full_trajectory:
                        t_io0 = time.perf_counter()
                        self._store_full_trajectory_frame(step, ase_frame)
                        self.timers["io_write"] += time.perf_counter() - t_io0
                    self._debug(
                        step,
                        decision="skip",
                        reason="no_bootstrap_atoms_after_mask",
                        candidates=f"{int(np.count_nonzero(init_candidate_mask))}/{len(atoms)}",
                    )
                    elapsed = time.perf_counter() - self.timers["start"]
                    step_time = time.perf_counter() - step_start
                    self._print_progress(step, elapsed, step_time)
                    continue
                self.model.update(
                    atoms=atoms,
                    train_atoms=init_atoms,
                    dft_forces=dft_forces,
                    dft_energy=dft_energy,
                    dft_stress=dft_stress,
                    force_only=self.deal_cfg.force_only,
                    train_hyperparameters=self.deal_cfg.train_hyps,
                    local_uncertainty_only=self._use_local_uncertainty_fast_path(),
                )
                bootstrap_time = time.perf_counter() - t_up0
                self.timers["update"] += bootstrap_time
                self._debug(
                    step,
                    event="bootstrap",
                    candidates=(
                        f"{int(np.count_nonzero(init_candidate_mask))}/{len(atoms)}"
                        if init_candidate_mask is not None
                        else f"{len(atoms)}/{len(atoms)}"
                    ),
                    train_atoms=len(init_atoms),
                    train_indices=self._format_atom_indices(init_atoms),
                    update_s=f"{bootstrap_time:.3f}",
                )

            # 3) Predict with SGP and compute uncertainties
            t_pred0 = time.perf_counter()
            candidate_atoms = (
                np.flatnonzero(candidate_mask).astype(int).tolist()
                if candidate_mask is not None
                else None
            )
            atomic_uncertainty = self.model.predict_uncertainty(
                atoms, candidate_atoms=candidate_atoms
            )
            atomic_uncertainty = self._apply_candidate_mask(
                atomic_uncertainty, candidate_mask
            )
            ase_frame.arrays["atomic_uncertainty"] = atomic_uncertainty
            ase_frame.info["max_atomic_uncertainty"] = (
                float(np.nanmax(atomic_uncertainty))
                if np.any(np.isfinite(atomic_uncertainty))
                else float("nan")
            )

            if self.deal_cfg.save_full_trajectory:
                t_io0 = time.perf_counter()
                self._store_full_trajectory_frame(step, ase_frame)
                self.timers["io_write"] += time.perf_counter() - t_io0

            max_atom_added = (
                self.deal_cfg.max_atoms_added
                if isinstance(self.deal_cfg.max_atoms_added, int)
                else int(np.ceil(self.deal_cfg.max_atoms_added * len(atoms)))
            )
            if candidate_mask is None:
                std_in_bound, target_atoms = self.model.select_atoms_by_uncertainty(
                    atoms,
                    threshold=self.deal_cfg.threshold,
                    update_threshold=self.deal_cfg.update_threshold,
                )
            else:
                std_in_bound, target_atoms = self.model.select_atoms_by_uncertainty(
                    atoms,
                    threshold=self.deal_cfg.threshold,
                    update_threshold=self.deal_cfg.update_threshold,
                    candidate_mask=candidate_mask,
                )
            if (
                0 < max_atom_added < len(target_atoms)
            ):  # only keep up to max_atoms_added atoms
                target_atoms = target_atoms[-max_atom_added:]
            predict_time = time.perf_counter() - t_pred0
            self.timers["predict"] += predict_time

            steps_since_last = step - self.last_dft_step
            update_time = 0.0

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
                self.model.update(
                    atoms=atoms,
                    train_atoms=list(target_atoms),
                    dft_forces=dft_forces,
                    dft_energy=dft_energy,
                    dft_stress=dft_stress,
                    force_only=self.deal_cfg.force_only,
                    train_hyperparameters=self.deal_cfg.train_hyps,
                    local_uncertainty_only=self._use_local_uncertainty_fast_path(),
                )
                update_time = time.perf_counter() - t_up0
                self.timers["update"] += update_time
                decision = "select"
                reason = "uncertainty_above_threshold"
            elif (
                std_in_bound and init_frame
            ):  # In initial frame case, store selected frame even if stds are okay
                self.last_dft_step = step
                self._store_selected_frame(step, ase_frame, target_atoms=init_atoms)
                decision = "select"
                reason = "bootstrap_frame"
            elif not std_in_bound:
                decision = "defer"
                reason = "min_steps_with_model"
            else:
                decision = "skip"
                reason = "uncertainty_within_threshold"

            if self.deal_cfg.debug:
                candidate_indices = (
                    np.flatnonzero(candidate_mask)
                    if candidate_mask is not None
                    else np.arange(len(atoms))
                )
                # Selection uses the largest component of atoms.stds, whereas
                # exported atomic_uncertainty may be a vector norm. Report the
                # same quantity that actually drives the threshold decision.
                selection_stds = np.asarray(
                    [
                        float(np.max(atoms.stds[index]))
                        for index in candidate_indices
                    ]
                )
                finite_selection_stds = selection_stds[
                    np.isfinite(selection_stds)
                ]
                max_selection_std = (
                    float(np.max(finite_selection_stds))
                    if len(finite_selection_stds)
                    else float("nan")
                )
                atoms_above_update = int(
                    np.count_nonzero(
                        finite_selection_stds > self.deal_cfg.update_threshold
                    )
                )
                selected_atoms = (
                    target_atoms + init_atoms if init_frame else target_atoms
                )
                self._debug(
                    step,
                    decision=decision,
                    reason=reason,
                    max_selection_std=f"{max_selection_std:.6g}",
                    threshold=f"{self.deal_cfg.threshold:.6g}",
                    update_threshold=f"{self.deal_cfg.update_threshold:.6g}",
                    candidates=f"{len(candidate_indices)}/{len(atoms)}",
                    above_update=atoms_above_update,
                    target_cap=max_atom_added,
                    selected_atoms=(
                        len(selected_atoms) if decision == "select" else 0
                    ),
                    train_atoms=len(target_atoms) if decision == "select" else 0,
                    selected_indices=self._format_atom_indices(
                        selected_atoms if decision == "select" else []
                    ),
                    steps_since_update=(
                        "bootstrap" if init_frame else steps_since_last
                    ),
                    min_steps=self.deal_cfg.min_steps_with_model,
                    predict_s=f"{predict_time:.3f}",
                    update_s=f"{update_time:.3f}",
                )
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
            self.model.write(f"{self.deal_cfg.output_prefix}_sgp.json")
            self.timers["io_write"] += time.perf_counter() - t_io0
            if self.deal_cfg.verbose:
                print(
                    f"[OUTPUT] Saved GP model to {self.deal_cfg.output_prefix}_sgp.json"
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

    def _store_selected_frame(self, step: int, ase_frame, target_atoms: Sequence[int]):
        """Keep a copy of the selected ASE frame for writing to XYZ."""
        sel = ase_frame.copy()
        sel.info["step"] = step
        if "original_frame" not in sel.info:
            sel.info["original_frame"] = step
        sel.info["threshold"] = self.deal_cfg.threshold
        sel.info["target_atoms"] = np.array(target_atoms, dtype=int)
        t_io0 = time.perf_counter()
        write(f"{self.deal_cfg.output_prefix}_selected.xyz", sel, append=(self.dft_count != 0))
        self.timers["io_write"] += time.perf_counter() - t_io0
        self.selected_frames.append(sel)
        self.dft_count += 1
