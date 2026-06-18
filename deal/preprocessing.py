from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from ase.io import iread, write


MaskMode = Literal["above", "below", "between", "outside"]


@dataclass
class MaskSummary:
    n_frames: int = 0
    n_atoms: int = 0
    n_selected_atoms: int = 0
    n_frames_with_selection: int = 0

    @property
    def selected_fraction(self) -> float:
        if self.n_atoms == 0:
            return 0.0
        return self.n_selected_atoms / self.n_atoms


@dataclass
class TrajectoryMasker:
    """Prepare a trajectory with a reusable per-atom mask array."""

    key: str
    threshold: float
    mask_key: str = "deal_mask"
    mode: MaskMode = "above"
    upper_threshold: float | None = None
    selected_value: int = 1
    rejected_value: int = 0

    def mask_values(self, values) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        if self.mode == "above":
            mask = values > self.threshold
        elif self.mode == "below":
            mask = values < self.threshold
        elif self.mode == "between":
            if self.upper_threshold is None:
                raise ValueError("'between' mode requires upper_threshold.")
            mask = (values > self.threshold) & (values < self.upper_threshold)
        elif self.mode == "outside":
            if self.upper_threshold is None:
                raise ValueError("'outside' mode requires upper_threshold.")
            mask = (values < self.threshold) | (values > self.upper_threshold)
        else:
            raise ValueError(f"Unknown mask mode: {self.mode}")
        return np.where(mask, self.selected_value, self.rejected_value).astype(int)

    def apply_to_atoms(self, atoms):
        if self.key not in atoms.arrays:
            raise RuntimeError(
                f"Frame is missing per-atom array '{self.key}'. "
                f"Available arrays: {sorted(atoms.arrays.keys())}"
            )

        values = np.asarray(atoms.arrays[self.key])
        if values.shape[0] != len(atoms):
            raise RuntimeError(
                f"Array '{self.key}' has incompatible shape {values.shape}; "
                f"expected first dimension {len(atoms)}."
            )
        if values.ndim > 1:
            values = np.nanmax(values.reshape(len(atoms), -1), axis=1)

        mask = self.mask_values(values)
        atoms.arrays[self.mask_key] = mask
        atoms.info[f"{self.mask_key}_count"] = int(np.count_nonzero(mask))
        atoms.info[f"{self.mask_key}_source"] = self.key
        atoms.info[f"{self.mask_key}_threshold"] = float(self.threshold)
        if self.upper_threshold is not None:
            atoms.info[f"{self.mask_key}_upper_threshold"] = float(
                self.upper_threshold
            )
        return atoms

    def run(
        self,
        input_file: str,
        output_file: str,
        index: str = ":",
        file_format: str | None = None,
    ) -> MaskSummary:
        summary = MaskSummary()
        first = True

        for atoms in iread(input_file, index=index, format=file_format):
            atoms = self.apply_to_atoms(atoms)
            selected = int(np.count_nonzero(atoms.arrays[self.mask_key]))
            summary.n_frames += 1
            summary.n_atoms += len(atoms)
            summary.n_selected_atoms += selected
            if selected > 0:
                summary.n_frames_with_selection += 1
            write(output_file, atoms, append=not first, format=file_format)
            first = False

        if summary.n_frames == 0:
            raise ValueError("No frames were read from the input trajectory.")
        return summary
