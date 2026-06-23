from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from ase.io import iread, write


MaskMode = Literal["above", "below", "between", "outside"]


def write_preprocessed_trajectory(
    images: Iterable,
    output_file: str,
    file_format: str | None = None,
    overwrite: bool = False,
) -> bool:
    """Write masked frames, returning False when an existing file is preserved."""
    output = Path(output_file)
    if output.exists() and not overwrite:
        return False
    output.parent.mkdir(parents=True, exist_ok=True)
    write(output, list(images), format=file_format)
    return True


@dataclass
class MaskSummary:
    n_frames: int = 0
    n_atoms: int = 0
    n_selected_atoms: int = 0
    n_frames_with_selection: int = 0
    lower_threshold: float | None = None
    upper_threshold: float | None = None
    plot_file: str | None = None

    @property
    def selected_fraction(self) -> float:
        if self.n_atoms == 0:
            return 0.0
        return self.n_selected_atoms / self.n_atoms


@dataclass
class TrajectoryMasker:
    """Prepare a trajectory with a reusable per-atom mask array."""

    key: str
    mask_threshold: float | None = None
    mask_key: str = "deal_mask"
    mode: MaskMode = "above"
    mask_upper_threshold: float | None = None
    mask_fraction: float = 0.3
    lower_factor: float = 1.1
    upper_factor: float = 4.0
    plot: bool | str = True
    selected_value: int = 1
    rejected_value: int = 0

    def __post_init__(self) -> None:
        if not self.key:
            raise ValueError("'key' cannot be empty.")
        if not self.mask_key:
            raise ValueError("'mask_key' cannot be empty.")
        if self.mask_threshold is not None:
            self.mask_threshold = float(self.mask_threshold)
        if self.mask_upper_threshold is not None:
            self.mask_upper_threshold = float(self.mask_upper_threshold)
        if self.mask_threshold is not None and self.mode in {"between", "outside"}:
            if self.mask_upper_threshold is None:
                raise ValueError(
                    f"'{self.mode}' mode requires mask_upper_threshold."
                )
            if self.mask_upper_threshold <= self.mask_threshold:
                raise ValueError(
                    "'mask_upper_threshold' must be greater than 'mask_threshold'."
                )
        if not 0 <= self.mask_fraction <= 1:
            raise ValueError("'mask_fraction' must be between 0 and 1.")
        if self.lower_factor <= 0 or self.upper_factor <= self.lower_factor:
            raise ValueError(
                "Automatic threshold factors must satisfy 0 < lower_factor < upper_factor."
            )

    def mask_values(self, values) -> np.ndarray:
        if self.mask_threshold is None:
            raise ValueError(
                "Automatic thresholds require apply_to_trajectory(), not mask_values()."
            )
        values = np.asarray(values, dtype=float)
        if self.mode == "above":
            mask = values > self.mask_threshold
        elif self.mode == "below":
            mask = values < self.mask_threshold
        elif self.mode == "between":
            if self.mask_upper_threshold is None:
                raise ValueError("'between' mode requires mask_upper_threshold.")
            mask = (values > self.mask_threshold) & (
                values < self.mask_upper_threshold
            )
        elif self.mode == "outside":
            if self.mask_upper_threshold is None:
                raise ValueError("'outside' mode requires mask_upper_threshold.")
            mask = (values < self.mask_threshold) | (
                values > self.mask_upper_threshold
            )
        else:
            raise ValueError(f"Unknown mask mode: {self.mode}")
        return np.where(mask, self.selected_value, self.rejected_value).astype(int)

    def apply_to_atoms(self, atoms):
        if self.mask_threshold is None:
            raise ValueError(
                "Automatic thresholds require apply_to_trajectory(), not apply_to_atoms()."
            )
        values = self._atom_values(atoms)
        mask = self.mask_values(values)
        self._store_mask(atoms, mask != self.rejected_value, self.mask_threshold)
        if self.mask_upper_threshold is not None:
            atoms.info[f"{self.mask_key}_upper_threshold"] = float(
                self.mask_upper_threshold
            )
        return atoms

    def apply_to_trajectory(self, images: Iterable) -> MaskSummary:
        """Add the mask to an in-memory trajectory and return aggregate counts."""
        images = list(images)
        if not images:
            raise ValueError("No frames were provided for preprocessing.")

        summary = MaskSummary()
        all_values = [self._atom_values(atoms) for atoms in images]
        frame_maxima = np.asarray([np.nanmax(values) for values in all_values])

        if self.mask_threshold is None:
            mean_max = float(np.nanmean(frame_maxima))
            summary.lower_threshold = self.lower_factor * mean_max
            summary.upper_threshold = self.upper_factor * mean_max

        for atoms, values, frame_max in zip(images, all_values, frame_maxima):
            if self.mask_threshold is None:
                frame_selected = (
                    summary.lower_threshold < frame_max < summary.upper_threshold
                )
                atom_threshold = self.mask_fraction * frame_max
                mask = (
                    values > atom_threshold
                    if frame_selected
                    else np.zeros(len(atoms), dtype=bool)
                )
                self._store_mask(atoms, mask, atom_threshold)
            else:
                self.apply_to_atoms(atoms)
            selected = int(np.count_nonzero(atoms.arrays[self.mask_key]))
            summary.n_frames += 1
            summary.n_atoms += len(atoms)
            summary.n_selected_atoms += selected
            if selected > 0:
                summary.n_frames_with_selection += 1

        if self.plot is not False:
            summary.plot_file = self._plot_selection(
                images, all_values, frame_maxima, summary
            )
        return summary

    def _atom_values(self, atoms) -> np.ndarray:
        if self.key not in atoms.arrays:
            raise RuntimeError(
                f"Frame is missing per-atom array '{self.key}'. "
                f"Available arrays: {sorted(atoms.arrays.keys())}"
            )
        values = np.asarray(atoms.arrays[self.key], dtype=float)
        if values.shape[0] != len(atoms):
            raise RuntimeError(
                f"Array '{self.key}' has incompatible shape {values.shape}; "
                f"expected first dimension {len(atoms)}."
            )
        if values.ndim > 1:
            values = np.nanmax(values.reshape(len(atoms), -1), axis=1)
        if values.size == 0:
            raise RuntimeError("Preprocessing does not support empty frames.")
        if not np.all(np.isfinite(values)):
            raise RuntimeError(
                f"Array '{self.key}' contains non-finite uncertainty values."
            )
        return values

    def _store_mask(self, atoms, mask, threshold: float) -> None:
        atoms.arrays[self.mask_key] = np.where(
            mask, self.selected_value, self.rejected_value
        ).astype(int)
        atoms.info[f"{self.mask_key}_count"] = int(np.count_nonzero(mask))
        atoms.info[f"{self.mask_key}_source"] = self.key
        atoms.info[f"{self.mask_key}_threshold"] = float(threshold)

    def _plot_selection(self, images, all_values, frame_maxima, summary) -> str:
        import matplotlib.pyplot as plt

        output = (
            Path("preprocessing_selection.png")
            if self.plot is True
            else Path(str(self.plot))
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        atom_values = np.concatenate(all_values)
        selected_atoms = np.concatenate(
            [
                values[np.asarray(atoms.arrays[self.mask_key], dtype=bool)]
                for atoms, values in zip(images, all_values)
            ]
        )
        selected_frames = frame_maxima[
            [np.any(atoms.arrays[self.mask_key]) for atoms in images]
        ]
        upper_bin = max(float(np.max(atom_values)), float(np.max(frame_maxima)))
        if upper_bin <= 0:
            upper_bin = 1.0
        bins = np.linspace(0, upper_bin, 100)
        total_atoms = atom_values.size
        total_masked_atoms = selected_atoms.size
        atom_percentage = 100 * total_masked_atoms / total_atoms
        frame_percentage = 100 * selected_frames.size / len(images)

        with plt.rc_context({"font.family": "monospace"}):
            fig, (ax_atoms, ax_frames) = plt.subplots(
                2, 1, figsize=(10, 8), sharex=True
            )
            ax_atoms.hist(
                atom_values,
                bins=bins,
                alpha=0.6,
                label="All atoms",
            )
            ax_atoms.hist(
                selected_atoms,
                bins=bins,
                alpha=0.6,
                label=(
                    f"Selected atoms [{total_masked_atoms}/{total_atoms} "
                    f"({atom_percentage:.1f}%)]"
                ),
            )
            ax_atoms.set_ylabel("Atomic environments")
            ax_atoms.set_yscale("symlog")
            ax_atoms.legend()

            ax_frames.hist(
                frame_maxima,
                bins=bins,
                alpha=0.6,
                label="All frames (max uncertainties per frame)",
            )
            ax_frames.hist(
                selected_frames,
                bins=bins,
                alpha=0.6,
                label=(
                    f"Selected frames [{selected_frames.size}/{len(images)} "
                    f"({frame_percentage:.1f}%)]"
                ),
            )
            if summary.lower_threshold is not None:
                ax_frames.axvline(
                    summary.lower_threshold,
                    color="green",
                    linestyle="dashed",
                    linewidth=2,
                    label=f"Lower threshold ({summary.lower_threshold:.3f})",
                )
                ax_frames.axvline(
                    summary.upper_threshold,
                    color="red",
                    linestyle="dashed",
                    linewidth=2,
                    label=f"Upper threshold ({summary.upper_threshold:.3f})",
                )
            ax_frames.set_xlabel(f'Uncertainty "{self.key}"')
            ax_frames.set_ylabel("Frames")
            ax_frames.legend()
            fig.suptitle(f'Preprocessing selection for "{self.key}"')
            fig.tight_layout()
            fig.savefig(output, dpi=150)
            plt.close(fig)
        return str(output)

    def run(
        self,
        input_file: str,
        output_file: str,
        index: str = ":",
        file_format: str | None = None,
        overwrite: bool = False,
    ) -> MaskSummary:
        images = list(iread(input_file, index=index, format=file_format))
        summary = self.apply_to_trajectory(images)
        write_preprocessed_trajectory(
            images,
            output_file,
            file_format=file_format,
            overwrite=overwrite,
        )
        return summary
