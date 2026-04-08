from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from numbers import Real

from ase import Atoms
from ase.io import iread
import numpy as np


@dataclass
class DataConfig:
    # --- data / trajectory ---
    files: Optional[str | List[str]] = None
    images: Optional[List[Atoms]] = None
    format: Optional[str] = None
    index: str = ":"  # ASE selection string
    colvar: Optional[List[str]] = None
    shuffle: bool = False
    seed: int = 24

    def __repr__(self) -> str:
        n_frames = len(self.images) if self.images is not None else 0
        formulas = set()
        if self.images:
            for atoms in self.images:
                try:
                    formulas.add(atoms.get_chemical_formula())
                except Exception:
                    pass
        if formulas:
            formula_repr = "{" + ", ".join(repr(f) for f in sorted(formulas)) + "}"
        else:
            formula_repr = "set()"
        return (
            "DataConfig("
            f"files={self.files}, "
            f"n_frames={n_frames}, "
            f"formula={formula_repr}, "
            f"format={self.format!r}, "
            f"index={self.index!r}, "
            f"colvar={self.colvar!r}, "
            f"shuffle={self.shuffle}, "
            f"seed={self.seed}"
            ")"
        )

    def __post_init__(self):
        if isinstance(self.files, str):
            self.files = [self.files]

        if self.files is not None and len(self.files) == 0:
            self.files = None
        if self.images is not None and len(self.images) == 0:
            self.images = None

        if self.files is not None and self.images is not None:
            raise ValueError(
                "Provide exactly one of 'files' or 'images' in DataConfig."
            )
        if self.files is None and self.images is None:
            raise ValueError(
                "Provide exactly one of 'files' or 'images' in DataConfig."
            )

        if self.files is not None:
            loaded_atoms = []
            for fname in self.files:
                loaded_atoms.extend(
                    list(iread(fname, index=self.index, format=self.format))
                )
            if len(loaded_atoms) == 0:
                raise ValueError("DataConfig does not contain any frames.")
            self.images = loaded_atoms
            self.files = None

        if self.images is not None:
            self.images = list(self.images)
            for i, atoms in enumerate(self.images):
                if not isinstance(atoms, Atoms):
                    raise TypeError(
                        f"DataConfig.images[{i}] is not an ASE Atoms object."
                    )

            # Remember global indices for downstream filtering/selection.
            if not all("original_frame" in atoms.info for atoms in self.images):
                for i, atoms in enumerate(self.images):
                    atoms.info["original_frame"] = i

            if self.shuffle:
                rng = np.random.default_rng(self.seed)
                rng.shuffle(self.images)


@dataclass
class DEALConfig:
    # --- selection parameters ---
    threshold: float = 1.0
    update_threshold: Optional[float] = None
    max_selected: Optional[int] = None
    max_iterations: int = 10
    threshold_factor: float = 0.75

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
    verbose: bool | str = False  # allowed values: true/false/"debug" (default: false)
    save_gp: bool = False
    save_full_trajectory: bool = False
    debug: bool = False  # internal debug flag

    # --- Validation of parameters ---
    def __post_init__(self):
        if not isinstance(self.threshold, Real):
            raise TypeError(
                f"'threshold' must be a scalar float/int, got {type(self.threshold)}."
            )
        self.threshold = float(self.threshold)

        if self.max_selected is not None:
            if isinstance(self.max_selected, bool) or not isinstance(
                self.max_selected, int
            ):
                raise TypeError(
                    "'max_selected' must be an int > 0 or None, "
                    f"got {type(self.max_selected)}."
                )
            if self.max_selected <= 0:
                raise ValueError(
                    f"'max_selected' must be > 0, got {self.max_selected}."
                )

        if isinstance(self.max_iterations, bool) or not isinstance(
            self.max_iterations, int
        ):
            raise TypeError(
                f"'max_iterations' must be an int >= 1, got {type(self.max_iterations)}."
            )
        if self.max_iterations < 1:
            raise ValueError(
                f"'max_iterations' must be >= 1, got {self.max_iterations}."
            )

        self.threshold_factor = float(self.threshold_factor)
        if not (0 < self.threshold_factor < 1):
            raise ValueError(
                f"'threshold_factor' must be in (0, 1), got {self.threshold_factor}."
            )

        if self.update_threshold is not None and not isinstance(
            self.update_threshold, Real
        ):
            raise TypeError(
                "'update_threshold' must be a scalar float/int or None, "
                f"got {type(self.update_threshold)}."
            )

        # --- Default update_threshold ---
        if self.update_threshold is None:
            self.update_threshold = 0.8 * self.threshold
        else:
            self.update_threshold = float(self.update_threshold)

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
