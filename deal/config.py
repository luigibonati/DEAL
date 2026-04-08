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
            self.atoms_list = list(self.atoms_list)
            for i, atoms in enumerate(self.atoms_list):
                if not isinstance(atoms, Atoms):
                    raise TypeError(
                        f"DataConfig.atoms_list[{i}] is not an ASE Atoms object."
                    )

            # Remember global indices for downstream filtering/selection.
            if not all("original_frame" in atoms.info for atoms in self.atoms_list):
                for i, atoms in enumerate(self.atoms_list):
                    atoms.info["original_frame"] = i

            if self.shuffle:
                rng = np.random.default_rng(self.seed)
                rng.shuffle(self.atoms_list)


@dataclass
class DEALConfig:
    # --- selection parameters ---
    threshold: float = 1.0
    update_threshold: Optional[float] = None

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
        if not isinstance(self.threshold, Real):
            raise TypeError(
                f"'threshold' must be a scalar float/int, got {type(self.threshold)}."
            )
        self.threshold = float(self.threshold)

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
