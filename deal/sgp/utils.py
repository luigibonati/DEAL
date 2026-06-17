from __future__ import annotations

from json import JSONEncoder
from math import inf
from typing import List

import numpy as np


class NumpyEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return JSONEncoder.default(self, obj)


def get_max_cutoff(cell) -> float:
    cell = np.asarray(cell, dtype=float)
    if cell.shape != (3, 3):
        return 0.0
    return 0.5 * min(np.linalg.norm(vector) for vector in cell)


def is_std_in_bound(
    std_tolerance: float,
    noise: float,
    structure,
    max_atoms_added: int = inf,
    update_style: str = "add_n",
    update_threshold: float = None,
) -> tuple[bool, List[int]]:
    if std_tolerance == 0:
        return True, [-1]
    if std_tolerance > 0:
        threshold = std_tolerance * np.abs(noise)
    else:
        threshold = np.abs(std_tolerance)

    nat = len(structure)
    max_stds = np.zeros(nat)
    for atom, std in enumerate(structure.stds):
        max_stds[atom] = np.max(std)
    stds_sorted = np.argsort(max_stds)

    if update_style == "add_n":
        target_atoms = list(stds_sorted[-max_atoms_added:])
    elif update_style == "threshold":
        target_atoms = [
            atom_index
            for atom_index in stds_sorted
            if max_stds[atom_index] > update_threshold
        ]
    else:
        raise NotImplementedError(f"Unknown update_style: {update_style}")

    if max_stds[stds_sorted[-1]] > threshold:
        return False, target_atoms
    return True, [-1]
