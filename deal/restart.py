"""Checkpoint files for resuming an interrupted DEAL selection."""

from __future__ import annotations

import json
import os

from ase.io import iread, write

from .sgp.utils import NumpyEncoder

RESTART_VERSION = 1


def restart_filename(output_prefix: str) -> str:
    return f"{output_prefix}.restart"


def write_restart(filename: str, state: dict) -> None:
    """Write the restart state atomically (a killed job never leaves half a file)."""
    tmp = f"{filename}.tmp"
    with open(tmp, "w") as f:
        json.dump({"version": RESTART_VERSION, **state}, f, cls=NumpyEncoder, indent=1)
    os.replace(tmp, filename)


def read_restart(filename: str) -> dict:
    with open(filename, "r") as f:
        state = json.load(f)
    if state.get("version") != RESTART_VERSION:
        raise ValueError(
            f"Unsupported restart file version {state.get('version')!r} in {filename}."
        )
    return state


def truncate_xyz(filename: str, n_frames: int) -> list:
    """
    Keep only the first n_frames of an extxyz file and return them.

    Frames written after the last checkpoint (e.g. by a job killed before it
    could checkpoint) are dropped, so a resumed run appends without duplicates.
    """
    if n_frames == 0 or not os.path.exists(filename):
        if os.path.exists(filename):
            os.remove(filename)
        return []
    frames = []
    for atoms in iread(filename, index=":"):
        if len(frames) == n_frames:
            break
        frames.append(atoms)
    if len(frames) < n_frames:
        raise ValueError(
            f"{filename} has {len(frames)} frames, but the restart file expects {n_frames}."
        )
    write(filename, frames)
    return frames
