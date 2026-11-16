from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from .cli import _apply_preprocessing, _parse_mask_arg, _resolve_default_mask
from .config import DataConfig, DEALConfig, SGPConfig
from .core import DEAL


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the atomic uncertainty of a trajectory with a saved GP "
            "(<prefix>_sgp.json). Writes the same file as save_full_trajectory."
        )
    )
    parser.add_argument(
        "-m", "--model", required=True, help="GP model written by save_gp/restart."
    )
    parser.add_argument(
        "-c",
        "--config",
        help="YAML file; its data, preprocessing and deal.mask entries are used.",
    )
    parser.add_argument("-f", "--file", help="Input trajectory.")
    parser.add_argument("--index", help="ASE frame index expression.")
    parser.add_argument("--format", help="ASE input file format.")
    parser.add_argument(
        "--mask",
        help=(
            "Candidate atom mask: false uses all atoms, true uses deal_mask, or "
            "pass a per-atom array name. Default: deal_mask if present."
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Output trajectory (default: <input>_uncertainty.xyz).",
    )
    parser.add_argument(
        "-t",
        "--threshold",
        type=float,
        help="Also report how many frames exceed this uncertainty.",
    )
    return parser.parse_args()


def _default_output(input_file: str) -> str:
    path = Path(input_file)
    return str(path.with_name(f"{path.stem}_uncertainty.xyz"))


def main() -> None:
    args = parse_args()
    cfg_dict = {}
    if args.config is not None:
        with open(args.config, "r") as handle:
            cfg_dict = yaml.safe_load(handle) or {}
    for key in ["data", "preprocessing", "deal"]:
        cfg_dict[key] = dict(cfg_dict.get(key) or {})

    data = cfg_dict["data"]
    if args.file is not None:
        data["files"] = [args.file]
    if args.index is not None:
        data["index"] = args.index
    if args.format is not None:
        data["format"] = args.format
    if args.mask is not None:
        cfg_dict["deal"]["mask"] = _parse_mask_arg(args.mask)

    files = data.get("files")
    if isinstance(files, str):
        files = [files]
    if not files:
        raise ValueError("deal-evaluate requires data.files in YAML or -f/--file.")
    # Only the frames are needed: do not shuffle them.
    data["shuffle"] = False

    _resolve_default_mask(cfg_dict)
    data_cfg = DataConfig(**data)
    _apply_preprocessing(data_cfg, cfg_dict["preprocessing"])

    mask = cfg_dict["deal"].get("mask")
    if mask is None and "deal_mask" in data_cfg.images[0].arrays:
        mask = True
    print(
        "[deal-evaluate] mask: "
        + ("all atoms" if mask in (None, False) else ("deal_mask" if mask is True else mask))
    )

    deal = DEAL(data_cfg, DEALConfig(mask=mask), SGPConfig())
    deal.load_model(args.model)
    deal.write_full_trajectory(
        filename=args.output or _default_output(files[0]),
        threshold=args.threshold,
        description=f"the GP in {args.model}",
    )


if __name__ == "__main__":
    main()
