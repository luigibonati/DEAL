from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from .config import DataConfig
from .preprocessing import TrajectoryMasker, write_preprocessed_trajectory


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create a per-atom DEAL mask from a trajectory array."
    )
    parser.add_argument(
        "-c", "--config", help="YAML file containing data and preprocessing blocks."
    )
    parser.add_argument("-f", "--file", help="Input trajectory.")
    parser.add_argument("-o", "--output", help="Output trajectory.")
    parser.add_argument(
        "-k", "--key", help="Input per-atom array used to build the mask."
    )
    parser.add_argument(
        "-t",
        "--mask-threshold",
        dest="mask_threshold",
        type=float,
        help="Fixed atom threshold. Omit to use automatic notebook thresholds.",
    )
    parser.add_argument(
        "--mask-upper-threshold",
        dest="mask_upper_threshold",
        type=float,
        help="Upper threshold for between/outside modes.",
    )
    parser.add_argument(
        "--mode", choices=["above", "below", "between", "outside"]
    )
    parser.add_argument("--mask-key", help="Output per-atom mask array name.")
    parser.add_argument("--index", help="ASE frame index expression.")
    parser.add_argument("--format", help="ASE input file format.")
    parser.add_argument("--output-format", help="ASE output file format.")
    parser.add_argument(
        "--plot",
        nargs="?",
        const=True,
        help="Write selection plot, disable with --plot false, or pass a filename.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=None,
        help="Replace an existing preprocessed trajectory.",
    )
    return parser.parse_args()


def _parse_plot(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {"true", "yes", "1"}:
        return True
    if value.lower() in {"false", "no", "0"}:
        return False
    return value


def _default_output(input_file: str) -> str:
    path = Path(input_file)
    return str(path.with_name(f"{path.stem}_preprocessed{path.suffix or '.xyz'}"))


def main() -> None:
    args = parse_args()
    config = {}
    if args.config is not None:
        with open(args.config, "r") as handle:
            config = yaml.safe_load(handle) or {}

    data = dict(config.get("data", {}))
    preprocessing = dict(config.get("preprocessing", {}))

    if args.file is not None:
        data["files"] = [args.file]
    if args.index is not None:
        data["index"] = args.index
    if args.format is not None:
        data["format"] = args.format

    cli_preprocessing = {
        "key": args.key,
        "mask_threshold": args.mask_threshold,
        "mask_upper_threshold": args.mask_upper_threshold,
        "mode": args.mode,
        "mask_key": args.mask_key,
        "output": args.output,
        "output_format": args.output_format,
        "overwrite": args.overwrite,
    }
    if args.plot is not None:
        cli_preprocessing["plot"] = _parse_plot(args.plot)
    preprocessing.update(
        {key: value for key, value in cli_preprocessing.items() if value is not None}
    )

    files = data.get("files")
    if isinstance(files, str):
        files = [files]
        data["files"] = files
    if not files:
        raise ValueError("deal-mask requires data.files in YAML or -f/--file.")
    if "key" not in preprocessing:
        raise ValueError("deal-mask requires preprocessing.key in YAML or -k/--key.")

    output = preprocessing.pop("output", None) or _default_output(files[0])
    output_format = preprocessing.pop("output_format", None)
    overwrite = preprocessing.pop("overwrite", False)
    data_cfg = DataConfig(**data)
    masker = TrajectoryMasker(**preprocessing)
    summary = masker.apply_to_trajectory(data_cfg.images or [])
    written = write_preprocessed_trajectory(
        data_cfg.images or [],
        output,
        file_format=output_format,
        overwrite=overwrite,
    )

    print(
        "[deal-mask] "
        f"frames={summary.n_frames} "
        f"atoms={summary.n_atoms} "
        f"selected_atoms={summary.n_selected_atoms} "
        f"frames_with_selection={summary.n_frames_with_selection} "
        f"selected_fraction={summary.selected_fraction:.6f}"
    )
    if summary.lower_threshold is not None:
        print(
            "[deal-mask] "
            f"lower_threshold={summary.lower_threshold:.6g} "
            f"upper_threshold={summary.upper_threshold:.6g}"
        )
    if summary.plot_file is not None:
        print(f"[deal-mask] plot={summary.plot_file}")
    status = "saved" if written else "kept existing"
    print(f"[deal-mask] trajectory={output} ({status})")


if __name__ == "__main__":
    main()
