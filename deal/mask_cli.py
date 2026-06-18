from __future__ import annotations

import argparse

from .preprocessing import TrajectoryMasker


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create a per-atom DEAL mask from a trajectory array."
    )
    parser.add_argument("-f", "--file", required=True, help="Input trajectory.")
    parser.add_argument("-o", "--output", required=True, help="Output trajectory.")
    parser.add_argument(
        "-k",
        "--key",
        required=True,
        help="Input per-atom array used to build the mask.",
    )
    parser.add_argument(
        "-t",
        "--threshold",
        required=True,
        type=float,
        help="Lower threshold for the mask.",
    )
    parser.add_argument(
        "--upper-threshold",
        type=float,
        default=None,
        help="Upper threshold for between/outside modes.",
    )
    parser.add_argument(
        "--mode",
        choices=["above", "below", "between", "outside"],
        default="above",
        help="Thresholding mode.",
    )
    parser.add_argument(
        "--mask-key",
        default="deal_mask",
        help="Output per-atom mask array name.",
    )
    parser.add_argument(
        "--index",
        default=":",
        help="ASE frame index expression.",
    )
    parser.add_argument(
        "--format",
        default=None,
        help="ASE file format.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    masker = TrajectoryMasker(
        key=args.key,
        threshold=args.threshold,
        mask_key=args.mask_key,
        mode=args.mode,
        upper_threshold=args.upper_threshold,
    )
    summary = masker.run(
        input_file=args.file,
        output_file=args.output,
        index=args.index,
        file_format=args.format,
    )
    print(
        "[deal-mask] "
        f"frames={summary.n_frames} "
        f"atoms={summary.n_atoms} "
        f"selected_atoms={summary.n_selected_atoms} "
        f"frames_with_selection={summary.n_frames_with_selection} "
        f"selected_fraction={summary.selected_fraction:.6f}"
    )


if __name__ == "__main__":
    main()
