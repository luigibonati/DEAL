import argparse
import os
import sys

from .utils import create_chemiscope_input


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create a chemiscope input file from DEAL selected structures."
    )
    parser.add_argument(
        "-p",
        "--prefix",
        dest="prefix",
        help="DEAL output prefix (uses <prefix>_selected.xyz and writes <prefix>_chemiscope.json.gz by default).",
    )
    parser.add_argument(
        "-t",
        "--trajectory",
        dest="trajectory",
        help="Path to the selected trajectory (e.g. deal_selected.xyz).",
    )
    parser.add_argument(
        "-o",
        "--output",
        dest="output",
        help="Output chemiscope filename (default: inferred from prefix or trajectory).",
    )
    parser.add_argument(
        "--colvar",
        dest="colvar",
        help="Optional COLVAR file to include in chemiscope properties.",
    )
    parser.add_argument(
        "--cv",
        action="append",
        dest="cvs",
        default=None,
        help="Property filter(s) to include from trajectory/COLVAR. Can be passed multiple times. Default: '*'.",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        dest="quiet",
        help="Disable informational output.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.trajectory is None and args.prefix is None:
        print(
            "[ERROR] Please provide either --trajectory or --prefix.",
            file=sys.stderr,
        )
        sys.exit(1)

    trajectory = args.trajectory
    if trajectory is None and args.prefix is not None:
        trajectory = f"{args.prefix}_selected.xyz"

    if not os.path.exists(trajectory):
        print(
            f"[ERROR] Trajectory file not found: {trajectory}",
            file=sys.stderr,
        )
        sys.exit(1)

    output = args.output
    if output is None and args.prefix is not None:
        output = f"{args.prefix}_chemiscope.json.gz"

    try:
        create_chemiscope_input(
            trajectory=trajectory,
            filename=output,
            colvar=args.colvar,
            cvs=args.cvs if args.cvs is not None else ["*"],
            verbose=not args.quiet,
        )
    except Exception as exc:
        print(f"[ERROR] Failed to create chemiscope file: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
