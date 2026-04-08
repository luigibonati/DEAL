import argparse
import sys
import yaml
import numpy as np

from .config import DataConfig, DEALConfig, FlareConfig
from .core import DEAL


def parse_args():
    parser = argparse.ArgumentParser(
        description="DEAL selector: read a YAML config or specify a trajectory and (optionally) a threshold."
    )

    parser.add_argument(
        "-c", "--config", dest="config", help="YAML configuration file."
    )
    parser.add_argument(
        "-f", "--file", dest="filename", help="Input trajectory file (e.g. traj.xyz)."
    )
    parser.add_argument(
        "-t",
        "--threshold",
        type=float,
        dest="threshold",
        help="GP uncertainty threshold that triggers selection.",
    )
    parser.add_argument(
        "--max-selected",
        type=int,
        dest="max_selected",
        help=(
            "Target number of selected configurations."
            "Mutually exclusive with --threshold."
        ),
    )

    return parser.parse_args()


def _run_incremental_cli(
    data_cfg: DataConfig, deal_dict: dict, flare_cfg: FlareConfig
) -> None:
    max_selected = deal_dict.get("max_selected")
    max_iterations = deal_dict.get("max_iterations", 10)
    threshold_factor = deal_dict.get("threshold_factor", 0.75)

    print(f"[DEAL] Running in incremental mode with max_selected = {max_selected}.")

    if max_selected is None:
        raise ValueError("Incremental mode requires 'max_selected' to be set.")

    all_images = list(data_cfg.images or [])
    selected_frames = set()
    deal = None

    for iteration in range(max_iterations):
        remaining_images = [
            atoms
            for atoms in all_images
            if atoms.info.get("original_frame") not in selected_frames
        ]

        if len(remaining_images) == 0:
            print("\n[DEAL] Stopping incremental mode: all input frames were selected.")
            break

        deal_threshold = np.round(threshold_factor ** (iteration + 1),3)
        print(f"\n[DEAL] Iteration {iteration} (threshold: {deal_threshold})")

        run_deal_dict = dict(deal_dict)
        run_deal_dict["threshold"] = deal_threshold
        deal_cfg = DEALConfig(**run_deal_dict)
        data_iter_cfg = DataConfig(images=remaining_images, seed=data_cfg.seed)

        if iteration == 0:
            deal = DEAL(data_iter_cfg, deal_cfg, flare_cfg)
        else:
            deal.configure_run(data_cfg=data_iter_cfg, deal_cfg=deal_cfg)

        previous_selected = len(selected_frames)
        deal.run()

        selected_frames.update(
            frame.info["original_frame"]
            for frame in deal.selected_frames
            if "original_frame" in frame.info
        )
        new_selected = len(selected_frames) - previous_selected
        print(
            f"[DEAL] New selected: {new_selected} "
        )

        if len(selected_frames) >= max_selected:
            print(
                f"\n[DEAL] Stopping incremental mode: max_selected is reached."
            )
            break
    else:
        print(
            f"\n[WARNING] Reached max_iterations={max_iterations} before "
            f"reaching max_selected={max_selected}. "
            f"Current selected: {len(selected_frames)}."
        )


def main() -> None:

    print("""
888888ba   88888888b  .d888888  88        
88     8b  88        d8     88  88        
88     88  88aaaa    88aaaaa88  88        
88     88  88        88     88  88        
88     8P  88        88     88  88        
8888888P   88888888P 88     88  888888888
""")

    args = parse_args()
    cfg_dict = {}

    # Start from YAML config if provided
    if args.config is not None:
        with open(args.config, "r") as f:
            cfg_dict = yaml.safe_load(f) or {}

    # Initialize dicts if not available
    for key in ["data", "deal", "flare"]:
        if key not in cfg_dict:
            cfg_dict[key] = {}

    # Overwrite / fill from CLI options
    if args.filename is not None:
        cfg_dict["data"]["files"] = [args.filename]
    if args.threshold is not None:
        cfg_dict["deal"]["threshold"] = args.threshold
    if args.max_selected is not None:
        cfg_dict["deal"]["max_selected"] = args.max_selected

    # Check file
    try:
        cfg_dict["data"]["files"][0]
    except KeyError:
        print(
            "[ERROR] No input trajectory specified. Please provide a trajectory file (-f/--file) or a YAML config file (-c/--config)."
        )
        sys.exit(1)

    # Build data/flare configs
    data_cfg = DataConfig(**cfg_dict["data"])
    flare_cfg = FlareConfig(**cfg_dict["flare"])

    deal_dict = dict(cfg_dict["deal"])
    has_threshold = deal_dict.get("threshold", None) is not None
    has_max_selected = deal_dict.get("max_selected", None) is not None

    if has_threshold and has_max_selected:
        print(
            "[ERROR] 'threshold' and 'max_selected' are mutually exclusive. "
            "Use 'threshold' for standard mode or 'max_selected' for incremental mode."
        )
        sys.exit(1)

    if has_max_selected:
        _run_incremental_cli(data_cfg, deal_dict, flare_cfg)
        return

    # Standard mode: handle one threshold or a list of independent thresholds
    thresholds = deal_dict.get("threshold", None)

    if isinstance(thresholds, list):
        for th in thresholds:
            run_deal_dict = dict(deal_dict)
            run_deal_dict["threshold"] = th
            prefix = deal_dict.get("output_prefix", "deal")
            run_deal_dict["output_prefix"] = f"{prefix}_{th}"

            print(f"[DEAL] Running with threshold: {th}")
            deal_cfg = DEALConfig(**run_deal_dict)
            DEAL(data_cfg, deal_cfg, flare_cfg).run()
    else:
        deal_cfg = DEALConfig(**deal_dict)
        DEAL(data_cfg, deal_cfg, flare_cfg).run()
