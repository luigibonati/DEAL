import argparse
import sys
import yaml

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

    return parser.parse_args()


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

    # Check file
    try:
        cfg_dict["data"]["files"][0]
    except KeyError:
        print(
            "[ERROR] No input trajectory specified. Please provide a trajectory file (-f/--file) or a YAML config file (-c/--config)."
        )
        sys.exit(1)

    # Build configs and run
    data_cfg = DataConfig(**cfg_dict["data"])
    deal_cfg = DEALConfig(**cfg_dict["deal"])
    flare_cfg = FlareConfig(**cfg_dict["flare"])

    # Handle multiple thresholds
    if isinstance(deal_cfg.threshold, list):
        prefix = deal_cfg.output_prefix
        for threshold, update_thresh in zip(
            deal_cfg.threshold, deal_cfg.update_threshold
        ):
            print("[DEAL] Running with threshold:", threshold)
            deal_cfg.threshold = threshold
            deal_cfg.update_threshold = update_thresh
            deal_cfg.output_prefix = f"{prefix}_{str(threshold)}"
            DEAL(data_cfg, deal_cfg, flare_cfg).run()
    else:
        DEAL(data_cfg, deal_cfg, flare_cfg).run()
