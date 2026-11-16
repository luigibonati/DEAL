import argparse
from copy import deepcopy
import os
import sys
import time
from typing import Optional
import yaml
import numpy as np

from .config import DataConfig, DEALConfig, SGPConfig
from .core import DEAL
from .restart import read_restart
from .preprocessing import TrajectoryMasker, write_preprocessed_trajectory


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
        "--max",
        "--max-selected",
        type=int,
        dest="max_selected",
        help=(
            "Target number of selected configurations "
            "(alias: --max-selected). "
            "Mutually exclusive with --threshold."
        ),
    )
    parser.add_argument(
        "--mask",
        dest="mask",
        help=(
            "Candidate atom mask: false uses all atoms, true uses deal_mask, "
            "or pass a per-atom array name."
        ),
    )
    parser.add_argument(
        "--preprocess-key",
        help="Per-atom array from which to derive the candidate mask in memory.",
    )
    parser.add_argument(
        "--preprocess-mask-threshold",
        type=float,
        help="Fixed atom-mask threshold used by preprocessing.",
    )
    parser.add_argument(
        "--preprocess-mode",
        choices=["above", "below", "between", "outside"],
        help="Comparison used by in-memory mask preprocessing.",
    )
    parser.add_argument(
        "--preprocess-mask-upper-threshold",
        type=float,
        help="Upper bound required by between/outside preprocessing modes.",
    )
    parser.add_argument(
        "--preprocess-plot",
        help="Write the preprocessing plot: true, false, or an output filename.",
    )
    parser.add_argument(
        "--restart",
        dest="restart",
        help=(
            "Resume from a <prefix>.restart file. The configuration stored in "
            "the file is used; -c and other options are ignored."
        ),
    )
    parser.add_argument(
        "--wall-time",
        type=float,
        dest="wall_time",
        help="Stop cleanly and write a restart checkpoint before this many seconds.",
    )

    return parser.parse_args()


def _parse_mask_arg(value: str):
    lowered = value.lower()
    if lowered in {"none", "null"}:
        return None
    if lowered in {"true", "yes", "1"}:
        return True
    if lowered in {"false", "no", "0"}:
        return False
    return value


def _parse_bool_or_filename(value: str):
    lowered = value.lower()
    if lowered in {"true", "yes", "1"}:
        return True
    if lowered in {"false", "no", "0"}:
        return False
    return value


def _resolve_default_mask(config: dict) -> None:
    """Use a configured preprocessing mask when DEAL's mask is automatic."""
    preprocessing = config["preprocessing"]
    if preprocessing and config["deal"].get("mask") is None:
        config["deal"]["mask"] = preprocessing.get("mask_key", "deal_mask")


def _apply_preprocessing(data_cfg: DataConfig, config: dict) -> None:
    """Derive a candidate mask directly on the loaded trajectory."""
    if not config:
        return

    missing = [name for name in ("key",) if name not in config]
    if missing:
        names = ", ".join(repr(name) for name in missing)
        raise ValueError(
            f"Preprocessing requires {names}."
        )

    config = dict(config)
    output = config.pop("output", None)
    overwrite = config.pop("overwrite", False)
    output_format = config.pop("output_format", None)
    selected_frames_only = config.pop("selected_frames_only", False)
    masker = TrajectoryMasker(**config)
    summary = masker.apply_to_trajectory(data_cfg.images or [])
    print(
        "[DEAL preprocessing] "
        f"frames={summary.n_frames} "
        f"atoms={summary.n_atoms} "
        f"selected_atoms={summary.n_selected_atoms} "
        f"frames_with_selection={summary.n_frames_with_selection} "
        f"selected_fraction={summary.selected_fraction:.6f}"
    )
    if summary.lower_threshold is not None:
        print(
            "[DEAL preprocessing] "
            f"lower_threshold={summary.lower_threshold:.6g} "
            f"upper_threshold={summary.upper_threshold:.6g}"
        )
    if summary.plot_file is not None:
        print(f"[DEAL preprocessing] plot={summary.plot_file}")
    if output is not None:
        written = write_preprocessed_trajectory(
            data_cfg.images or [],
            output,
            file_format=output_format,
            overwrite=overwrite,
            selected_frames_only=selected_frames_only,
            mask_key=masker.mask_key,
        )
        status = "saved" if written else "kept existing"
        print(f"[DEAL preprocessing] trajectory={output} ({status})")


def _run_incremental_cli(
    data_cfg: DataConfig,
    deal_dict: dict,
    sgp_cfg: SGPConfig,
    config: Optional[dict] = None,
    resume: Optional[dict] = None,
    wall_start: Optional[float] = None,
) -> None:
    max_selected = deal_dict.get("max_selected")
    max_iterations = deal_dict.get("max_iterations", 10)
    threshold_factor = deal_dict.get("threshold_factor", 0.7)
    rng = np.random.default_rng(data_cfg.seed) # use same seed for shuffling at each iteration

    print(f"[DEAL] Running in incremental mode with max_selected = {max_selected}.")

    if max_selected is None:
        raise ValueError("Incremental mode requires 'max_selected' to be set.")

    base_cfg = DEALConfig(**deal_dict)
    save_final_trajectory = base_cfg.save_full_trajectory == "final"
    all_images = list(data_cfg.images or [])
    selected_frames = set()
    deal = None
    start_iteration = 0
    resume_run = None  # set when resuming in the middle of an iteration
    selection_done = False

    if resume is not None:
        incremental = resume["incremental"]
        selected_frames = set(incremental["selected_frames"])
        rng.bit_generator.state = incremental["rng_state"]
        selection_done = resume.get("selection_done", False)
        if resume["run"]["completed"]:
            start_iteration = incremental["iteration"] + 1
        else:
            start_iteration = incremental["iteration"]
            resume_run = resume["run"]

    by_original_frame = {atoms.info["original_frame"]: atoms for atoms in all_images}

    def restart_context(iteration):
        return {
            "mode": "incremental",
            "config": config,
            "n_input_frames": len(all_images),
            "selection_done": selection_done,
            "incremental": {
                "iteration": iteration,
                "selected_frames": sorted(int(i) for i in selected_frames),
                "rng_state": rng.bit_generator.state,
            },
        }

    iteration = max(start_iteration - 1, 0)
    for iteration in range(start_iteration, max_iterations):
        if selection_done:
            break
        if resume_run is not None:
            # same frame order as the interrupted iteration (already shuffled)
            remaining_images = [by_original_frame[i] for i in resume_run["frame_order"]]
        else:
            remaining_images = [
                atoms
                for atoms in all_images
                if atoms.info.get("original_frame") not in selected_frames
            ]

            if iteration != 0:
                rng.shuffle(remaining_images) # shuffle remaining frames to avoid bias from frame ordering during incremental selection
            
        if len(remaining_images) == 0:
            print("\n[DEAL] Stopping incremental mode: all input frames were selected.")
            break

        deal_threshold = np.round(threshold_factor ** (iteration + 1),3)
        print(f"\n[DEAL] Iteration {iteration+1} (threshold: {deal_threshold})")

        run_deal_dict = dict(deal_dict)
        run_deal_dict["threshold"] = deal_threshold
        deal_cfg = DEALConfig(**run_deal_dict)
        # Each iteration only sees the remaining frames: with
        # save_full_trajectory="final" the trajectory is written once at the
        # end with the final GP (true keeps appending on-the-fly frames instead).
        if deal_cfg.save_full_trajectory == "final":
            deal_cfg.save_full_trajectory = False
        data_iter_cfg = DataConfig(images=remaining_images, seed=data_cfg.seed)

        if deal is None:
            deal = DEAL(data_iter_cfg, deal_cfg, sgp_cfg)
            if wall_start is not None:
                deal.wall_start = wall_start
            if resume is not None:
                deal.restore(resume)
        else:
            deal.configure_run(data_cfg=data_iter_cfg, deal_cfg=deal_cfg)

        deal.restart_context = restart_context(iteration)
        start_step = resume_run["step"] if resume_run is not None else 0
        resume_run = None

        previous_selected = len(selected_frames)
        if not deal.run(start_step=start_step):
            print(f"[DEAL] Resume with: deal --restart {deal.deal_cfg.output_prefix}.restart")
            return

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
            selection_done = True
        if base_cfg.restart:
            deal.restart_context = restart_context(iteration)
            deal.write_checkpoint(next_step=len(remaining_images), completed=True)
        if selection_done:
            break
    else:
        if not selection_done:
            print(
                f"\n[WARNING] Reached max_iterations={max_iterations} before "
                f"reaching max_selected={max_selected}. "
                f"Current selected: {len(selected_frames)}."
            )

    if deal is None and resume is not None:
        # resumed after the selection was over: only the outputs are left
        deal = DEAL(DataConfig(images=all_images, seed=data_cfg.seed), base_cfg, sgp_cfg)
        deal.restore(resume)
    if deal is None:
        return

    if save_final_trajectory:
        if base_cfg.restart:
            # the selection is over: a restart only redoes the final pass
            selection_done = True
            deal.restart_context = restart_context(iteration)
            deal.write_checkpoint(next_step=0, completed=True)
        deal.write_full_trajectory(all_images)

    if base_cfg.restart:
        deal.restart_context = restart_context(iteration)
        deal.write_checkpoint(next_step=0, completed=True, finished=True)


def main() -> None:

    print("""
888888ba   88888888b  .d888888  88        
88     8b  88        d8     88  88        
88     88  88aaaa    88aaaaa88  88        
88     88  88        88     88  88        
88     8P  88        88     88  88        
8888888P   88888888P 88     88  888888888
""")

    wall_start = time.perf_counter()
    args = parse_args()
    cfg_dict = {}
    resume = None

    if args.restart is not None:
        resume = read_restart(args.restart)
        if resume.get("finished", False):
            print(f"[DEAL] {args.restart}: the selection is already finished.")
            return
        cfg_dict = resume["config"]
        if args.wall_time is not None:
            cfg_dict["deal"]["wall_time"] = args.wall_time
        print(f"[DEAL] Resuming from {args.restart}.")
    # Start from YAML config if provided
    elif args.config is not None:
        with open(args.config, "r") as f:
            cfg_dict = yaml.safe_load(f) or {}

    # Initialize dicts if not available
    for key in ["data", "preprocessing", "deal", "sgp"]:
        if key not in cfg_dict:
            cfg_dict[key] = {}
    # Overwrite / fill from CLI options
    if args.filename is not None:
        cfg_dict["data"]["files"] = [args.filename]
    if args.threshold is not None:
        cfg_dict["deal"]["threshold"] = args.threshold
    if args.max_selected is not None:
        cfg_dict["deal"]["max_selected"] = args.max_selected
    if args.mask is not None:
        cfg_dict["deal"]["mask"] = _parse_mask_arg(args.mask)
    if args.preprocess_key is not None:
        cfg_dict["preprocessing"]["key"] = args.preprocess_key
    if args.preprocess_mask_threshold is not None:
        cfg_dict["preprocessing"]["mask_threshold"] = (
            args.preprocess_mask_threshold
        )
    if args.preprocess_mode is not None:
        cfg_dict["preprocessing"]["mode"] = args.preprocess_mode
    if args.preprocess_mask_upper_threshold is not None:
        cfg_dict["preprocessing"]["mask_upper_threshold"] = (
            args.preprocess_mask_upper_threshold
        )
    if args.preprocess_plot is not None:
        cfg_dict["preprocessing"]["plot"] = _parse_bool_or_filename(
            args.preprocess_plot
        )
    if args.wall_time is not None:
        cfg_dict["deal"]["wall_time"] = args.wall_time

    # A preprocessing mask should be consumed automatically unless the user
    # explicitly selected or disabled a DEAL mask. ``null`` means automatic.
    _resolve_default_mask(cfg_dict)

    # Check file
    try:
        cfg_dict["data"]["files"][0]
    except KeyError:
        print(
            "[ERROR] No input trajectory specified. Please provide a trajectory file (-f/--file) or a YAML config file (-c/--config)."
        )
        sys.exit(1)

    # Configuration stored in restart files, with absolute input paths so
    # that a resumed job does not depend on its working directory.
    config = deepcopy(cfg_dict)
    files = config["data"]["files"]
    files = [files] if isinstance(files, str) else files
    config["data"]["files"] = [os.path.abspath(f) for f in files]

    # Build data/SGP configs
    data_cfg = DataConfig(**cfg_dict["data"])
    _apply_preprocessing(data_cfg, cfg_dict["preprocessing"])
    sgp_cfg = SGPConfig(**cfg_dict["sgp"])

    deal_dict = dict(cfg_dict["deal"])
    has_threshold = deal_dict.get("threshold", None) is not None
    has_max_selected = deal_dict.get("max_selected", None) is not None

    if has_threshold and has_max_selected:
        print(
            "[ERROR] 'threshold' and 'max_selected' are mutually exclusive. "
            "Use 'threshold' for standard mode or 'max_selected' for incremental mode."
        )
        sys.exit(1)

    if resume is not None and resume["n_input_frames"] != len(data_cfg.images):
        print(
            f"[ERROR] The input now has {len(data_cfg.images)} frames, but the "
            f"restart file was written for {resume['n_input_frames']}."
        )
        sys.exit(1)

    if has_max_selected:
        _run_incremental_cli(
            data_cfg, deal_dict, sgp_cfg,
            config=config, resume=resume, wall_start=wall_start,
        )
        return

    # Standard mode: handle one threshold or a list of independent thresholds
    thresholds = deal_dict.get("threshold", None)

    if isinstance(thresholds, list):
        if deal_dict.get("restart") or deal_dict.get("wall_time") is not None:
            print("[ERROR] 'restart' and 'wall_time' need a single threshold.")
            sys.exit(1)
        for th in thresholds:
            run_deal_dict = dict(deal_dict)
            run_deal_dict["threshold"] = th
            prefix = deal_dict.get("output_prefix", "deal")
            run_deal_dict["output_prefix"] = f"{prefix}_{th}"

            print(f"[DEAL] Running with threshold: {th}")
            deal_cfg = DEALConfig(**run_deal_dict)
            DEAL(data_cfg, deal_cfg, sgp_cfg).run()
    else:
        _run_threshold_cli(data_cfg, deal_dict, sgp_cfg, config, resume, wall_start)


def _run_threshold_cli(
    data_cfg: DataConfig,
    deal_dict: dict,
    sgp_cfg: SGPConfig,
    config: dict,
    resume: Optional[dict],
    wall_start: float,
) -> None:
    """Standard mode with a single threshold, optionally resumed."""
    deal_cfg = DEALConfig(**deal_dict)
    start_step = 0
    if resume is not None:
        by_original_frame = {a.info["original_frame"]: a for a in data_cfg.images}
        data_cfg = DataConfig(
            images=[by_original_frame[i] for i in resume["run"]["frame_order"]],
            seed=data_cfg.seed,
        )
    deal = DEAL(data_cfg, deal_cfg, sgp_cfg)
    deal.wall_start = wall_start
    deal.restart_context = {
        "mode": "threshold",
        "config": config,
        "n_input_frames": len(data_cfg.images),
    }
    if resume is not None:
        deal.restore(resume)
        start_step = resume["run"]["step"]

    if not deal.run(start_step=start_step):
        print(f"[DEAL] Resume with: deal --restart {deal_cfg.output_prefix}.restart")
        return
    if deal_cfg.restart:
        deal.write_checkpoint(
            next_step=len(data_cfg.images), completed=True, finished=True
        )
