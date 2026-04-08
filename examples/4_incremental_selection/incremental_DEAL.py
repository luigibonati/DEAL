from pathlib import Path
from ase.io import read, write
import numpy as np

from deal import DataConfig, DEALConfig, FlareConfig, DEAL

# ------------------------
# user parameters
# ------------------------
traj_input_file = "a_input/shuffled.xyz"
deal_folder = "b_selection"
max_iterations = 5
max_selected = 16
threshold_factor = 0.75

# ------------------------
# setup
# ------------------------
Path(deal_folder).mkdir(parents=True, exist_ok=True)
traj_input = read(traj_input_file, ":50")

if 'original_frame' not in traj_input[0].info:
    for i,atoms in enumerate(traj_input):
        atoms.info['original_frame'] = i

# ------------------------
# incremental DEAL loop
# ------------------------
frames_selected = set()

thresholds = [np.round(np.power(threshold_factor,i+1), 3) for i in range(max_iterations)]

for iter, deal_threshold in enumerate(thresholds):

    print(f"\nITERATION {iter} (threshold: {deal_threshold})")

    # Initalize or update DEAL
    if iter == 0:
        deal_cfg = DEALConfig(
            threshold=deal_threshold,
            max_atoms_added=0.15,
            initial_atoms=0.2,
            output_prefix=f"{deal_folder}/deal",
            verbose=False,
            save_full_trajectory=True
        )
        data_cfg = DataConfig(atoms_list=traj_input)
        flare_cfg = FlareConfig()
        deal = DEAL(data_cfg, deal_cfg, flare_cfg)
    else:
        # update input and threshold without rebuilding the SGP
        data_cfg = DataConfig(atoms_list=traj_new)
        #deal_cfg.threshold = deal_threshold
        deal_cfg = DEALConfig(
            threshold=deal_threshold,
            max_atoms_added=0.15,
            initial_atoms=0.2,
            output_prefix=f"{deal_folder}/deal",
            verbose=False,
            save_full_trajectory=True
        )
        
        deal.configure_run(
            data_cfg=data_cfg,
            deal_cfg=deal_cfg
        )

    deal.run()

    # create input for next iteration
    traj_selected = deal.selected_frames
    frames_selected.update(f.info["original_frame"] for f in traj_selected)

    traj_new = [f for f in traj_input if f.info["original_frame"] not in frames_selected]

    # ------------------------
    # stopping condition
    # ------------------------
    if len(frames_selected) >= max_selected:
        print(
            f"\nStopping loop: "
            f"(selected = {len(frames_selected)} >= max_selected = {max_selected})"
        )
        break

    if len(traj_new) == 0:
        print("\nStopping loop: all input frames were selected.")
        break
