from pathlib import Path
from ase.io import read, write
import numpy as np

from deal import DataConfig, DEALConfig, FlareConfig, DEAL

# ------------------------
# user parameters
# ------------------------
traj_input_file = "input/fcu_ev5.xyz.gz"
deal_folder = "b_selection"
max_iterations = 10
max_selected = 20

# ------------------------
# setup
# ------------------------
Path(deal_folder).mkdir(parents=True, exist_ok=True)
traj_input = read(traj_input_file, ":")

# ------------------------
# incremental DEAL loop
# ------------------------
frames_selected = set()

threshold_factor = 0.75
thresholds = [np.round(np.power(threshold_factor,i+1), 3) for i in range(max_iterations)]

for iter, deal_threshold in enumerate(thresholds):

    print(f"\nITERATION {iter} (threshold: {deal_threshold})")

    # Initalize or update DEAL
    if iter == 0:
        deal_cfg = DEALConfig(
            threshold=deal_threshold,
            max_atoms_added=0.15,
            initial_atoms=0.2,
            output_prefix=f"{deal_folder}/deal_{deal_threshold}",
            verbose=False,
            save_full_trajectory=True
        )
        data_cfg = DataConfig(images=traj_input)
        flare_cfg = FlareConfig()
        deal = DEAL(data_cfg, deal_cfg, flare_cfg)
    else:
        # update input and threshold without rebuilding the SGP
        data_cfg = DataConfig(images=traj_new)
        deal_cfg = DEALConfig(
            threshold=deal_threshold,
            max_atoms_added=0.15,
            initial_atoms=0.2,
            output_prefix=f"{deal_folder}/deal_{deal_threshold}",
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

# ------------------------
# Plot selection results
# ------------------------

import matplotlib.pyplot as plt
import glob

files = glob.glob(f"{deal_folder}/deal_*_trajectory_uncertainty.xyz")

trajectories = {}
for file in files:
    threshold = file.split("/")[-1].split("_")[1]
    trajectories[float(threshold)] = {
        "trajectory": read(file, index=":"),
        "max_uncertainty": [atoms.info['max_atomic_uncertainty'] for atoms in read(file, index=":")]
    }

thresholds = sorted(trajectories.keys(), reverse=True)
trajectories[max(thresholds)]["max_uncertainty"][0] = 1

# Plot
fig, axs = plt.subplots(
    len(thresholds),
    1,
    sharex=True,
    sharey=True,
    figsize=(5.5, 1.5 * len(thresholds)),
    gridspec_kw={"hspace": 0},
    dpi=100
)
axs = np.atleast_1d(axs)

for i, th in enumerate(thresholds):
    frames = np.array([atoms.info["original_frame"] for atoms in trajectories[th]["trajectory"]])
    max_unc = np.array(trajectories[th]["max_uncertainty"])
    mask = max_unc > th

    axs[i].plot(frames, max_unc)
    axs[i].axhline(th, color="red", linestyle="--", label=f"threshold={th}")
    axs[i].scatter(frames[mask], max_unc[mask], color="orange", marker="*", s=120, zorder=3)
    axs[i].margins(x=0)
    axs[i].legend(frameon=False, loc="upper right", ncol=2)
    if i % 2 == 0:
        axs[i].set_ylabel("Max Uncertainty")

axs[-1].set_xlim(0,200)
axs[-1].set_xlabel("original_frame")
fig.suptitle('Incremental DEAL selection')
plt.tight_layout()
plt.savefig(f"incremental_DEAL_selection.png", dpi=300, bbox_inches='tight' )