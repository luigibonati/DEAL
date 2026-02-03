from pathlib import Path
from ase.io import read, write

from deal import DataConfig, DEALConfig, FlareConfig, DEAL

# ------------------------
# user parameters
# ------------------------
traj_input_file = "a_input/shuffled.xyz"
deal_folder = "b_selection"
max_selected = 50

thresholds = [
    0.75, 0.5, 
    0.4, 0.3, 0.2,
    0.15, 0.125, 0.1,
    0.09, 0.08, 0.07, 0.06, 0.05,
]
# ------------------------
# setup
# ------------------------
Path(deal_folder).mkdir(parents=True, exist_ok=True)
traj_input = read(traj_input_file, ":")
if 'original_frame' not in traj_input[0].info:
    for i,atoms in enumerate(traj_input):
        atoms.info['original_frame'] = i

# ------------------------
# incremental DEAL loop
# ------------------------
for iter, deal_threshold in enumerate(thresholds):

    print(f"\nITERATION {iter} (threshold: {deal_threshold})")

    input_file = traj_input_file if iter == 0 else f"{deal_folder}/input_iter{iter}.xyz"
    output_prefix = f"{deal_folder}/deal_iter{iter}"

    data_cfg = DataConfig(files=input_file)

    deal_cfg = DEALConfig(
        threshold=deal_threshold,
        max_atoms_added=0.15,
        initial_atoms=0.2,
        output_prefix=output_prefix,
        verbose=False,
    )

    flare_cfg = FlareConfig()

    deal = DEAL(data_cfg, deal_cfg, flare_cfg)
    deal.run()

    # ------------------------
    # reorder input for next iteration (selected first)
    # ------------------------
    traj_selected = read(f"{output_prefix}_selected.xyz", ":")
    frames_selected = [f.info["original_frame"] for f in traj_selected]
    traj_new = traj_selected + [f for f in traj_input if f.info["original_frame"] not in frames_selected]
    write(f"{deal_folder}/input_iter{iter+1}.xyz", traj_new)

    # ------------------------
    # stopping condition
    # ------------------------
    if len(traj_selected) >= max_selected:
        print(
            f"\nStopping loop: "
            f"(selected = {len(traj_selected)} >= max_selected = {max_selected})"
        )
        break