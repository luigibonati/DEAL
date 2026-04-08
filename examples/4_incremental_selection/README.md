## Incremental selection 

aka using progressively lower thresholds to select more and more datapoints.

### Prepare data (shuffle)

```python
import random 
from ase.io import read,write

# read data
traj = read('a_input/input_fcu.xyz',index=":")
for i,atoms in enumerate(traj):
    atoms.info['original_frame'] = i

# shuffle
random.seed(42)
random.shuffle(traj)

# write
write('a_input/shuffled.xyz',traj)
```

### Run incremental DEAL 

Progressively lower the threshold to include more data at each iteration.

Customize the parameters in the file `incremental_DEAL.py`:

```python
# ------------------------
# user parameters
# ------------------------
traj_input_file = "a_input/shuffled.xyz"
deal_folder = "b_selection"
max_iterations = 5
max_selected = 16
threshold_factor = 0.75
```

Run it!

```bash
python incremental_DEAL.py
```

How it works:

- `DataConfig` is created from in-memory `atoms_list` (no temporary `input_iter*.xyz` files).
- `DEAL` is initialized once (first iteration).
- At each following iteration, `configure_run(data_cfg=..., deal_cfg=...)` updates the threshold and the input set (remaining, not-yet-selected frames) without rebuilding the SGP.
- A single output prefix is used: `b_selection/deal`.

Output files:

- `b_selection/deal_selected.xyz`: cumulative selected frames across iterations.
- `b_selection/deal_trajectory_uncertainty.xyz`: trajectory with per-frame atomic uncertainty (if `save_full_trajectory=True`).

The loop stops when either:

- `len(frames_selected) >= max_selected`, or
- all frames have already been selected.

Example output:

```raw 
ITERATION 0 (threshold: 0.75)
[DEAL] Examined:    50 | Selected:     6 | Speed:   0.42 s/step | Elapsed:    30.01 s

ITERATION 1 (threshold: 0.562)
[DEAL] Examined:    44 | Selected:     9 | Speed:   0.48 s/step | Elapsed:    29.41 s

ITERATION 2 (threshold: 0.422)
[DEAL] Examined:    41 | Selected:    16 | Speed:   0.35 s/step | Elapsed:    56.98 s

Stopping loop: (selected = 16 >= max_selected = 16)
```
