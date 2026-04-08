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
max_iterations = 10
max_selected = 50
threshold_factor = 0.75
```

Run it!

```bash
python incremental_DEAL.py
```

How it works:

- `DEAL` is initialized once (first iteration).
- At each following iteration, `configure_run(data_cfg=..., deal_cfg=...)` updates the threshold and the input set (remaining, not-yet-selected frames) without rebuilding the SGP.
- The subsequent `DataConfig` is created from in-memory `images` using the non-selected frames.
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
[DEAL] Examined:   200 | Selected:     7 | Speed:   0.45 s/step | Elapsed:    98.32 s

ITERATION 1 (threshold: 0.562)
[DEAL] Examined:   193 | Selected:    12 | Speed:   0.44 s/step | Elapsed:   203.37 s

ITERATION 2 (threshold: 0.422)
[DEAL] Examined:   188 | Selected:    21 | Speed:   0.43 s/step | Elapsed:   353.09 s

ITERATION 3 (threshold: 0.316)
[DEAL] Examined:   179 | Selected:    33 | Speed:   0.43 s/step | Elapsed:   579.85 s

ITERATION 4 (threshold: 0.237)
[DEAL] Examined:   167 | Selected:    46 | Speed:   0.46 s/step | Elapsed:   891.99 s

ITERATION 5 (threshold: 0.178)
[DEAL] Examined:   154 | Selected:    57 | Speed:   0.46 s/step | Elapsed:  1227.35 s

Stopping loop: (selected = 57 >= max_selected = 50)
```
