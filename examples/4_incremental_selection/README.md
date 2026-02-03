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
max_selected = 50
```

Run it!

```bash
python incremental_DEAL.py
```

This will produce files with prefix `deal_selected_iter[i]_` inside the `deal_folder`. At the end of each iteration, the selected structures are moved at the beginning of the new input file, and DEAL is re-run with a lower threshold to select more and more structures until `max_selected` is reached. Example output:

```raw 
ITERATION 0 (threshold: 0.75)
[DEAL] Examined:   200 | Selected:     7 | Speed:   0.10 s/step | Elapsed:    34.78 s

ITERATION 1 (threshold: 0.5)
[DEAL] Examined:   200 | Selected:    14 | Speed:   0.18 s/step | Elapsed:    58.48 s

ITERATION 2 (threshold: 0.4)
[DEAL] Examined:   200 | Selected:    21 | Speed:   0.12 s/step | Elapsed:   134.87 s

ITERATION 3 (threshold: 0.3)
[DEAL] Examined:   200 | Selected:    26 | Speed:   0.26 s/step | Elapsed:   130.11 s

ITERATION 4 (threshold: 0.2)
[DEAL] Examined:   200 | Selected:    38 | Speed:   0.39 s/step | Elapsed:   238.08 s

ITERATION 5 (threshold: 0.15)
[DEAL] Examined:   200 | Selected:    51 | Speed:   0.25 s/step | Elapsed:   399.52 s

Stopping loop: (selected = 51 >= max_selected = 50)
```