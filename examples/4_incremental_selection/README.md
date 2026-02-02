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

This will produce files with prefix `deal_selected_iter[i]_` inside the `deal_folder`. At the end of each iteration, the selected structures are moved at the beginning of the new input file, and DEAL is re-run with a lower threshold to select more and more structures until `max_selected` is reached. 