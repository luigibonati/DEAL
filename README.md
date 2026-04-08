# **DEAL**

**Data Efficient Active Learning for Machine Learning Potentials**

> Reference: **Perego S. & Bonati L.**
> *Data efficient machine learning potentials for modeling catalytic reactivity via active learning and enhanced sampling*,
> **npj Computational Materials 10, 291 (2024)**
> doi: [10.1038/s41524-024-01481-6](https://doi.org/10.1038/s41524-024-01481-6)

DEAL selects non-redundant structures from atomistic trajectories via Sparse Gaussian Processes (SGP), to be used to train machine-learning interatomic potentials.

It consists of two steps:
1. preselection using the MLP uncertainty (e.g. max uncertainty obtained with query-by-committee)
2. select a dataset of non-redundant configurations using the local predictive variance of a Gaussian Process

In addition, step 2. can be also used to perform subsampling of a trajectory even without the uncertainty pre-selection (see [examples](examples/README.md))

A short practical [introduction](DEAL.md) describes the main ingredients (SGP, local descriptors, uncertainty).

## Code highlights

* Select structures based on SGP predictive variance. 
* Analyze selected structures (e.g. along the trajectory or as a function of a CV)

    <img src="imgs/analysis.png" alt="drawing" width="824"/>
* Interactive visualization using [chemiscope](https://chemiscope.org/)

    <a href="https://chemiscope.org/?load=https://raw.githubusercontent.com/luigibonati/DEAL/refs/heads/main/examples/2_subsampling_formate/selection/deal_0.1_chemiscope.json.gz"> <img src="imgs/chemiscope-viewer.png" alt="drawing" width="412"></a>

---

## Table of contents
- 📚 [Contents](#-contents)
- 🔧 [Dependencies](#-dependencies)
- 🚀 [Installation](#-installation)
- 🧪 [Usage](#-usage)
  - [Minimal example](#minimal-example)
  - [With a YAML config file](#with-a-yaml-config-file)
  - [Python usage](#python-usage)
  - [Output files](#output-files)
  - [Multiple thresholds](#multiple-thresholds)
  - [Incremental selection (CLI)](#incremental-selection-cli)
  - [Create a chemiscope file](#create-a-chemiscope-file)
  - 🎛️ [Choice of the parameters](#choice-of-the-parameters)

---

## 📚 Contents

* **`deal/`** – The core Python package.
* **`examples/`** – Two realistic workflows demonstrating how to use DEAL in practice.
* **`npj_supporting_data/`** – Jupyter notebooks reproducing the full workflow described in the publication, including the use of Gaussian Process Regression for reaction-pathway exploration.
* **`tests/`** – A test suite to verify that the installation is correct and all components work as expected.

## 🔧 Dependencies

DEAL requires:

* `python<=3.13` 
* [`flare==1.3.3b`](https://github.com/mir-group/flare)
* `ase`
* `pandas`
* `numpy`
* `chemiscope` (for visualization)

---

## 🚀 Installation

Below is a complete installation sequence.

Create environment

```bash
conda create -n deal python=3.12
conda activate deal
```

Install **FLARE 1.3.3b**

(See: [https://mir-group.github.io/flare/installation/install.html](https://mir-group.github.io/flare/installation/install.html))

```bash
conda install -y gcc gxx cmake openmp liblapacke openblas -c conda-forge
git clone https://github.com/mir-group/flare.git -b 1.3.3b
cd flare
pip install .
cd ..
```

3. Install DEAL

```bash
git clone https://github.com/luigibonati/DEAL.git
cd DEAL
pip install .
```



---

##  Usage

DEAL can be run either with a command-line tool (`deal`) or using the python class (`DEAL`).

---


###  Minimal example

```bash
deal --file traj.xyz --threshold 0.1
```

or run in incremental mode by targeting a number of selected frames (this will decrease the threshold at each iteration):

```bash
deal --file traj.xyz --max-selected 50
```

DEAL will automatically:

* detect atomic species from the first frame
* use default GP/kernel/descriptor parameters
* use default output names


### 📄 With a YAML config file

For more customization you can create an `input.yaml` file (below are the default):

```bash
deal -c input.yaml
```

```yaml
data:
  files: ["traj.xyz"]     # can be a single file or a list of files
  #format: "extxyz"        # file format (e.g. extxyz, xyz, ...)
  #index: ":"              # frame selection [see ASE notation]
  #shuffle: false          # whether to shuffle the frames before processing 
  #seed: 42

deal:
  threshold: 0.1           standard mode: scalar or list of values
  # OR
  #max_selected: 50       # incremental mode: mutually exclusive with `threshold`
  #max_iterations: 10     # max iterations for incremental mode
  #threshold_factor: 0.75 # threshold_i = threshold_factor**(i+1)
  
  update_threshold: 0.08 # if not set it is chosen as 0.8 * threshold 
  max_atoms_added: 0.2    # limit the number of selected environments added per configuration (can be int (number of atoms) float (0,1) (fraction of total atoms), or -1 (no limit)
  initial_atoms: none     # specify which atoms to use for GP initialization (list, fraction or number. Default: none, 1 atom per species)
  output_prefix: deal     # prefix for output files (threshold will be appended as suffix)
  force_only: true
  train_hyps: false       # whether to re-train hyperparameters at each iteration (slower) 
  verbose: true           # allowed values: true/false/"debug" (default: true)
  save_gp: false
  save_full_trajectory: false  # if true, writes <prefix>_trajectory_uncertainty.xyz with per-atom array "atomic_uncertainty"

flare_calc:
  gp: SGP_Wrapper         # (see flare's documentation)
  kernels:
    - name: NormalizedDotProduct
      sigma: 2 
      power: 2
  descriptors:
    - name: B2
      nmax: 8
      lmax: 3
      cutoff_function: cosine
      radial_basis: chebyshev
  cutoff: 4.5
  
```
One can also use a base config file and override via CLI the options:
```bash
deal -c input.yaml -t 0.15
deal -c input.yaml --max-selected 50
```

### Python Usage

```python
# Import 
from deal import DataConfig, DEALConfig, FlareConfig, DEAL

# Define Config (uses defaults where not provided)
data_cfg = DataConfig(files="traj.xyz")
deal_cfg = DEALConfig(
    threshold=0.1,
    output_prefix="deal",    
)
flare_cfg = FlareConfig()

# Instantiate DEAL class
deal = DEAL(data_cfg, deal_cfg, flare_cfg)

# Run 
deal.run()

```

### Output files

In both cases the following file is generated (with the default `output_prefix=deal`):

1. **`deal_selected.xyz` – selected frames** 

Contains the atomic configurations where the GP uncertainty exceeded the threshold.
Includes atoms.info["original_frame"] indicating the original trajectory index.
If `save_gp: true`, DEAL also writes:
2. **`deal_flare.json` – final GP model**

If `save_full_trajectory: true`, DEAL also writes:
3. **`deal_trajectory_uncertainty.xyz` – full trajectory with uncertainties**

Contains all processed frames with:
- per-atom array `atomic_uncertainty` (saved in `atoms.arrays`)
- frame scalar `max_atomic_uncertainty` (saved in `atoms.info`)

### Multiple thresholds

If the CLI receives a list of thresholds, DEAL will run once per threshold
(standard mode only).
```yaml
deal:
  threshold:
    - 0.10
    - 0.15
    - 0.20
```

Equivalent behaviour in Python:

```python
for thr in [0.10, 0.15, 0.20]:
    deal_cfg.threshold = thr
    deal_cfg.output_prefix = f"run_thr{thr}"
    DEAL(data_cfg, deal_cfg, flare_cfg).run()
```

### Incremental selection (CLI)

Use incremental mode when you want to select up to a target number of structures.

```yaml
deal:
  max_selected: 50
  max_iterations: 10
  threshold_factor: 0.75
```

At each iteration `i` (starting from 1), the threshold is decreased as:

`threshold_i = threshold_factor**i`

The run stops when one of these conditions is met: either selected structures reach `max_selected` or `max_iterations` is reached.

Note: `threshold` and `max_selected` are mutually exclusive in the CLI.

### Create a chemiscope file

After running `deal`, one can create a chemiscope visualization file with:

```bash
deal-chemiscope --prefix deal
```

or explicitly with the trajectory file and optionally also a COLVAR file:

```bash
deal-chemiscope --trajectory deal_selected.xyz --colvar COLVAR
```

`deal-chemiscope` creates a chemiscope visualization file from the selected structures.
The file includes all numeric properties available in `atoms.info` for each frame, and,
when `--colvar` is provided, it also includes the properties loaded from the COLVAR file.

This writes `deal_chemiscope.json.gz` (by default with `--prefix deal`).
It can be viewed online at https://chemiscope.org/ or inside Python:
```python
import chemiscope
chemiscope.show_input('deal_chemiscope.json.gz')
```
See also the Chemiscope [documentation](https://chemiscope.org/docs/).


## 🎛️ Choice of the parameters

Below a quick guide, see the [introduction](DEAL.md) to DEAL for a more in-depth explanation.

**Descriptors**

Local environments are characterized via the Atomic Cluster Expansion formalism as implemented in `flare`. Key hyperparameters: body order (`B1/B2`), radial degree `nmax`, angular degree `lmax`, and `cutoff` (in Å).

```yaml
  descriptors:
    - name: B2
      nmax: 8
      lmax: 3
      cutoff_function: cosine
      radial_basis: chebyshev
  cutoff: 4.5
```      

**Threshold**
```yaml
  threshold: 0.1
  update_threshold: 0.08  # if not set it is chosen as 0.8 * threshold      
  max_atoms_added: -1 # no limit on the number of selected environment of a given configuration to the GP.
  initial_atoms: 0.15 # use up to 15% of the atoms (of each species) for GP initialization
```      

The`threshold` parameter in the DEAL configuration controls when a local environment is flagged by the SGP’s predictive variance (normalized by the noise hyperparameter). If any environment exceeds the threshold, the GP is updated and that environment (plus any others above `update_threshold`, up to `max_atoms_added`) is added. 

Some tips:

- The uncertainty values are unitless, and range between 0 and 1. Lower threshold means more selected structures; higher threshold, fewer selections. 
- A good starting point is around 0.1. As a rule of thumb, homogeneous, condensed and/or crystalline systems tend to have fewer different local environments and require smaller thresholds (<<0.1), whereas heterogeneous systems may require larger ones (>0.1). This is also connected with the number of species (more species -> higher treshold required).
- Try a few values and compare how many structures are selected; distributions often are very similar across thresholds, what changes is the number of structures. One can decide based on the computational budget (for DFT calculations).
- A practical strategy is to perform **incremental selection**: start with a high threshold, then decrease it progressively until a target number of structures is reached.
- Use chemiscope/OVITO to visualize the selected structures and identify which environments triggered selection. 

Note: the training time scales unfavorably with the number of samples. For a large dataset, it is advised to divide it in chuncks and run DEAL separately on each of them, and then performing a second DEAL selection on the output structures. 
