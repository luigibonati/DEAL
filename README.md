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
* `chemiscope`
* `pandas`
* `numpy`

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
  format: "extxyz"        # file format (e.g. extxyz, xyz, ...)
  index: ":"              # frame selection (e.g. ":", "0:100", [0,10,20]) [see ASE notation]
  colvar: none            # collective variables file associated with the trajectory (optional, used for monitoring CVs in chemiscope)
  shuffle: false          # whether to shuffle the frames before processing (suggested true for MD data)
  seed: 42

deal:
  threshold: 0.1         #can be a single value or a list of values
  update_threshold: 0.08 # if not set it is chosen as 0.8 * threshold 
  max_atoms_added: 0.2    # limit the number of selected environments added per configuration (can be int (number of atoms) float (0,1) (fraction of total atoms), or -1 (no limit)
  initial_atoms: none     # specify which atoms to use for GP initialization (list, fraction or number. Default: none, 1 atom per species)
  output_prefix: deal    # prefix for output files (threshold will be appended as suffix)
  force_only: true
  train_hyps: false      # whether to re-train hyperparameters at each iteration (slower) 
  verbose: true          # allowed values: true/false/"debug" (default: true)
  save_gp: false

flare_calc:
  gp: SGP_Wrapper        # (see flare's documentation)
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

In both cases the following files (with the default `output_prefix=deal`):

1. **`deal_selected.xyz` – selected frames** 

Contains the atomic configurations where the GP uncertainty exceeded the threshold.
Includes atoms.info["frame"] indicating the original trajectory index.

2. **`deal_chemiscope.json.gz` – chemiscope visualization file**

Can be viewed online at https://chemiscope.org/ or inside Python:
```python
import chemiscope
chemiscope.show_input('deal_chemiscope.json.gz')
```

### Multiple thresholds

If the CLI receives a list of thresholds, DEAL will run once per threshold.
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
- A practical strategy is to perform **incremental selection**: start with a high threshold, then decrease it progressively until a target number of structures is reached (see example #4).
- Use chemiscope to inspect selected structures and identify which environments triggered selection. 

Note: the training time scales unfavorably with the number of samples. For a large dataset, it is advised to divide it in chuncks and run DEAL separately on each of them, and then performing a second DEAL selection on the output structures. 