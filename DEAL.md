# Introduction to DEAL

DEAL (Data-Efficient Active Learning) [1] is a trajectory subsampling strategy for building machine-learning interatomic potentials with fewer expensive electronic-structure labels (typically DFT single points). Its main objective is to reduce redundancy: instead of labeling many similar configurations, DEAL selects a compact set that covers distinct local environments.

## 1) Active Learning of Machine-Learning Potentials

<p align="center"> <img src="imgs/active_learning.png" width="600"/> </p>

A standard active-learning cycle for machine learning potentials is:
1. Run MD simulations (possibly with enhanced-sampling) with the current ML potential.
3. Select configurations for DFT labeling.
4. Retrain/update the potential.
5. Repeat.

In many workflows, step 2 is performed with *query-by-committee*, where an ensemble of models predicts on the same trajectory and high disagreement is used as a selection criterion. This is effective, but often redundant: due to the slow mixing of MD, many selected frames are near-duplicates in local-environment space. DEAL was designed to subsample these configurations into a non-redundant subset, reducing unnecessary DFT calculations.

## 2) Data-Efficient Active Learning in DEAL

> DEAL uses the uncertainty prediction of a Gaussian Process model to select the configurations based on the structural similarity via local-environment descriptors.

We first introduce the ingredients used by DEAL, particularly the sparse Gaussian process (SGP) based on ACE local descriptors and the uncertainty measure, as implemented in the FLARE [2] package.

### 2.1 Local descriptors (ACE)

In the Atomic Cluster Expansion formalism, each atom $i$ is represented by a descriptor vector $\mathbf{d}_i$ built from its local environment $\rho_i$, that is, the list of neighbors within a cutoff radius:
<p align="center"> <img src="imgs/ACE.png" width="200"/> </p>

At high level, these descriptors encode:
- radial information (how neighbor density varies with distance,
- angular information (how neighbors are arranged in direction),
- chemical identity (which species contribute to the local density).

Relevant config parameters are:
- `flare_calc.cutoff` (global cutoff radius $r_c$, or, optionally, pair-dependent cutoffs specified via `cutoff_matrix`)
- `flare_calc.descriptors.nmax` (radial resolution)
- `flare_calc.descriptors.lmax` (angular resolution)

Notes:
- **Multiple species**. For multi-component systems, neighbor density of different species are concatenated, so environments with similar geometry but different chemistry remain distinguishable.
- **Radial resolution**. For fixed `nmax`, increasing `cutoff` expands the spatial domain represented by the same number of radial basis functions, decreasing radial detail per unit distance. A useful rule-of-thumb is: $\Delta r \sim \frac{r_c}{n_{\max}}$, where $r_c$ is the cutoff. If `cutoff` is increased, `nmax` should be increased as well to avoid losing resolution.

### 2.2 Energy model and kernel representation

We define a model for the energy using a (sparse) Gaussian Process. This is based on a local decomposition of total energy, where the i-th atomic contribution depends on atom-centered local environments $\rho_i$:

$$
E = \sum_{i=1}^{N} \varepsilon(\rho_i)
$$

For a given atomic environment $\rho$, the local-energy prediction can be written as:

$$
\varepsilon(\rho) = \sum_{m=1}^{M} \alpha_m \, k(\rho, \rho_m)
$$

where:

- $\{\rho_m\}_{m=1}^{M}$ are environments in the training set
- $k(\rho, \rho_m)$ is the kernel similarity measured as:
$$
k(\rho_i,\rho_j)=\sigma^2
\left(
\frac{\mathbf{d}_i\cdot\mathbf{d}_j}{\|\mathbf{d}_i\|\,\|\mathbf{d}_j\|}
\right)^{2}
$$

- $\alpha_m$ are learned GP weights during training.

So the model compares each current environment against reference environments in the training set and combines these similarities. To keep memory and compute manageable, FLARE uses a **sparse GP** updates: only a subset of environments is retained as reference environments.

### 2.3 GP predictive variance 

The key feature of a GP is that it provides a predictions of both the value and its associated uncertainty. For a query local environment $\rho$, the GP predictive variance associated with the the atomic energy contribution has the form:

$$
\mathrm{Var}[\varepsilon(\rho)]
=\frac{k(\rho,\rho)-\mathbf{k}^\top\mathbf{K}^{-1}\mathbf{k}}{
    \sigma^2
}
$$
which depends on the matrix of kernel similarities between the enviroment and the reference set $\mathbf{k}$ (each element is $k(\rho,\rho_m)$), as well as the one between the reference structures themselves ($\mathbf{K}$). In other words, the predicted uncertainty is based only on structural novelty measured in local-environment space, not on the (DFT) energies. Furthermore, in FLARE the variance is normalized by the lengthscale hyperparameter $\sigma$, to yield a number betweeen 0 and 1.

## 3) How DEAL selection works

Now that we have introduced all the ingredients, we can describe the DEAL selection process which is repeated for each configuration of candidate structures (e.g. generated by the query-by-committee strategy):

1. Compute local (per-atom) uncertainty values.
2. If at least one atom exceeds the selection threshold, select the configuration.
    - a. Add the triggering environment(s) to the sparse set.
    - b. Update the uncertainty model.

Relevant `deal` config parameters:

- `deal.threshold`: uncertainty value which triggers selection. 
- `deal.update_threshold`: threshold for adding additional environments from the same selected structure 
- `deal.max_atoms_added`: cap on how many local environments are added per selected structure.
- `deal.initial_atoms`: which atoms add to the sparse set in the first configuration (index or percentage).

## 4) Practical guidance for threshold choice and workflow

- The uncertainty values are unitless, and range between 0 and 1. Lower threshold means more selected structures; higher threshold, fewer selections. 
- A good starting point is around 0.1. As a rule of thumb, homogeneous, condensed and/or crystalline systems tend to have fewer different local environments and require smaller thresholds (<<0.1), whereas heterogeneous systems may require larger ones (>0.1). This is also connected with the number of species (more species -> higher treshold required).
- Try a few values and compare how many structures are selected; distributions often are very similar across thresholds, what changes is the number of structures. One can decide based on the computational budget (for DFT calculations).
- A practical strategy is to perform **incremental selection**: start with a high threshold, then decrease it progressively until a target number of structures is reached (see example #4).
- Use chemiscope to inspect selected structures and identify which environments triggered selection. 

Note: the training time scales unfavorably with the number of samples. For a large dataset, it is advised to divide it in chuncks and run DEAL separately on each of them, and then performing a second DEAL selection on the output structures. 

## References

[1] DEAL: https://www.nature.com/articles/s41524-024-01481-6

[2] FLARE SGP: https://www.nature.com/articles/s41467-022-32294-0
