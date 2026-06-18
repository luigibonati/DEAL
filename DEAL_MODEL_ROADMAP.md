# DEAL-Specific Model Roadmap

The goal is to stop treating the upstream model package as a full dependency
surface and keep only
the model machinery needed by DEAL:

1. Build local atom-centered descriptors for each ASE frame.
2. Predict per-atom GP uncertainty.
3. Select atoms/configurations above DEAL thresholds.
4. Update the sparse GP with selected force labels.
5. Expose a small calculator-like API that can later drive uncertainty-aware MD.

## Current Boundary

`deal.model.DealActiveLearningModel` is the only model API used by
`deal.core.DEAL`:

- `to_model_atoms`
- `predict_uncertainty`
- `select_atoms_by_uncertainty`
- `update`
- `write`

This keeps current behavior intact while giving us a narrow replacement target.

## Trajectory Preprocessing

`deal.preprocessing.TrajectoryMasker` prepares trajectories with a reusable
per-atom mask before active learning starts.  The command-line entry point is:

```bash
deal-mask -f input.xyz -o masked.xyz -k force_std_comp_max -t 0.05 --mask-key deal_mask
```

The active-learning config can then consume the generated mask with:

```yaml
deal:
  mask: true
```

`mask: false` disables masking and lets every atom participate. `mask: true`
uses the default preprocessing array, `deal_mask`. A string value uses that
specific per-atom array name.

This first version only thresholds an existing per-atom signal.  It gives us a
stable preprocessing hook for later selectors such as QBC without mixing those
policies into the sparse-GP algebra.

## Masked Sparse-GP Prediction Direction

The current mask is applied around the model: the frame is still described by
the SGP, then atoms outside `deal_mask` are ignored by selection/update logic.
The next model-level step is to pass candidate atom indices from DEAL into the
SGP uncertainty call so local GP uncertainty is computed only for masked atoms.
The intended Python surface is:

```python
model.predict_uncertainty(atoms, candidate_atoms=mask.nonzero()[0])
```

The native side should preserve full-frame force/update compatibility while
allowing local uncertainty kernels to be evaluated for only those central atoms.

## Pieces Still From Upstream

The Python-side active-learning subset now lives in `deal.sgp`:

- `deal.sgp.atoms`
- `deal.sgp.calculator`
- `deal.sgp.sparse_gp`
- `deal.sgp.utils`

DEAL now builds its own native extension, `deal.sgp._C_deal_sgp`, and the
canonical Python bridge is `deal.sgp._C_sgp`.

The DEAL native extension currently exposes:

- `Structure`
- `SparseGP`
- `B2` descriptors
- `NormalizedDotProduct`
- local uncertainty prediction

The public Python API now uses `SGPConfig` and `SGPAtoms`; upstream-specific
public aliases have been removed from DEAL's active code.

## Uncertainty-Aware MD Hook

The next layer should be a small calculator that returns:

- predicted forces
- per-atom uncertainty
- a frame-level uncertainty score
- an action decision such as `continue`, `slow_down`, `select_for_dft`, or `stop`

That policy should sit above the model, not inside the GP algebra.  The GP
should answer "what is the uncertainty?"; the MD controller should decide what
to do with it.
