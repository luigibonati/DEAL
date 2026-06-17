# DEAL-Specific Model Roadmap

The goal is to stop treating FLARE as a full dependency surface and keep only
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

## Pieces Still Borrowed From FLARE

The Python-side active-learning subset now lives in `deal.sgp`:

- `deal.sgp.atoms`
- `deal.sgp.calculator`
- `deal.sgp.sparse_gp`
- `deal.sgp.utils`

DEAL now builds its own native extension, `deal.sgp._C_deal_sgp`, and the
compatibility bridge in `deal.sgp._C_flare` imports that extension first.
FLARE's compiled `_C_flare` is now only a fallback for old editable installs
that have not been rebuilt.

The DEAL native extension currently exposes:

- `Structure`
- `SparseGP`
- `B2` descriptors
- `NormalizedDotProduct`
- local uncertainty prediction

The next cleanup step is to rename the compatibility bridge and classes away
from FLARE terminology where that does not affect saved-model compatibility.

## Uncertainty-Aware MD Hook

The next layer should be a small calculator that returns:

- predicted forces
- per-atom uncertainty
- a frame-level uncertainty score
- an action decision such as `continue`, `slow_down`, `select_for_dft`, or `stop`

That policy should sit above the model, not inside the GP algebra.  The GP
should answer "what is the uncertainty?"; the MD controller should decide what
to do with it.
