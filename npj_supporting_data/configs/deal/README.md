## DEAL configuration files

This folder contains the config files for the DEAL (Data-Efficient Active Learning) scheme, which uses the uncertainty of local environments provided by the GP to identify a small set of non-redundant configurations from a trajectory (e.g. one composed of high-uncertainty configurations). In particular, we use FLARE to train the GP.

Two settings are available in the `configs/deal/`:

* `deal-pretrain-dft.yaml`: the selection is made using a GP model pretrained on DFT energies/forces while performing DFT calculations on the fly
* `deal-nodft.yaml`: train a model from scratch, using the MD energy/forces as labels, and only at the end perform the single-point calculations on the selected structures

These two strategies can deliver a very similar selection of structures (see SI of the manuscript), but the second one is much faster because no DFT calculations are performed serially; they can be done at the end in an embarrassingly parallel way.
