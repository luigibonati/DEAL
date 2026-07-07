# MACE

# Training 

Training is divided in two steps, similar to the swa recipe from MACE.

1. force_weight = 100, energy_weight = 1, lr = 0.01, first 800 epochs (w/early stopping) 
2. force_weight = 100, energy_weight = 1000, lr = 0.001, max 1000 epochs (w/early stopping) 

# Postprocessing

`mace_uncertainty.py` takes a LAMMPS/XYZ trajectory and evaluates the uncertainty on forces using an ensemble of MACE models.
