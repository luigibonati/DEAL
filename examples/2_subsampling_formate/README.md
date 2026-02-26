## Formate dehydrogenation on Cu
- Ref.: Batzner et al., *Nature Communications*, 13,2453 (2022), doi:[10.1038/s41467-022-29939-5](https://doi.org/10.1038/s41467-022-29939-5).
- Data: Formate dehydrogenation on Cu(110), series of MD simulations initiated from NEB images.
- Summary: 
    1. Filter the structures based on GP predictive variance. 
    2. Create a chemiscope file and visualize selected structures.

___ 

### INSTRUCTION

#### Input
Get trajectory (xyz) from supporting data 
```bash
cd a_input 
bash get_data.sh
```

#### Colvar (optional)
Evaluate relevant CVs (e.g. coordination numbers) using PLUMED
```bash
plumed driver --plumed plumed.dat --ixyz ../input/fcu.xyz --length-units A --box 10.638,10.03,30.0
cd ..
```

#### Selection
Run DEAL with different thresholds
```bash
deal -c input.yaml
```
Consider running it on a HPC cluster or on a large-memory machine.
An example SLURM script to run DEAL on Daint@Alps-CSCS HPC system is provdied for [`N2 example`](../1_activelearning_N2/README.md).

#### Chemiscope
Create chemiscope files from DEAL outputs
```bash
deal-chemiscope --trajectory b_selection/deal_0.15_selected.xyz --colvar a_input/COLVAR
```

#### Analyze results
The selection can be analyzed with the jupyter notebook [`formate-subsampling.ipynb`](`formate-subsampling.ipynb`).

The generated `*_chemiscope.json.gz` files can be directly visualized online at https://chemiscope.org/


