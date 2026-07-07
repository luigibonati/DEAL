from ase.io import read,write
from mace.calculators import MACECalculator
from tqdm import tqdm
import numpy as np
from copy import copy
import torch

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", type=str,
                    help="path of the MACE model")
parser.add_argument("-t", "--type", type=str, default="MACE",
                    help="MACE type")
parser.add_argument("-f", "--file", type=str, default="traj.lammps",
                    help="trajectory file")
parser.add_argument("-o", "--outfile", type=str, default="traj-std.xyz",
                    help="output file name")
parser.add_argument("--stride", type=int, default=1,
                    help="stride of traj")


args = parser.parse_args()
print("MODEL  |", args.model)
print("TYPE   |", args.type)

# TRAJ
traj = read(args.file, index=f'::{args.stride}')
print(f"INFO   | read {args.file} ({len(traj)})")

# MODEL
calc = MACECalculator(model_paths=args.model,device='cuda',
                        model_type=args.type)
print(f"TYPE   | model loaded ({calc})")

# EVALUATE
for atoms in tqdm(traj):
    calc.calculate(atoms)
    atoms.calc = copy(calc)

    if calc.num_models>1:
        # |mean| and std forces over commitee
        fmod = np.abs(calc.results['forces'])
        fstd = calc.results['forces_comm'].std(axis=0)
        # calculate variance as maximum per atom component
        var = np.amax(fstd,axis=1,keepdims=True)
        atoms.set_array('force_std_comp_max', var.flatten())

# WRITE OUTPUT
write(args.outfile,traj)
print(f"INFO   | written {args.outfile} ({len(traj)})")
