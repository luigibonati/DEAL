# copy data from supporting data repository if not already present
if  [ -f traj-std-ev10.xyz ]; then
    echo "Trajectory file already present, skipping download."
else
    wget https://raw.githubusercontent.com/luigibonati/DEAL/refs/tags/v1.0.0/notebooks/2_convergence/N2_opes_outputs/traj-std-ev10.xyz
    wget https://raw.githubusercontent.com/luigibonati/DEAL/refs/tags/v1.1.0/npj_notebooks/notebooks/2_convergence/N2_opes_outputs/COLVAR
fi

