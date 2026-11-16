# N₂ with preselection

This example derives an atom mask from the per-atom `force_std_comp_max`
uncertainty and then runs incremental DEAL selection with a target of 140
structures.

Download the trajectory:

```bash
curl -L \
  https://raw.githubusercontent.com/luigibonati/DEAL/refs/tags/v1.0.0/notebooks/2_convergence/N2_opes_outputs/traj-std-ev10.xyz \
  -o traj.xyz
```

Run with the minimal YAML configuration:

```bash
deal -c input.yaml
```

The equivalent CLI invocation is:

```bash
deal --file traj.xyz \
  --max-selected 140 \
  --preprocess-key force_std_comp_max
```

Because no fixed preprocessing threshold is specified, DEAL uses its automatic
frame and atom thresholds. The YAML also lowers `lower_factor` to 0.9 for a
broader frame preselection; this option has no CLI flag, so the CLI invocation
uses the default (1.1). The resulting mask is consumed in memory.
