# Formate without preselection

This example applies incremental DEAL selection directly to a formate
dehydrogenation trajectory, without an uncertainty-based preselection step. The
target is 140 structures.

Download the trajectory:

```bash
curl --fail --location \
  'https://archive.materialscloud.org/records/ycbvx-knj69/files/fcu.xyz?download=1' \
  --output traj.xyz
```

If you previously ran the old command, run this corrected command again to
replace the invalid `traj.xyz` file.

Run with the minimal YAML configuration:

```bash
deal -c input.yaml
```

The equivalent CLI invocation is:

```bash
deal --file traj.xyz --max-selected 140
```
