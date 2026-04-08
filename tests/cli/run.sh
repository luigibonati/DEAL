#!/usr/bin/env bash
set -euo pipefail

# Standard CLI mode (threshold)
deal --file "../data/traj.xyz" --threshold 0.5
python -m deal.chemiscope_cli --trajectory "deal_selected.xyz" --output "deal_cli_chemiscope.json.gz"

test -s "deal_cli_chemiscope.json.gz"

# Incremental CLI mode (max-selected)
rm -f deal_selected.xyz
deal --file "../data/traj.xyz" --max-selected 5

