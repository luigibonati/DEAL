#!/usr/bin/env bash
set -euo pipefail

# Standard CLI mode (threshold)
deal --file "../data/traj.xyz" --threshold 0.5 --mask false
python -m deal.chemiscope_cli --trajectory "deal_selected.xyz" --output "deal_cli_chemiscope.json.gz"

test -s "deal_cli_chemiscope.json.gz"

# Incremental CLI mode (max-selected) must work without a deal_mask array.
rm -f deal_selected.xyz
deal --file "../data/traj.xyz" --max-selected 5
