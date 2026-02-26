#!/usr/bin/env bash
set -euo pipefail

deal --file "../data/traj.xyz" --threshold 0.5
python -m deal.chemiscope_cli --trajectory "deal_selected.xyz" --output "deal_cli_chemiscope.json.gz"

test -s "deal_cli_chemiscope.json.gz"
