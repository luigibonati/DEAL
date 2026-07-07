#!/usr/bin/env bash
set -euo pipefail

python run_deal.py
python test_config.py
python test_masked_prediction.py
python test_runtime_info.py
python test_sgp_atoms.py
