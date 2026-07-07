#!/usr/bin/env bash
set -euo pipefail

curl --fail --location \
  'https://archive.materialscloud.org/records/ycbvx-knj69/files/fcu.xyz?download=1' \
  --output fcu.xyz
