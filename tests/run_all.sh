#!/usr/bin/env bash
set -euo pipefail

for dir in */ ; do
    # check if it is a directory
    if [ -d "$dir" ]; then
        # check if run.sh exists and is executable
        if [ -f "$dir/run.sh" ]; then
            echo "==> TEST: $dir"
            ( cd "$dir" && bash -euo pipefail run.sh )
        fi
    fi
done
