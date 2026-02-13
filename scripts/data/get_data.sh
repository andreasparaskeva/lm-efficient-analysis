#!/bin/bash
set -euo pipefail

# Wrapper for dataset download script.
./scripts/data/load_data.sh "$@"
