#!/bin/bash
set -euo pipefail

# Wrapper for dataset download script.
./scripts/load_data.sh "$@"
