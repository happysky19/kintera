#!/usr/bin/env bash
# Portable replacement for the recovered run.sh, which carried one machine's absolute paths.
#   KINTERA_REPO=<kintera checkout at code_pin.yml> bash run_recovered.sh <script> [args...]
# probe.py writes its generated decks next to itself and its JSONL to stdout, as it always did.
set -euo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
: "${KINTERA_REPO:?set KINTERA_REPO to a kintera checkout at the pin in code_pin.yml}"
export PYTHONPATH="$here${PYTHONPATH:+:$PYTHONPATH}"   # one_cell.py does `from probe import *`
export OMP_NUM_THREADS=1
exec python -P "$here/$1" "${@:2}"
