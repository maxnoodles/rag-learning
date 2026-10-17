#!/bin/sh
set -eu
section_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
exec "$section_dir/../.venv/bin/python" "$section_dir/run.py" "$@"
