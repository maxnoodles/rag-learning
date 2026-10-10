#!/bin/sh
set -eu
TASK_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
if [ ! -x "$TASK_ROOT/.venv/bin/python" ]; then
  echo '项目 Python 环境不存在；请参考 README.md 的环境说明。' >&2
  exit 1
fi
export TIKTOKEN_CACHE_DIR="$TASK_ROOT/artifacts/tiktoken-cache"
exec "$TASK_ROOT/.venv/bin/python" -u "$TASK_ROOT/run.py" "$@"
