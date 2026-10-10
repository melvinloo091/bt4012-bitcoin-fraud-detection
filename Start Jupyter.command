#!/bin/zsh
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR" || exit 1
if [[ ! -x "$PROJECT_DIR/.venv/bin/python" ]]; then
  print "The project Python environment is missing. See README.md for setup."
  read "?Press Enter to close."
  exit 1
fi
export JUPYTER_CONFIG_DIR="$PROJECT_DIR/.jupyter-config"
export JUPYTER_RUNTIME_DIR="$PROJECT_DIR/.jupyter-runtime"
export IPYTHONDIR="$PROJECT_DIR/.ipython"
export MPLCONFIGDIR="$PROJECT_DIR/.cache/matplotlib"
exec "$PROJECT_DIR/.venv/bin/python" -m jupyterlab --notebook-dir="$PROJECT_DIR" --ip=127.0.0.1 --ServerApp.port=8888 --ServerApp.port_retries=20
