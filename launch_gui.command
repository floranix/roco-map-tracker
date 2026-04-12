#!/bin/zsh
cd "$(dirname "$0")"

if [ -x ".venv/bin/python" ]; then
  exec ./.venv/bin/python main.py --gui
fi

exec python3 main.py --gui
