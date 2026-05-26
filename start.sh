#!/bin/zsh
cd "$(dirname "$0")"

if [ ! -d ".venv" ]; then
    echo "First run: creating virtual environment..."
    python3 -m venv .venv
    .venv/bin/pip install -r requirements.txt -q
    echo "Dependencies installed."
fi

exec .venv/bin/python3 server.py
