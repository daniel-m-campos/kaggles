#!/bin/bash -e

if [ -d .venv ]; then
    rm -rf .venv
fi

PYTHON_VERSION="3.11"

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
SOURCE_ROOT=$DIR/..
cd $SOURCE_ROOT || exit

pyenv install -s $PYTHON_VERSION
pyenv local $PYTHON_VERSION
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
pre-commit install
