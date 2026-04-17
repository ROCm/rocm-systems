#!/bin/bash
# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

SCRIPT_DIR_NAME=$(dirname -- "${BASH_SOURCE[0]}")
VENV_NAME=.venv

# Create and activate virtual environment
python3 -m venv ${VENV_NAME}
source ${VENV_NAME}/bin/activate

# Upgrade pip
python3 -m pip install --upgrade pip

# Install requirements for each backend
python3 -m pip install -r ${SCRIPT_DIR_NAME}/requirements.txt
