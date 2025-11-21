#!/bin/bash -e

ROCPROFILER_SDK_PATH=$(cd $(dirname ${BASH_SOURCE[0]})/../.. && pwd)

# echo "ROCPROFILER_SDK_PATH: ${ROCPROFILER_SDK_PATH}"

echo -e "Installing dependencies from: ${ROCPROFILER_SDK_PATH}/requirements.txt"
python3 -m pip install --user -r ${ROCPROFILER_SDK_PATH}/requirements.txt
