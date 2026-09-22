#!/bin/bash

SCRIPT_PATH=$(cd $(dirname ${BASH_SOURCE[0]}) && pwd)
ROCPROFILER_SDK_PATH="$(cd ${SCRIPT_PATH}/../.. && pwd)"

${SCRIPT_PATH}/install-deps.sh

echo -e "Redirecting to location: $ROCPROFILER_SDK_PATH"
cd ${ROCPROFILER_SDK_PATH}

echo -e "Configuring rocprofiler-sdk: ${ROCPROFILER_SDK_PATH}/build"
cmake -B build -DROCPROFILER_BUILD_{CI,TESTS,SAMPLES}=ON -DROCPROFILER_ENABLE_CLANG_TIDY=ON "${@}"

echo -e "Building rocprofiler-sdk: ${ROCPROFILER_SDK_PATH}/build"
cmake --build build --target all --parallel $(nproc)
