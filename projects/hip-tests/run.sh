#!/usr/bin/env sh

cd ./build/catch_tests

# HT_LOG_ENABLE=1 ctest -L gl
# HT_LOG_ENABLE=1 ctest -V -R Unit_hipDeviceMemAlloc_Functional
HT_LOG_ENABLE=1 ctest -N -V
