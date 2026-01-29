#!/usr/bin/env sh

OLD_PATH=$(pwd)
cd ..

export HIP_DIR="$(readlink -f hip)"
export CLR_DIR="$(readlink -f clr)"
export HIPTESTS_DIR="$(readlink -f hip-tests)"
export ROCM_PATH=/opt/rocm

export CMAKE_INSTALL_PREFIX=~/Projects/rocm-systems/rocm

cd $OLD_PATH

# mkdir -p build; cd build
# cmake ../catch/ -DHIP_PLATFORM=amd  \
#     -DCMAKE_INSTALL_PREFIX=$(realpath "$CMAKE_INSTALL_PREFIX")
# VERBOSE=1 make -j$(nproc) build_tests

mkdir -p build_samples; cd build_samples
# cmake ../samples/ -DHIP_PATH=/path/to/hip -DROCM_PATH=/opt/rocm
cmake ../samples -DHIP_PLATFORM=amd \
    -DCMAKE_INSTALL_PREFIX=$(realpath "$CMAKE_INSTALL_PREFIX") \
    -DCMAKE_BUILD_TYPE=Release
VERBOSE=1 make build_samples

# ctest # run tests
