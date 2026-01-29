#!/usr/bin/env sh

# MUST RUN THE FOLLOWING IN THE PARENT FOLDER OF HIP, CLR, AND HIP-TESTS
# https://amd.atlassian.net/wiki/spaces/CLRT/pages/1047954012/Onboarding+Setting+up+on+a+Linux+Workstation

OLD_PATH=$(pwd)
cd ..

export HIP_DIR="$(readlink -f hip)"
export CLR_DIR="$(readlink -f clr)"
export HIPTESTS_DIR="$(readlink -f hip-tests)"
# export ROCM_PATH=/opt/rocm
export ROCM_PATH=/opt/rocm

export CMAKE_INSTALL_PREFIX=~/Projects/rocm-systems/rocm

cd $OLD_PATH

mkdir -p build

cmake -S . -B build \
    -DHIP_COMMON_DIR=$(realpath "$HIP_DIR")                     \
    -DCMAKE_PREFIX_PATH=/opt/rocm/                              \
    -DCMAKE_INSTALL_PREFIX=$(realpath "$CMAKE_INSTALL_PREFIX")  \
    -DCLR_BUILD_HIP=ON                                          \
    -DCLR_BUILD_OCL=OFF                                         \
    -DHIP_PLATFORM=amd                                          \
    -DCMAKE_BUILD_TYPE=Debug 2>&1 | tee cmake.log

make -C build -j 6
make -C build install
