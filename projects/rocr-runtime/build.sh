#!/usr/bin/env sh

OLD_PATH=$(pwd)
cd ..

export HIP_DIR="$(readlink -f hip)"
export CLR_DIR="$(readlink -f clr)"
export HIPTESTS_DIR="$(readlink -f hip-tests)"
export ROCM_PATH=/opt/rocm

export CMAKE_INSTALL_PREFIX=~/Projects/rocm-systems/rocm

cd $OLD_PATH

mkdir -p build

cmake -S . -B build \
    -DCMAKE_INSTALL_PREFIX=$(realpath "$CMAKE_INSTALL_PREFIX")

make -C build -j 6
make -C build install
