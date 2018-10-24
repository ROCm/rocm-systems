#!/bin/bash



# these are required:
ROCM_DIR=/home/cfreehil/git/compute/out/ubuntu-16.04/16.04
#ROCM_DIR=/opt/rocm

mkdir -p build
cd build

cmake -DROCRTST_BLD_TYPE=$ROCRTST_BLD_TYPE \
    -DROCM_DIR=$ROCM_DIR \
    -DROCRTST_BLD_TYPE="Debug" \
    ..

echo "Executing \"make\"..."
make

cd ..
