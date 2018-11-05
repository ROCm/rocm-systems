#!/bin/bash



# these are required:
ROCM_DIR=/home/cfreehil/github/rocm_smi_lib/build
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
