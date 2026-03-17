#!/bin/bash

full=${full:-0}

export HIP_DIR=$(readlink -f ../hip)
echo "HIP_DIR: $HIP_DIR"
mkdir -p build
pushd build

if [[ ${full} -eq 1 ]]; then
  rm -rf *
  cmake .. -DCMAKE_BUILD_TYPE=Release -DCLR_BUILD_HIP=ON -DHIP_COMMON_DIR=$HIP_DIR -D__HIP_ENABLE_PCH=OFF
fi

make -j