#! /usr/bin/env bash

cmake -S projects/rccl -B projects/rccl/build \
  -DCMAKE_PREFIX_PATH=/opt/rocm \
  -DGPU_TARGETS=gfx950 \
  -DBUILD_TESTS=ON \  
  -DENABLE_ROCSHMEM_GIN=ON \
  -DROCSHMEM_SOURCE_DIR=projects/rocshmem/src
cmake --build projects/rccl/build --target rccl-UnitTestsFixtures -j$(nproc)

cd projects/rccl/build/test
./rccl-UnitTestsFixtures --gtest_filter='GinAnvilDeviceTest.*'

# ./rccl-UnitTestsFixtures

if [ 0 -eq 1 ]; then
cmake -S projects/rccl -B projects/rccl/build \
  -DBUILD_TESTS=ON \
  -DENABLE_ROCSHMEM=ON \
  -DROCSHMEM_SOURCE_DIR=projects/rocshmem/src \
  ...
fi

