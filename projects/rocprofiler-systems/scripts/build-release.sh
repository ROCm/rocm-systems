#!/bin/bash

VERSION=0.0.3
ROCM_VERSION=4.3.0

STANDARD_ARGS="-DCPACK_GENERATOR=STGZ -DCMAKE_BUILD_TYPE=Release -DHOSTTRACE_BUILD_DYNINST=ON -DTIMEMORY_BUILD_PORTABLE=ON"

cmake -B build-release/core ${STANDARD_ARGS} -DDYNINST_BUILD_{TBB,BOOST,ELFUTILS,LIBIBERTY}=ON -DDYNINST_USE_OpenMP=OFF -DHOSTTRACE_USE_MPI_HEADERS=ON -DHOSTTRACE_USE_ROCTRACER=OFF .
cmake --build build-release/core --target package --parallel 8
cp build-release/core/hosttrace-${VERSION}-Linux.sh build-release/hosttrace-${VERSION}-Linux.sh

cmake -B build-release/rocm-mpi ${STANDARD_ARGS} -DDYNINST_BUILD_{TBB,BOOST,ELFUTILS,LIBIBERTY}=ON -DDYNINST_USE_OpenMP=ON -DHOSTTRACE_USE_MPI_HEADERS=ON -DHOSTTRACE_USE_ROCTRACER=ON .
cmake --build build-release/rocm-mpi --target package --parallel 8
cp build-release/rocm-mpi/hosttrace-${VERSION}-Linux.sh build-release/hosttrace-${VERSION}-Linux-ROCm-${ROCM_VERSION}.sh
