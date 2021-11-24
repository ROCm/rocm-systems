#!/bin/bash

: ${VERSION:=0.0.3}
: ${ROCM_VERSION:=4.3.0}
: ${NJOBS:=8}

STANDARD_ARGS="-DCPACK_GENERATOR=STGZ -DCMAKE_BUILD_TYPE=Release -DOMNITRACE_BUILD_DYNINST=ON -DTIMEMORY_BUILD_PORTABLE=ON"

cmake -B build-release/core ${STANDARD_ARGS} -DDYNINST_BUILD_{TBB,BOOST,ELFUTILS,LIBIBERTY}=ON -DDYNINST_USE_OpenMP=OFF -DOMNITRACE_USE_MPI_HEADERS=ON -DOMNITRACE_USE_ROCTRACER=OFF -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=OFF -DOMNITRACE_MAX_THREADS=2048 .
cmake --build build-release/core --target package --parallel 8
cp build-release/core/omnitrace-${VERSION}-Linux.sh build-release/omnitrace-${VERSION}-Linux.sh

cmake -B build-release/rocm-mpi ${STANDARD_ARGS} -DDYNINST_BUILD_{TBB,BOOST,ELFUTILS,LIBIBERTY}=ON -DDYNINST_USE_OpenMP=ON -DOMNITRACE_USE_MPI_HEADERS=ON -DOMNITRACE_USE_ROCTRACER=ON -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=OFF -DOMNITRACE_MAX_THREADS=2048 .
cmake --build build-release/rocm-mpi --target package --parallel 8
cp build-release/rocm-mpi/omnitrace-${VERSION}-Linux.sh build-release/omnitrace-${VERSION}-Linux-ROCm-${ROCM_VERSION}.sh
