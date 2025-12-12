#!/bin/bash

#==================================================
# Configuration options
#==================================================
# ROCm location
export MY_ROCM_DIR=/opt/rocm

# hip location
export MY_HIP_DIR=/opt/rocm/hip

# UCX configuration options
export MY_UCX_SOURCE=https://github.com/openucx/ucx.git
export MY_UCX_BRANCH=v1.15.x
export MY_UCX_DIR=$PWD/ucx/install

# OpenMPI configuration options
export MY_OMPI_SOURCE=https://github.com/open-mpi/ompi.git
export MY_OMPI_BRANCH=v5.0.x
export MY_OMPI_DIR=$PWD/ompi/install

# HIP MPI testsuite
export MY_HIP_MPI_TEST_SOURCE=git@github.com:ROCm/hip-mpi-testsuite.git
export MY_HIP_MPI_TEST_DIR=$PWD/hip-mpi-testsuite

# OSU Benchmark configuration options
export MY_OSU_SOURCE=https://mvapich.cse.ohio-state.edu/download/mvapich
export MY_OSU_FILE=osu-micro-benchmarks-7.2.tar.gz
export MY_OSU_DIR=$PWD/osu-micro-benchmarks-7.2

# RCCL configuration options
export MY_RCCL_SOURCE=https://github.com/ROCm/rccl.git
export MY_RCCL_BRANCH=develop
export MY_RCCL_DIR=$PWD/rccl

# RCCL-tests configuration options
export MY_RCCL_TESTS_SOURCE=https://github.com/ROCm/rccl-tests.git
export MY_RCCL_TESTS_BRANCH=master
export MY_RCCL_TESTS_DIR=$PWD/rccl-tests

# Transferbench configuration options
export MY_TRANSFERBENCH_SOURCE=https://github.com/ROCm/TransferBench.git
export MY_TRANSFERBENCH_BRANCH=develop
export MY_TRANSFERBENCH_DIR=$PWD/TransferBench

# Results location
export MY_RESULTS_DIR=$PWD/output

# Compilation location
export MY_COMPILATION_DIR=$PWD/output

mkdir $PWD/output

# Step 0: Calculate the number of GPUs
NUMBER_OF_GPUS="$(/opt/rocm/bin/rocm_agent_enumerator | wc -l)"
NUMBER_OF_GPUS=$((NUMBER_OF_GPUS-1))
echo "The number of GPUs is : $NUMBER_OF_GPUS"

# Step 1: Build UCX with ROCm support
echo "Step 1: Install UCX?"
read -p '(y/n) ' RESPONSE
if [ "$RESPONSE" = "y" ]; then    
    Build_UCX=PASSED
    echo "Cloning fresh copy of UCX: $MY_UCX_BRANCH"
    rm -rf ucx || Build_UCX=FAILED
    git clone $MY_UCX_SOURCE -b $MY_UCX_BRANCH || Build_UCX=FAILED
    cd ucx || Build_UCX=FAILED
    ./autogen.sh | tee -a $MY_COMPILATION_DIR/compile.log || Build_UCX=FAILED
    mkdir -p build || Build_UCX=FAILED
    cd build || Build_UCX=FAILED
    ../configure --prefix=$MY_UCX_DIR --with-rocm=$MY_ROCM_DIR --without-knem --enable-gtest | tee -a $MY_COMPILATION_DIR/compile.log || Build_UCX=FAILED
    make -j | tee -a $MY_COMPILATION_DIR/compile.log || Build_UCX=FAILED
    make install | tee -a $MY_COMPILATION_DIR/compile.log || Build_UCX=FAILED
    cd ../.. || Build_UCX=FAILED
else Build_UCX=SKIPPED
fi

# Step 2: UCX unit tests
echo "Step 2: Run UCX gtests?"
read -p '(y/n) ' RESPONSE
if [ "$RESPONSE" = "y" ]; then
    Run_UCX_gtests=PASSED
    export LD_LIBRARY_PATH=$MY_UCX_DIR/lib:$MY_ROCM_LIB/lib:$LD_LIBRARY_PATH
    cd $MY_UCX_DIR/../build || Run_UCX_gtests=FAILED
    ./test/gtest/gtest --gtest_filter=*rocm*:-*Pitch* | tee -a $MY_RESULTS_DIR/results.log || Run_UCX_gtests=FAILED
    cd ../../ || Run_UCX_gtests=FAILED
else Run_UCX_gtests=SKIPPED 
fi

# Step 3: Install OpenMPI with UCX support
echo "Step 3: Install OpenMPI?"
read -p '(y/n) ' RESPONSE
if [ "$RESPONSE" = "y" ]; then
    Install_OpenMPI=PASSED
    echo "Cloning fresh copy of OpenMPI: $MY_OMPI_BRANCH"
    rm -rf ompi || Install_OpenMPI=FAILED
    git clone --recursive $MY_OMPI_SOURCE -b $MY_OMPI_BRANCH || Install_OpenMPI=FAILED
    cd ompi || Install_OpenMPI=FAILED
    ./autogen.pl | tee -a $MY_COMPILATION_DIR/compile.log || Install_OpenMPI=FAILED
    mkdir -p build || Install_OpenMPI=FAILED
    cd build || Install_OpenMPI=FAILED
    ../configure --prefix=$MY_OMPI_DIR --with-rocm=$MY_ROCM_DIR --with-ucx=$MY_UCX_DIR --disable-sphinx --disable-oshmem --disable-mpi-fortran --with-prrte=internal --with-hwloc=internal --with-libevent=internal | tee -a $MY_COMPILATION_DIR/compile.log || Install_OpenMPI=FAILED
    make -j | tee -a $MY_COMPILATION_DIR/compile.log || Install_OpenMPI=FAILED
    make install | tee -a $MY_COMPILATION_DIR/compile.log || Install_OpenMPI=FAILED
    cd ../.. || Install_OpenMPI=FAILED
else Install_OpenMPI=SKIPPED
fi

# Step 4: Install hip-mpi-testsuite benchmarks
echo "Step 4: Install hip-mpi-testsuite benchmarks?"
read -p '(y/n) ' RESPONSE
if [ "$RESPONSE" = "y" ]; then
    Install_hip_mpi_testsuite=PASSED
    rm -rf $MY_HIP_MPI_TEST_DIR || Install_hip_mpi_testsuite=FAILED
    git clone $MY_HIP_MPI_TEST_SOURCE | tee -a $MY_COMPILATION_DIR/compile.log || Install_hip_mpi_testsuite=FAILED
    cd $MY_HIP_MPI_TEST_DIR || Install_hip_mpi_testsuite=FAILED
    export LD_LIBRARY_PATH=$MY_OMPI_DIR/lib:$MY_UCX_DIR/lib:$MY_ROCM_DIR/lib:$LD_LIBRARY_PATH
    export PATH=$MY_OMPI_DIR/bin:$PATH
    ./configure CXX=mpiCC --with-rocm=$MY_ROCM_DIR | tee -a $MY_COMPILATION_DIR/compile.log || Install_hip_mpi_testsuite=FAILED
    make -j | tee -a $MY_COMPILATION_DIR/compile.log || Install_hip_mpi_testsuite=FAILED
    cd .. || Install_hip_mpi_testsuite=FAILED
else Install_hip_mpi_testsuite=SKIPPED
fi

# Step 5: Run hip-mpi-testsuite benchmarks
echo "Step 5: Run hip-mpi-testsuite benchmarks?"
read -p '(y/n) ' RESPONSE
if [ "$RESPONSE" = "y" ]; then
    Run_hip_mpi_testsuite=PASSED
    cd $MY_HIP_MPI_TEST_DIR || Run_hip_mpi_testsuite=FAILED
    export LD_LIBRARY_PATH=$MY_OMPI_DIR/lib:$MY_UCX_DIR/lib:$MY_ROCM_DIR/lib:$LD_LIBRARY_PATH
    export PATH=$MY_OMPI_DIR/bin:$PATH
    cd scripts/ || Run_hip_mpi_testsuite=FAILED
    ./run_all.sh | tee -a $MY_RESULTS_DIR/results.log || Run_hip_mpi_testsuite=FAILED
    cd ../.. || Run_hip_mpi_testsuite=FAILED
else Run_hip_mpi_testsuite=SKIPPED
fi

# Step 6: Install OSU benchmarks
echo "Step 6: Install OSU benchmarks?"
read -p '(y/n) ' RESPONSE
if [ "$RESPONSE" = "y" ]; then
    Install_OSU=PASSED
    echo "Fetching fresh copy of OSU benchmarks"
    rm -rf $MY_OSU_DIR || Install_OSU=FAILED
    wget $MY_OSU_SOURCE/$MY_OSU_FILE || Install_OSU=FAILED
    tar -xzf $MY_OSU_FILE || Install_OSU=FAILED

    cd $MY_OSU_DIR || Install_OSU=FAILED
    autoreconf -ivf | tee -a $MY_COMPILATION_DIR/compile.log || Install_OSU=FAILED
    export LD_LIBRARY_PATH=$MY_OMPI_DIR/lib:$MY_UCX_DIR/lib:$MY_ROCM_DIR/lib:$LD_LIBRARY_PATH
    export PATH=$MY_OMPI_DIR/bin:$PATH
    ./configure --enable-rocm --with-rocm=$MY_ROCM_DIR CC=$MY_OMPI_DIR/bin/mpicc CXX=$MY_OMPI_DIR/bin/mpicxx LDFLAGS="-L$MY_OMPI_DIR/lib/ -lmpi -L$MY_ROCM_DIR/lib/ -lamdhip64" | tee -a $MY_COMPILATION_DIR/compile.log || Install_OSU=FAILED
    make -j 8 | tee -a $MY_COMPILATION_DIR/compile.log || Install_OSU=FAILED
    cd .. || Install_OSU=FAILED
else Install_OSU=SKIPPED
fi

# Step 7: OSU pt2pt bw tests
echo "Step 7: Run OSU pt2pt bw benchmarks?"
read -p '(y/n) ' RESPONSE
if [ "$RESPONSE" = "y" ]; then
    OSU_pt2pt_bw_test=PASSED
    cd $MY_OSU_DIR || OSU_pt2pt_bw_test=FAILED
    export LD_LIBRARY_PATH=$MY_OMPI_DIR/lib:$MY_UCX_DIR/lib:$MY_ROCM_DIR/lib:$LD_LIBRARY_PATH
    export PATH=$MY_OMPI_DIR/bin:$PATH
        for MEM1 in D H; do
        for MEM2 in D H; do
            $MY_OMPI_DIR/bin/mpirun -np 2 --mca osc ucx --mca pml ucx -x LD_LIBRARY_PATH ./c/mpi/pt2pt/standard/osu_bw $MEM1 $MEM2 | tee -a $MY_RESULTS_DIR/results.log || OSU_pt2pt_bw_test=FAILED
        done
        done
    cd ../
else OSU_pt2pt_bw_test=SKIPPED
fi

# Step 8: OSU pt2pt latency tests
echo "Step 8: Run OSU pt2pt latency benchmarks?"
read -p '(y/n) ' RESPONSE
if [ "$RESPONSE" = "y" ]; then
    OSU_pt2pt_latency_test=PASSED
    cd $MY_OSU_DIR || OSU_pt2pt_latency_test=FAILED
    export LD_LIBRARY_PATH=$MY_OMPI_DIR/lib:$MY_UCX_DIR/lib:$MY_ROCM_DIR/lib:$LD_LIBRARY_PATH
    export PATH=$MY_OMPI_DIR/bin:$PATH
        for MEM1 in D H; do
        for MEM2 in D H; do
            $MY_OMPI_DIR/bin/mpirun -np 2 --mca osc ucx --mca pml ucx -x LD_LIBRARY_PATH ./c/mpi/pt2pt/standard/osu_latency $MEM1 $MEM2 | tee -a $MY_RESULTS_DIR/results.log || OSU_pt2pt_latency_test=FAILED
        done
        done
    cd ../
else OSU_pt2pt_latency_test=SKIPPED
fi

# Step 9: OSU collective tests
echo "Step 9: Run OSU collective benchmarks?"
read -p '(y/n) ' RESPONSE
if [ "$RESPONSE" = "y" ]; then
    OSU_collective_tests=PASSED
    cd $MY_OSU_DIR || OSU_collective_tests=FAILED
    export LD_LIBRARY_PATH=$MY_OMPI_DIR/lib:$MY_UCX_DIR/lib:$MY_ROCM_DIR/lib:$LD_LIBRARY_PATH
    export PATH=$MY_OMPI_DIR/bin:$PATH
       $MY_OMPI_DIR/bin/mpirun --mca coll ^hcoll,han,adapt --mca osc ucx --mca pml ucx -x LD_LIBRARY_PATH -np $NUMBER_OF_GPUS ./c/mpi/collective/blocking/osu_bcast -m :33554432 -d rocm | tee -a $MY_RESULTS_DIR/results.log || OSU_collective_tests=FAILED
       $MY_OMPI_DIR/bin/mpirun --mca coll ^hcoll,han,adapt --mca osc ucx --mca pml ucx -x LD_LIBRARY_PATH -np $NUMBER_OF_GPUS ./c/mpi/collective/blocking/osu_reduce -m :33554432 -d rocm | tee -a $MY_RESULTS_DIR/results.log || OSU_collective_tests=FAILED
       $MY_OMPI_DIR/bin/mpirun --mca coll ^hcoll,han,adapt --mca osc ucx --mca pml ucx -x LD_LIBRARY_PATH -np $NUMBER_OF_GPUS ./c/mpi/collective/blocking/osu_gather -m :33554432 -d rocm | tee -a $MY_RESULTS_DIR/results.log || OSU_collective_tests=FAILED
       $MY_OMPI_DIR/bin/mpirun --mca coll ^hcoll,han,adapt --mca osc ucx --mca pml ucx -x LD_LIBRARY_PATH -np $NUMBER_OF_GPUS ./c/mpi/collective/blocking/osu_scatter -m :33554432 -d rocm | tee -a $MY_RESULTS_DIR/results.log || OSU_collective_tests=FAILED
       $MY_OMPI_DIR/bin/mpirun --mca coll ^hcoll,han,adapt --mca osc ucx --mca pml ucx -x LD_LIBRARY_PATH -np $NUMBER_OF_GPUS ./c/mpi/collective/blocking/osu_allgather -m :33554432 -d rocm | tee -a $MY_RESULTS_DIR/results.log || OSU_collective_tests=FAILED
       $MY_OMPI_DIR/bin/mpirun --mca coll ^hcoll,han,adapt --mca osc ucx --mca pml ucx -x LD_LIBRARY_PATH -np $NUMBER_OF_GPUS ./c/mpi/collective/blocking/osu_alltoall -m :33554432 -d rocm | tee -a $MY_RESULTS_DIR/results.log || OSU_collective_tests=FAILED
    cd ../ || OSU_collective_tests=FAILED
else OSU_collective_tests=SKIPPED
fi

# Step 10: Install RCCL
echo "Step 10: Install RCCL?"
read -p '(y/n) ' RESPONSE
if [ "$RESPONSE" = "y" ]; then
    Install_RCCL=PASSED
    rm -rf $MY_RCCL_DIR || Install_RCCL=FAILED
    git clone $MY_RCCL_SOURCE -b $MY_RCCL_BRANCH || Install_RCCL=FAILED
    cd $MY_RCCL_DIR || Install_RCCL=FAILED
    export LD_LIBRARY_PATH=$MY_ROCM_DIR/lib:$LD_LIBRARY_PATH
    ./install.sh -dt | tee -a $MY_COMPILATION_DIR/compile.log || Install_RCCL=FAILED
    cd .. || Install_RCCL=FAILED
else Install_RCCL=SKIPPED
fi

# Step 11: Run RCCL Unittests
echo "Step 11: Run RCCL Unittests?"
read -p '(y/n) ' RESPONSE
if [ "$RESPONSE" = "y" ]; then
    Run_RCCL_Unittests=PASSED
    cd $MY_RCCL_DIR || Run_RCCL_Unittests=FAILED
    export LD_LIBRARY_PATH=$MY_RCCL_DIR/build/release:$MY_ROCM_DIR/lib:$LD_LIBRARY_PATH
    HSA_FORCE_FINE_GRAIN_PCIE=1 ./build/release/test/rccl-UnitTests | tee -a $MY_RESULTS_DIR/results.log || Run_RCCL_Unittests=FAILED
    cd .. || Run_RCCL_Unittests=FAILED
else Run_RCCL_Unittests=SKIPPED
fi

# Step 12: Install RCCL-tests
echo "Step 12: Install RCCL-tests?"
read -p '(y/n) ' RESPONSE
if [ "$RESPONSE" = "y" ]; then
    Install_RCCL_tests=PASSED
    rm -rf $MY_RCCL_TESTS_DIR || Install_RCCL_tests=FAILED
    git clone $MY_RCCL_TESTS_SOURCE -b $MY_RCCL_TESTS_BRANCH || Install_RCCL_tests=FAILED
    cd $MY_RCCL_TESTS_DIR || Install_RCCL_tests=FAILED
    export LD_LIBRARY_PATH=$MY_ROCM_DIR/lib:$LD_LIBRARY_PATH
    make MPI=1 MPI_HOME=$MY_OMPI_DIR HIP_HOME=$MY_HIP_DIR RCCL_HOME=$MY_RCCL_DIR/build/release | tee -a $MY_COMPILATION_DIR/compile.log || Install_RCCL_tests=FAILED
    cd .. || Install_RCCL_tests=FAILED
else Install_RCCL_tests=SKIPPED
fi

# Step 13: Run RCCL-tests
echo "Step 13: Run RCCL-tests?"
read -p '(y/n) ' RESPONSE
if [ "$RESPONSE" = "y" ]; then
    Run_RCCL_tests=PASSED
    cd $MY_RCCL_TESTS_DIR || Run_RCCL_tests=FAILED
    export LD_LIBRARY_PATH=$MY_OMPI_DIR/lib:$MY_RCCL_DIR/build/release:$LD_LIBRARY_PATH
    export PATH=$MY_OMPI_DIR/bin:$MY_ROCM_DIR/bin:$PATH
    set -x
    HSA_FORCE_FINE_GRAIN_PCIE=1 mpirun -np $NUMBER_OF_GPUS ./build/broadcast_perf -b 16 -e 4G -f 2 -g 1 -c 1 | tee -a $MY_RESULTS_DIR/results.log || Run_RCCL_tests=FAILED
    HSA_FORCE_FINE_GRAIN_PCIE=1 mpirun -np $NUMBER_OF_GPUS ./build/reduce_perf -b 16 -e 4G -f 2 -g 1 -c 1 | tee -a $MY_RESULTS_DIR/results.log || Run_RCCL_tests=FAILED
    HSA_FORCE_FINE_GRAIN_PCIE=1 mpirun -np $NUMBER_OF_GPUS ./build/scatter_perf -b 16 -e 4G -f 2 -g 1 -c 1 | tee -a $MY_RESULTS_DIR/results.log || Run_RCCL_tests=FAILED
    HSA_FORCE_FINE_GRAIN_PCIE=1 mpirun -np $NUMBER_OF_GPUS ./build/all_gather_perf -b 16 -e 4G -f 2 -g 1 -c 1 | tee -a $MY_RESULTS_DIR/results.log || Run_RCCL_tests=FAILED
    HSA_FORCE_FINE_GRAIN_PCIE=1 mpirun -np $NUMBER_OF_GPUS ./build/sendrecv_perf -b 16 -e 4G -f 2 -g 1 -c 1 | tee -a $MY_RESULTS_DIR/results.log || Run_RCCL_tests=FAILED
    HSA_FORCE_FINE_GRAIN_PCIE=1 mpirun -np $NUMBER_OF_GPUS ./build/gather_perf -b 16 -e 4G -f 2 -g 1 -c 1 | tee -a $MY_RESULTS_DIR/results.log || Run_RCCL_tests=FAILED
    HSA_FORCE_FINE_GRAIN_PCIE=1 mpirun -np $NUMBER_OF_GPUS ./build/alltoall_perf -b 16 -e 4G -f 2 -g 1 -c 1 | tee -a $MY_RESULTS_DIR/results.log || Run_RCCL_tests=FAILED
    HSA_FORCE_FINE_GRAIN_PCIE=1 mpirun -np $NUMBER_OF_GPUS ./build/all_reduce_perf -b 16 -e 4G -f 2 -g 1 -c 1 | tee -a $MY_RESULTS_DIR/results.log || Run_RCCL_tests=FAILED
    cd .. || Run_RCCL_tests=FAILED
else Run_RCCL_tests=SKIPPED
fi

# Step 14: Install and Run TransferBench
echo "Step 14: Run TransferBench?"
read -p '(y/n) ' RESPONSE
if [ "$RESPONSE" = "y" ]; then
    Install_Run_TransferBench=PASSED
    rm -rf $MY_TRANSFERBENCH_DIR || Install_Run_TransferBench=FAILED
    git clone $MY_TRANSFERBENCH_SOURCE -b $MY_TRANSFERBENCH_BRANCH || Install_Run_TransferBench=FAILED
    cd $MY_TRANSFERBENCH_DIR || Install_Run_TransferBench=FAILED
    mkdir build || Install_Run_TransferBench=FAILED
    cd build || Install_Run_TransferBench=FAILED
    CXX=$MY_ROCM_DIR/bin/hipcc cmake .. | tee -a $MY_COMPILATION_DIR/compile.log || Install_Run_TransferBench=FAILED
    make | tee -a $MY_COMPILATION_DIR/compile.log || Install_Run_TransferBench=FAILED
    cd .. || Install_Run_TransferBench=FAILED
    ./TransferBench | tee -a $MY_RESULTS_DIR/results.log || Install_Run_TransferBench=FAILED
    ./TransferBench p2p | tee -a $MY_RESULTS_DIR/results.log || Install_Run_TransferBench=FAILED
    cd .. || Install_Run_TransferBench=FAILED
else Install_Run_TransferBench=SKIPPED
fi

echo "Step 1: Install UCX?: $Build_UCX"
echo "Step 2: Run UCX gtests? : $Run_UCX_gtests"
echo "Step 3: Install OpenMPI? : $Install_OpenMPI"
echo "Step 4: Install hip-mpi-testsuite benchmarks? : $Install_hip_mpi_testsuite"
echo "Step 5: Run hip-mpi-testsuite benchmarks? : $Run_hip_mpi_testsuite"
echo "Step 6: Install OSU benchmarks? : $Install_OSU"
echo "Step 7: Run OSU pt2pt bw benchmarks? : $OSU_pt2pt_bw_test"
echo "Step 8: Run OSU pt2pt latency benchmarks? : $OSU_pt2pt_latency_test"
echo "Step 9: Run OSU collective benchmarks?: $OSU_collective_tests"
echo "Step 10: Install RCCL? : $Install_RCCL"
echo "Step 11: Run RCCL Unittests? : $Run_RCCL_Unittests"
echo "Step 12: Install RCCL-tests? : $Install_RCCL_tests"
echo "Step 13: Run RCCL-tests? : $Run_RCCL_tests"
echo "Step 14: Run TransferBench? : $Install_Run_TransferBench"


