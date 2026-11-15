#
# rocprofv3 kernel replay under MPI: every rank replays its own rank-local kernels while kernels
# the filters exclude (stand-ins for collective communication kernels) run exactly once.
# Disabled (not built) when no MPI implementation is available.
#
cmake_minimum_required(VERSION 3.21.0 FATAL_ERROR)

project(
    rocprofiler-sdk-tests-rocprofv3-kernel-replay-mpi
    LANGUAGES CXX
    VERSION 0.0.0)

find_package(rocprofiler-sdk REQUIRED)
find_package(Python3 REQUIRED)
find_package(MPI COMPONENTS CXX)

if(NOT MPI_CXX_FOUND OR NOT MPIEXEC_EXECUTABLE)
    message(STATUS "kernel-replay-mpi tests disabled: MPI not found")
    return()
endif()

string(REPLACE "LD_PRELOAD=" "" PRELOAD_ENV "${ROCPROFILER_MEMCHECK_PRELOAD_ENV}")

if(PRELOAD_ENV)
    set(PRELOAD_ARGS --suppress-marker-preload --preload ${PRELOAD_ENV})
else()
    set(PRELOAD_ARGS)
endif()

set(MPI_RANKS 2)
set(ITERS 4)
set(PASSES 2)
set(RANGE_FIRST 2)
set(RANGE_LAST 3)

set(kr-mpi-env
    "PYTHONPATH=${rocprofiler-sdk_LIB_DIR}/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}/site-packages"
    OMPI_ALLOW_RUN_AS_ROOT=1
    OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1)

set(kr-mpi-pmc --pmc SQ_WAVES SQ_INSTS_VALU GRBM_COUNT --pmc SQ_WAVES SQ_INSTS_VALU
               GRBM_GUI_ACTIVE)

# All ranks profiled. --kernel-exclude-regex keeps the collective stand-in out of replay
# and --kernel-iteration-range 2-3 limits replay to two launches of the compute kernel;
# every other launch must run once. %rank% gives each rank its own output directory.
rocprofiler_add_integration_execute_test(
    rocprofv3-test-kernel-replay-mpi-generate
    COMMAND
        ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${MPI_RANKS} ${MPIEXEC_PREFLAGS}
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> ${kr-mpi-pmc} --replay-mode kernel
        --kernel-replay-beta-enabled --kernel-exclude-regex peer_visible_kernel
        --kernel-iteration-range "${RANGE_FIRST}-${RANGE_LAST}" --output-format json -d
        ${CMAKE_CURRENT_BINARY_DIR}/all-ranks/rank.%rank% -o out ${PRELOAD_ARGS} --
        $<TARGET_FILE:kernel-replay-mpi> 65536 ${ITERS}
        ${CMAKE_CURRENT_BINARY_DIR}/all-ranks/counts ${MPIEXEC_POSTFLAGS}
    DEPENDS kernel-replay-mpi
    TIMEOUT 180
    LABELS "kernel-replay;mpi-tests"
    ENVIRONMENT "${kr-mpi-env}"
    PRELOAD "${PRELOAD_ENV}"
    FIXTURES_SETUP rocprofv3-test-kernel-replay-mpi-data)

rocprofiler_add_integration_validate_test(
    rocprofv3-test-kernel-replay-mpi-validate BUNDLE_TESTS
    TEST_PATHS validate.py
    COPY conftest.py
    CONFIG pytest.ini
    ARGS --output-dir
         ${CMAKE_CURRENT_BINARY_DIR}/all-ranks
         --mpi-ranks
         ${MPI_RANKS}
         --profiled-ranks
         0
         1
         --iters
         ${ITERS}
         --groups
         ${PASSES}
         --range-first
         ${RANGE_FIRST}
         --range-last
         ${RANGE_LAST}
    TIMEOUT 120
    LABELS "kernel-replay;mpi-tests"
    PYTHON_PATH
        "${rocprofiler-sdk_LIB_DIR}/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}/site-packages"
    FIXTURES_REQUIRED rocprofv3-test-kernel-replay-mpi-data)

# Only rank 0 is profiled. Rank 1 runs the application unprofiled and must still complete
# the MPI collectives with rank 0 while rank 0 replays; rank 1 must not replay anything.
rocprofiler_add_integration_execute_test(
    rocprofv3-test-kernel-replay-mpi-one-rank-generate
    COMMAND
        ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${MPI_RANKS} ${MPIEXEC_PREFLAGS}
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> ${kr-mpi-pmc} --replay-mode kernel
        --kernel-replay-beta-enabled --kernel-exclude-regex peer_visible_kernel
        --kernel-iteration-range "${RANGE_FIRST}-${RANGE_LAST}" --profile-mpi-ranks 0
        --output-format json -d ${CMAKE_CURRENT_BINARY_DIR}/one-rank/rank.%rank% -o out
        ${PRELOAD_ARGS} -- $<TARGET_FILE:kernel-replay-mpi> 65536 ${ITERS}
        ${CMAKE_CURRENT_BINARY_DIR}/one-rank/counts ${MPIEXEC_POSTFLAGS}
    DEPENDS kernel-replay-mpi
    TIMEOUT 180
    LABELS "kernel-replay;mpi-tests"
    ENVIRONMENT "${kr-mpi-env}"
    PRELOAD "${PRELOAD_ENV}"
    FIXTURES_SETUP rocprofv3-test-kernel-replay-mpi-one-rank-data)

rocprofiler_add_integration_validate_test(
    rocprofv3-test-kernel-replay-mpi-one-rank-validate BUNDLE_TESTS
    TEST_PATHS validate.py
    COPY conftest.py
    CONFIG pytest.ini
    ARGS --output-dir
         ${CMAKE_CURRENT_BINARY_DIR}/one-rank
         --mpi-ranks
         ${MPI_RANKS}
         --profiled-ranks
         0
         --iters
         ${ITERS}
         --groups
         ${PASSES}
         --range-first
         ${RANGE_FIRST}
         --range-last
         ${RANGE_LAST}
    TIMEOUT 120
    LABELS "kernel-replay;mpi-tests"
    PYTHON_PATH
        "${rocprofiler-sdk_LIB_DIR}/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}/site-packages"
    FIXTURES_REQUIRED rocprofv3-test-kernel-replay-mpi-one-rank-data)
