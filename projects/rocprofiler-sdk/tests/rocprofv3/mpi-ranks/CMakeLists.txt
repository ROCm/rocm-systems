#
# rocprofv3 MPI ranks filtering test
#
cmake_minimum_required(VERSION 3.21.0 FATAL_ERROR)

project(
    rocprofiler-sdk-tests-rocprofv3-mpi-ranks
    LANGUAGES CXX
    VERSION 0.0.0)

find_package(rocprofiler-sdk REQUIRED)

string(REPLACE "LD_PRELOAD=" "" PRELOAD_ENV "${ROCPROFILER_MEMCHECK_PRELOAD_ENV}")

if(PRELOAD_ENV)
    set(PRELOAD_ARGS --suppress-marker-preload --preload ${PRELOAD_ENV})
else()
    set(PRELOAD_ARGS)
endif()

# Locate an MPI implementation (OpenMPI, MPICH, etc.). MPI is optional: when not found the
# MPI-launched tests below are disabled.
find_package(MPI)

if(NOT MPI_FOUND OR NOT MPIEXEC_EXECUTABLE)
    set(IS_DISABLED ON)
else()
    set(IS_DISABLED OFF)
endif()

# Run-as-root permission for OpenMPI; ignored by other MPI implementations.
set(MPI_ENV_VARS OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1)

# Use rocprofv3's built-in %rank% output key so the rank directory is resolved via the
# profiler's MPI-rank detection logic instead of relying on MPI-implementation-specific
# environment variables.
set(MPI_RANK_DIR_VAR "rank.%rank%")

# Test with MPI - profile only rank 0 out of 4 ranks. Use ${MPI_RANK_DIR_VAR} to create
# separate output directories per rank, independent of the MPI implementation.
rocprofiler_add_integration_execute_test(
    rocprofv3-test-mpi-ranks-with-mpi
    COMMAND
        ${CMAKE_COMMAND} -E env ${MPI_ENV_VARS} ${MPIEXEC_EXECUTABLE}
        ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS}
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> --hip-trace --kernel-trace
        --output-format csv json -d
        ${CMAKE_CURRENT_BINARY_DIR}/mpi-ranks-trace/${MPI_RANK_DIR_VAR} -o out
        --profile-mpi-ranks 0 ${PRELOAD_ARGS} -- $<TARGET_FILE:simple-transpose>
        ${MPIEXEC_POSTFLAGS}
    DEPENDS simple-transpose
    TIMEOUT 120
    LABELS "integration-tests" "mpi-tests"
    PRELOAD "${PRELOAD_ENV}"
    FIXTURES_SETUP rocprofv3-test-mpi-ranks-with-mpi
    DISABLED ${IS_DISABLED})

# Test with MPI - profile ranks 0-1,3 out of 4 ranks. Use ${MPI_RANK_DIR_VAR} to create
# separate output directories per rank, independent of the MPI implementation.
rocprofiler_add_integration_execute_test(
    rocprofv3-test-mpi-ranks-with-mpi-multiple
    COMMAND
        ${CMAKE_COMMAND} -E env ${MPI_ENV_VARS} ${MPIEXEC_EXECUTABLE}
        ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS}
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> --hip-trace --kernel-trace
        --output-format csv json -d
        ${CMAKE_CURRENT_BINARY_DIR}/mpi-ranks-multiple-trace/${MPI_RANK_DIR_VAR} -o out
        --profile-mpi-ranks 0-1,3 ${PRELOAD_ARGS} -- $<TARGET_FILE:simple-transpose>
        ${MPIEXEC_POSTFLAGS}
    DEPENDS simple-transpose
    TIMEOUT 120
    LABELS "integration-tests" "mpi-tests"
    PRELOAD "${PRELOAD_ENV}"
    FIXTURES_SETUP rocprofv3-test-mpi-ranks-with-mpi-multiple
    DISABLED ${IS_DISABLED})

rocprofiler_add_integration_validate_test(
    rocprofv3-test-mpi-ranks-with-mpi
    TEST_PATHS validate.py
    COPY conftest.py
    CONFIG pytest.ini
    ARGS --output-dir ${CMAKE_CURRENT_BINARY_DIR}/mpi-ranks-trace --test-mode
         with-mpi-single
    TIMEOUT 45
    LABELS "integration-tests" "mpi-tests"
    FIXTURES_REQUIRED rocprofv3-test-mpi-ranks-with-mpi
    DISABLED ${IS_DISABLED})

rocprofiler_add_integration_validate_test(
    rocprofv3-test-mpi-ranks-with-mpi-multiple
    TEST_PATHS validate.py
    COPY conftest.py
    CONFIG pytest.ini
    ARGS --output-dir ${CMAKE_CURRENT_BINARY_DIR}/mpi-ranks-multiple-trace --test-mode
         with-mpi-multiple
    TIMEOUT 45
    LABELS "integration-tests" "mpi-tests"
    FIXTURES_REQUIRED rocprofv3-test-mpi-ranks-with-mpi-multiple
    DISABLED ${IS_DISABLED})

# Test without MPI - should work gracefully (no MPI env vars set)
rocprofiler_add_integration_execute_test(
    rocprofv3-test-mpi-ranks-without-mpi
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> --hip-trace --kernel-trace
        --output-format csv json -d ${CMAKE_CURRENT_BINARY_DIR}/no-mpi-trace -o out
        --profile-mpi-ranks 0 ${PRELOAD_ARGS} -- $<TARGET_FILE:simple-transpose>
    DEPENDS simple-transpose
    TIMEOUT 120
    LABELS "integration-tests"
    PRELOAD "${PRELOAD_ENV}"
    FIXTURES_SETUP rocprofv3-test-mpi-ranks-without-mpi)

rocprofiler_add_integration_validate_test(
    rocprofv3-test-mpi-ranks-without-mpi
    TEST_PATHS validate.py
    COPY conftest.py
    CONFIG pytest.ini
    ARGS --output-dir ${CMAKE_CURRENT_BINARY_DIR}/no-mpi-trace --test-mode without-mpi
    TIMEOUT 45
    LABELS "integration-tests"
    FIXTURES_REQUIRED rocprofv3-test-mpi-ranks-without-mpi)
