#
#
#
cmake_minimum_required(VERSION 3.21.0 FATAL_ERROR)

project(
    rocprofiler-sdk-tests-openmp-init-deadlock
    LANGUAGES CXX
    VERSION 0.0.0)

find_package(rocprofiler-sdk REQUIRED)

# the application is only built when LLVM libomp is found (see
# tests/bin/openmp-init-deadlock); skip registering the test if the target was not created
if(NOT TARGET openmp-init-deadlock)
    message(
        STATUS
            "[openmp-init-deadlock] application target not available; skipping regression test"
        )
    return()
endif()

# Regression test for the OMPT-init deadlock fixed in rocprofiler-sdk client discovery.
#
# rocprofiler-sdk is loaded as the OMPT tool (OMP_TOOL_LIBRARIES). When the linked
# blas-stub library's DT_INIT enters an omp parallel region, libomp performs OMPT tool
# discovery while holding its init lock. Previously, client discovery dlopen'd the
# configure-exporting client library (whose DT_INIT was still pending), re-entering libomp
# and self-deadlocking. With the fix, discovery resolves symbols via the ELF symbol table
# without dlopen, so the application runs to completion.
#
# The assertion is twofold: the TIMEOUT guards against the deadlock (a hang fails the
# test), and PASS_REGULAR_EXPRESSION ensures the application actually reached main (so an
# unrelated early exit cannot pass silently).
rocprofiler_add_integration_execute_test(
    openmp-init-deadlock
    TARGET openmp-init-deadlock
    TIMEOUT 60
    LABELS "integration-tests;openmp-init-deadlock"
    ENVIRONMENT
        "OMP_TOOL=enabled"
        "OMP_NUM_THREADS=2"
        "OMP_TOOL_LIBRARIES=$<TARGET_FILE:rocprofiler-sdk::rocprofiler-sdk-shared-library>"
        "LD_LIBRARY_PATH=$<TARGET_FILE_DIR:rocprofiler-sdk::rocprofiler-sdk-shared-library>:$ENV{LD_LIBRARY_PATH}"
    PASS_REGULAR_EXPRESSION "reached main: no OMPT-init deadlock")
