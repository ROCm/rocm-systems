# MIT License
#
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

# -------------------------------------------------------------------------------------- #
#
# UCX tests - MPI examples with UCX transport
#
# -------------------------------------------------------------------------------------- #

# UCX tests require MPI examples since UCX is MPI's transport layer
if(NOT ROCPROFSYS_USE_MPI AND NOT ROCPROFSYS_USE_MPI_HEADERS)
    return()
endif()

# Detect MPI implementation by checking include paths
set(_DETECTED_MPI_IMPL "unknown")
if("${MPI_C_COMPILER_INCLUDE_DIRS};${MPI_C_HEADER_DIR}" MATCHES "openmpi")
    set(_DETECTED_MPI_IMPL "openmpi")
elseif("${MPI_C_COMPILER_INCLUDE_DIRS};${MPI_C_HEADER_DIR}" MATCHES "mpich")
    set(_DETECTED_MPI_IMPL "mpich")
endif()

# Only proceed if OpenMPI is detected
if(NOT "${_DETECTED_MPI_IMPL}" STREQUAL "openmpi")
    message(
        WARNING
        "Skipping UCX tests - requires OpenMPI (detected: ${_DETECTED_MPI_IMPL}). UCX tests use OpenMPI-specific environment variables (OMPI_MCA_*)."
    )
    return()
endif()

# Force OpenMPI to use UCX transport via environment variables
set(_ucxp_mpi_environment
    "OMPI_MCA_pml=ucx" # Use UCX point-to-point messaging layer
    "OMPI_MCA_osc=ucx" # Use UCX one-sided communications
    "OMPI_MCA_pml_ucx_tls=tcp,self" # Force TCP and self (not sysv/posix/cma which bypass UCX functions)
    "OMPI_MCA_pml_ucx_devices=any" # Accept any device (not just InfiniBand/Mellanox)
    "OMPI_MCA_btl=^vader,sm" # Disable shared memory BTLs to force communication through UCX
    "UCX_TLS=tcp,self" # Tell UCX to use TCP for inter-process, self for intra-process
    "OMPI_MCA_pml_base_verbose=100" # Show which PML is selected
    "UCX_LOG_LEVEL=info" # Enable UCX logging to show transport usage
    "OMPI_ALLOW_RUN_AS_ROOT=1" # Allow running as root in CI environments
    "OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1" # Confirm the choice to run as root
)

# Enhanced UCX environment with more detailed logging
set(_ucx_environment
    "${_base_environment}"
    "ROCPROFSYS_USE_UCX=ON"
    "ROCPROFSYS_VERBOSE=3"
    "ROCPROFSYS_DL_VERBOSE=3"
    "ROCPROFSYS_PERFETTO_BACKEND=inprocess"
    "ROCPROFSYS_PERFETTO_FILL_POLICY=ring_buffer"
    "ROCPROFSYS_USE_PID=OFF"
    "${_ucxp_mpi_environment}"
)

# Enable ROCPD for UCX tests if valid ROCm is installed and a valid GPU is detected
if(${ENABLE_ROCPD_TEST} AND ${_VALID_GPU})
    list(APPEND _ucx_environment "ROCPROFSYS_USE_ROCPD=ON")
endif()

set(_UCX_PASS_REGEX "ucx_gotcha|category::ucx")

# Helper function to add fixture dependencies to UCX tests
function(add_ucx_fixture_dependency TEST_BASE_NAME)
    foreach(_test_suffix sampling binary-rewrite binary-rewrite-run sys-run)
        if(TEST ${TEST_BASE_NAME}-${_test_suffix})
            set_property(
                TEST ${TEST_BASE_NAME}-${_test_suffix}
                APPEND
                PROPERTY FIXTURES_REQUIRED ucx_available
            )
        endif()
    endforeach()
endfunction()

# Add a runtime validation test that checks if UCX is functional
# This test runs before all other UCX tests and acts as a fixture
add_test(
    NAME ucx-validation-check
    COMMAND
        ${CMAKE_COMMAND} -E env ${_ucxp_mpi_environment} ${MPIEXEC_EXECUTABLE}
        ${MPIEXEC_EXECUTABLE_ARGS} ${MPIEXEC_NUMPROC_FLAG} 2 $<TARGET_FILE:mpi-send-recv>
)

# Set this test as a fixture that must pass for UCX tests to run
set_tests_properties(
    ucx-validation-check
    PROPERTIES
        LABELS "ucx;validation"
        FIXTURES_SETUP ucx_available
        FAIL_REGULAR_EXPRESSION
            "PML ucx cannot be selected|UCX is not available|No UCX support found|Failed to select"
)

# UCX trace test
rocprofiler_systems_add_test(
    SKIP_BASELINE SKIP_RUNTIME SKIP_SAMPLING
    NAME "ucx-send-recv"
    TARGET mpi-send-recv
    MPI ON
    NUM_PROCS 2
    LABELS "ucx;send-recv"
    REWRITE_ARGS
        -e
        -v
        2
        --label
        file
        line
        --min-instructions
        0
    ENVIRONMENT "${_ucx_environment};ROCPROFSYS_TRACE_LEGACY=ON;ROCPROFSYS_PERFETTO_COMBINE_TRACES=ON"
    REWRITE_RUN_PASS_REGEX
        "${_UCX_PASS_REGEX}|Successfully executed: .+rocprof-sys-merge-output.sh.*"
    REWRITE_RUN_FAIL_REGEX
        "Script not found|Failed to execute|ROCPROFSYS_ABORT_FAIL_REGEX"
    SYS_RUN_PASS_REGEX
        "${_UCX_PASS_REGEX}|Using UCX|pml.*ucx"
)

# Add fixture dependency
add_ucx_fixture_dependency(ucx-send-recv)

# Validation test for UCX perfetto trace to ensure communication tracks are present
rocprofiler_systems_add_validation_test(
    NAME ucx-send-recv-sys-run
    PERFETTO_METRIC "ucx"
    PERFETTO_FILE "merged.proto"
    LABELS "ucx;perfetto"
    ARGS --counter-names "UCX Comm Recv" "UCX Comm Send" -p
)

# Validation test for UCX rocpd output
if(${ENABLE_ROCPD_TEST} AND ${_VALID_GPU} AND TEST ucx-send-recv-sys-run)
    set_property(TEST ucx-send-recv-sys-run APPEND PROPERTY LABELS rocpd)

    # For MPI tests, ROCPD creates separate DB files for each rank with PID suffix (rocpd-<pid>.db)
    # Create a setup test that finds and symlinks one of them to a predictable name
    set(UCX_ROCPD_OUTPUT_DIR
        "${PROJECT_BINARY_DIR}/rocprof-sys-tests-output/ucx-send-recv-sys-run"
    )

    add_test(
        NAME ucx-send-recv-sys-run-rocpd-setup
        COMMAND
            ${CMAKE_COMMAND} -E env bash -c
            "ROCPD_DB=$(ls ${UCX_ROCPD_OUTPUT_DIR}/rocpd-*.db 2>/dev/null | head -1) && ln -sf $(basename \"$ROCPD_DB\") ${UCX_ROCPD_OUTPUT_DIR}/rocpd.db"
        WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
    )

    set_tests_properties(
        ucx-send-recv-sys-run-rocpd-setup
        PROPERTIES
            LABELS "ucx;rocpd;setup"
            DEPENDS ucx-send-recv-sys-run
            FIXTURES_REQUIRED ucx_available
    )

    # Standard validation test - now it can use the predictable symlink name
    rocprofiler_systems_add_validation_test(
        NAME ucx-send-recv-sys-run
        ROCPD_FILE "rocpd.db"
        LABELS "ucx;rocpd"
        ARGS --validation-rules
            "${CMAKE_CURRENT_LIST_DIR}/rocpd-validation-rules/ucx/validation-rules.json"
    )

    # Make validation depend on the setup test and UCX fixture
    if(TEST validate-ucx-send-recv-sys-run-rocpd)
        set_property(
            TEST validate-ucx-send-recv-sys-run-rocpd
            APPEND
            PROPERTY DEPENDS ucx-send-recv-sys-run-rocpd-setup
        )
        set_property(
            TEST validate-ucx-send-recv-sys-run-rocpd
            APPEND
            PROPERTY FIXTURES_REQUIRED ucx_available
        )
    endif()
endif()

# UCX with MPIP integration test
rocprofiler_systems_add_test(
    SKIP_BASELINE SKIP_RUNTIME
    NAME "ucx-mpip-integration"
    TARGET mpi-all2all
    MPI ON
    NUM_PROCS 2
    LABELS "ucx;mpip"
    REWRITE_ARGS
        -e
        -v
        2
        --label
        file
        line
        args
        --min-instructions
        0
    ENVIRONMENT
        "${_ucx_environment};ROCPROFSYS_USE_MPIP=ON"
    RUN_ARGS 30
    REWRITE_RUN_PASS_REGEX "${_UCX_PASS_REGEX}"
    SAMPLING_PASS_REGEX "${_UCX_PASS_REGEX}"
)

# Add fixture dependency
add_ucx_fixture_dependency(ucx-mpip-integration)

# UCX with different message sizes
foreach(_MSG_SIZE 1024 4096 16384)
    rocprofiler_systems_add_test(
        SKIP_BASELINE SKIP_RUNTIME
        NAME "ucx-bcast-${_MSG_SIZE}"
        TARGET mpi-bcast
        MPI ON
        NUM_PROCS 2
        LABELS "ucx;bcast"
        REWRITE_ARGS
            -e
            -v
            2
            --label
            file
            line
            --min-instructions
            0
        ENVIRONMENT "${_ucx_environment}"
        RUN_ARGS ${_MSG_SIZE}
        REWRITE_RUN_PASS_REGEX "${_UCX_PASS_REGEX}"
        SAMPLING_PASS_REGEX "${_UCX_PASS_REGEX}"
    )

    # Add fixture dependency
    add_ucx_fixture_dependency(ucx-bcast-${_MSG_SIZE})
endforeach()

# Test UCX active message functionality
rocprofiler_systems_add_test(
    SKIP_BASELINE SKIP_RUNTIME SKIP_SAMPLING
    NAME "ucx-active-messages"
    TARGET mpi-allreduce
    MPI ON
    NUM_PROCS 2
    LABELS "ucx;am"
    REWRITE_ARGS
        -e
        -v
        2
        --label
        file
        line
        --min-instructions
        0
    ENVIRONMENT "${_ucx_environment};OMPI_MCA_btl=^vader,tcp,openib,uct"
    RUN_ARGS 64
    REWRITE_RUN_PASS_REGEX "${_UCX_PASS_REGEX}"
)

# Add fixture dependency
add_ucx_fixture_dependency(ucx-active-messages)
