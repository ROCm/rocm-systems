#
# Host range replay in an MPI job: every rank replays the same range, in lockstep.
#
# main.cpp is a ring halo-exchange solver (MPI_Sendrecv + MPI_Allreduce every step) run inside one
# range opened with the host-begin protocol on each rank. client.cpp implements peer_sync_cb with
# MPI collectives, so the ranks agree at each pass boundary whether to restore and re-run, or to
# stop. The scenarios cover:
#   - every rank replaying the range for four passes (the collectives must pair up on every pass),
#   - ranks that ask for different pass counts (the job runs the smaller),
#   - one rank declining (a device allocation leaked inside the range): all ranks run it once,
#   - one rank not arming the jump buffer: all ranks run it once.
#
cmake_minimum_required(VERSION 3.21.0 FATAL_ERROR)

if(NOT CMAKE_HIP_COMPILER)
    find_program(
        amdclangpp_EXECUTABLE
        NAMES amdclang++
        HINTS ${ROCM_PATH} ENV ROCM_PATH /opt/rocm
        PATHS ${ROCM_PATH} ENV ROCM_PATH /opt/rocm
        PATH_SUFFIXES bin llvm/bin NO_CACHE)
    mark_as_advanced(amdclangpp_EXECUTABLE)

    if(amdclangpp_EXECUTABLE)
        set(CMAKE_HIP_COMPILER "${amdclangpp_EXECUTABLE}")
    endif()
endif()

project(rocprofiler-sdk-tests-range-replay-host-mpi LANGUAGES CXX HIP)

find_package(MPI COMPONENTS CXX)
if(NOT MPI_CXX_FOUND OR NOT MPIEXEC_EXECUTABLE)
    message(
        STATUS "range-replay-host-mpi: MPI not found, skipping the MPI range replay tests"
        )
    return()
endif()

foreach(_TYPE DEBUG MINSIZEREL RELEASE RELWITHDEBINFO)
    if("${CMAKE_HIP_FLAGS_${_TYPE}}" STREQUAL "")
        set(CMAKE_HIP_FLAGS_${_TYPE} "${CMAKE_CXX_FLAGS_${_TYPE}}")
    endif()
endforeach()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_HIP_STANDARD 17)
set(CMAKE_HIP_EXTENSIONS OFF)
set(CMAKE_HIP_STANDARD_REQUIRED ON)

find_package(rocprofiler-sdk REQUIRED COMPONENTS headers shared-library)
find_package(Threads REQUIRED)

add_library(range-replay-host-mpi-client SHARED)
target_sources(range-replay-host-mpi-client PRIVATE client.cpp common.hpp)
target_include_directories(range-replay-host-mpi-client
                           PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(
    range-replay-host-mpi-client PRIVATE rocprofiler-sdk::rocprofiler-sdk
                                         rocprofiler-sdk::tests-build-flags MPI::MPI_CXX)

# The application resolves the range replay entry points with dlsym instead of linking the
# SDK, like the single-process host replay test.
set_source_files_properties(main.cpp PROPERTIES LANGUAGE HIP)
add_executable(range-replay-host-mpi-testapp)
target_sources(range-replay-host-mpi-testapp PRIVATE main.cpp common.hpp)
target_include_directories(range-replay-host-mpi-testapp
                           PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(
    range-replay-host-mpi-testapp
    PRIVATE rocprofiler-sdk::rocprofiler-sdk-headers rocprofiler-sdk::tests-build-flags
            Threads::Threads ${CMAKE_DL_LIBS} MPI::MPI_CXX)
add_dependencies(range-replay-host-mpi-testapp range-replay-host-mpi-client)

if(ROCPROFILER_MEMCHECK_PRELOAD_ENV_VALUE)
    set(PRELOAD_ENV
        "${ROCPROFILER_MEMCHECK_PRELOAD_ENV_VALUE}:$<TARGET_FILE:range-replay-host-mpi-client>"
        )
else()
    set(PRELOAD_ENV "$<TARGET_FILE:range-replay-host-mpi-client>")
endif()

set(MPI_RANKS 2)

# The tool is preloaded into the ranks only, not into mpiexec itself.
function(add_rr_host_mpi_test NAME)
    cmake_parse_arguments(arg "" "" "ENVIRONMENT" ${ARGN})
    rocprofiler_add_integration_execute_test(
        ${NAME}
        COMMAND
            ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${MPI_RANKS} ${MPIEXEC_PREFLAGS}
            ${CMAKE_COMMAND} -E env "LD_PRELOAD=${PRELOAD_ENV}"
            $<TARGET_FILE:range-replay-host-mpi-testapp> ${MPIEXEC_POSTFLAGS}
        DEPENDS range-replay-host-mpi-client
        TIMEOUT 240
        LABELS "integration-tests;range-replay;mpi-tests"
        ENVIRONMENT
            "ROCPROF_RANGE_REPLAY_VERIFY=1"
            "RR_MPI_OUT=${CMAKE_CURRENT_BINARY_DIR}/${NAME}" OMPI_ALLOW_RUN_AS_ROOT=1
            OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 ${arg_ENVIRONMENT}
        PASS_REGULAR_EXPRESSION "\\[rr-mpi\\] rank 1 PASS"
        FAIL_REGULAR_EXPRESSION
            "\\[rr-mpi\\] FAIL|\\[app\\] FAIL|\\[app\\] rank [0-9]+ .* FAIL|${ROCPROFILER_DEFAULT_FAIL_REGEX}"
        )
endfunction()

# Every rank replays the range: four passes, restores in lockstep.
add_rr_host_mpi_test(test-range-replay-host-mpi-replay ENVIRONMENT
                     "RR_MPI_PASSES=4;RR_MPI_EXPECT_PASSES=4")

# Rank 1 asks for two passes, rank 0 for four: both run two.
add_rr_host_mpi_test(test-range-replay-host-mpi-uneven-passes ENVIRONMENT
                     "RR_MPI_PASSES=4;RR_MPI_PASSES_SPECIAL=2;RR_MPI_EXPECT_PASSES=2")

# Rank 1 leaks a device allocation inside the range (declines ALLOCATION_CHANGED_IN_RANGE
# = 12): rank 0 declines with PEER_DECLINED = 16, and both run the range once.
add_rr_host_mpi_test(
    test-range-replay-host-mpi-peer-declines
    ENVIRONMENT
    "RR_MPI_MODE=leak;RR_MPI_PASSES=4;RR_MPI_EXPECT_PASSES=1;RR_MPI_EXPECT_STATUS=16;RR_MPI_SPECIAL_STATUS=12"
    )

# Rank 1 does not arm the jump buffer (HOST_JMP_NOT_SET = 15): rank 0 declines with
# PEER_DECLINED = 16, and both run the range once.
add_rr_host_mpi_test(
    test-range-replay-host-mpi-peer-unarmed
    ENVIRONMENT
    "RR_MPI_MODE=unarmed;RR_MPI_PASSES=4;RR_MPI_EXPECT_PASSES=1;RR_MPI_EXPECT_STATUS=16;RR_MPI_SPECIAL_STATUS=15"
    )
