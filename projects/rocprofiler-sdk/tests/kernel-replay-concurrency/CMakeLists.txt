#
# Kernel-replay P0-1 concurrency regression test (deterministic)
#
# A replay window on an agent snapshots/restores the agent's tracked device memory. If the snapshot
# over-captures HIP's per-stream kernarg pool (coarse-grained segment, executable flag), restore()
# tears the kernel arguments of a concurrent NON-replayed dispatch, so its device write is lost. The
# client tool replays a "hog" kernel, opts a "victim" kernel out of replay, and -- via a kr_coord_t
# handshake -- lets the workload time the victim's write to land exactly inside the window, making
# the corruption 100% deterministic. The fix excludes executable (kernarg-pool) allocations from the
# snapshot.
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

project(rocprofiler-sdk-tests-kernel-replay-concurrency LANGUAGES CXX HIP)

foreach(_TYPE DEBUG MINSIZEREL RELEASE RELWITHDEBINFO)
    if("${CMAKE_HIP_FLAGS_${_TYPE}}" STREQUAL "")
        set(CMAKE_HIP_FLAGS_${_TYPE} "${CMAKE_CXX_FLAGS_${_TYPE}}")
    endif()
endforeach()

find_package(rocprofiler-sdk REQUIRED)
find_package(Threads REQUIRED)

# Client tool (LD_PRELOAD-ed): replays the hog, opts the victim out, forces queue
# interposition, and exposes the kr_coord_t handshake used to time the victim write into
# the replay window.
add_library(kernel-replay-concurrency-client SHARED)
target_sources(kernel-replay-concurrency-client PRIVATE client.cpp)
target_link_libraries(kernel-replay-concurrency-client
                      PRIVATE rocprofiler-sdk::rocprofiler-sdk)

# Workload application (HIP): coordinated hog (replayed) + victim (not replayed). dlsym
# resolves the tool's kr_coord_get handshake, so it links against ${CMAKE_DL_LIBS}.
set_source_files_properties(main.cpp PROPERTIES LANGUAGE HIP)
add_executable(kernel-replay-concurrency-testapp)
target_sources(kernel-replay-concurrency-testapp PRIVATE main.cpp)
target_link_libraries(kernel-replay-concurrency-testapp PRIVATE Threads::Threads
                                                                ${CMAKE_DL_LIBS})

if(ROCPROFILER_MEMCHECK_PRELOAD_ENV_VALUE)
    set(PRELOAD_ENV
        "${ROCPROFILER_MEMCHECK_PRELOAD_ENV_VALUE}:$<TARGET_FILE:kernel-replay-concurrency-client>"
        )
else()
    set(PRELOAD_ENV "$<TARGET_FILE:kernel-replay-concurrency-client>")
endif()

# Args: cycles n hog_reps. Each cycle forces exactly one victim write into one replay
# window, so a regressed build corrupts on cycle 1 (deterministic); 50 cycles keeps a
# safety margin well under the timeout.
rocprofiler_add_integration_execute_test(
    test-kernel-replay-concurrency
    COMMAND $<TARGET_FILE:kernel-replay-concurrency-testapp> 50 65536 1000
    DEPENDS kernel-replay-concurrency-client
    TIMEOUT 120
    LABELS "integration-tests;kernel-replay"
    PRELOAD "${PRELOAD_ENV}"
    PASS_REGULAR_EXPRESSION "\\[repro\\] PASS"
    FAIL_REGULAR_EXPRESSION "\\[repro\\] FAIL|${ROCPROFILER_DEFAULT_FAIL_REGEX}")
