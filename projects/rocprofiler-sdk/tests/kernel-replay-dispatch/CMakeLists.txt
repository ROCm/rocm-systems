#
# Kernel-replay dispatch replacement integration tests.
#
# A tool replays one HIP dispatch three times and, on pass 1, submits its own copy of a
# different kernel through replay_set_dispatch, with its own kernel argument buffer and twice
# the grid. It checks the substitute's effect before each restore, that the application
# keeps the output of the final (original) pass, and optionally that thread trace reports the
# substitute's kernel id on the replaced pass.
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

project(rocprofiler-sdk-tests-kernel-replay-dispatch LANGUAGES CXX HIP)

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

find_package(rocprofiler-sdk REQUIRED)
find_package(
    hsa-runtime64
    REQUIRED
    CONFIG
    HINTS
    ${rocm_version_DIR}
    ${ROCM_PATH}
    PATHS
    ${rocm_version_DIR}
    ${ROCM_PATH})

add_library(kernel-replay-dispatch-client SHARED)
target_sources(kernel-replay-dispatch-client PRIVATE client.cpp)
target_link_libraries(
    kernel-replay-dispatch-client
    PRIVATE rocprofiler-sdk::rocprofiler-sdk rocprofiler-sdk::tests-build-flags
            hsa-runtime64::hsa-runtime64)

set_source_files_properties(main.cpp PROPERTIES LANGUAGE HIP)
add_executable(kernel-replay-dispatch-testapp)
target_sources(kernel-replay-dispatch-testapp PRIVATE main.cpp)
target_link_libraries(kernel-replay-dispatch-testapp
                      PRIVATE rocprofiler-sdk::tests-build-flags)

if(ROCPROFILER_MEMCHECK_PRELOAD_ENV_VALUE)
    set(PRELOAD_ENV
        "${ROCPROFILER_MEMCHECK_PRELOAD_ENV_VALUE}:$<TARGET_FILE:kernel-replay-dispatch-client>"
        )
else()
    set(PRELOAD_ENV "$<TARGET_FILE:kernel-replay-dispatch-client>")
endif()

function(add_kr_dispatch_test NAME)
    cmake_parse_arguments(arg "" "DISABLED" "ENVIRONMENT" ${ARGN})
    set(_disabled)
    if(arg_DISABLED)
        set(_disabled DISABLED ${arg_DISABLED})
    endif()
    rocprofiler_add_integration_execute_test(
        ${NAME}
        COMMAND $<TARGET_FILE:kernel-replay-dispatch-testapp>
        DEPENDS kernel-replay-dispatch-client
        TIMEOUT 120
        LABELS "integration-tests;kernel-replay"
        PRELOAD "${PRELOAD_ENV}"
        ENVIRONMENT ${arg_ENVIRONMENT} ${_disabled}
        PASS_REGULAR_EXPRESSION "\\[kd\\] PASS"
        FAIL_REGULAR_EXPRESSION
            "\\[kd\\] FAIL|\\[app\\] FAIL|${ROCPROFILER_DEFAULT_FAIL_REGEX}"
        SKIP_REGULAR_EXPRESSION
            "ATT unavailable|substitute code object unavailable|cluster dispatch unavailable"
        )
endfunction()

# Pass 1 submits the substitute with its own kernel arguments and a doubled grid.
add_kr_dispatch_test(test-kernel-replay-dispatch-replace)

# Same, with thread trace on every pass: pass 1 must report the substitute's kernel id.
add_kr_dispatch_test(test-kernel-replay-dispatch-replace-att ENVIRONMENT
                     "KR_DISPATCH_ATT=1")

# gfx1250 cluster dispatch: the replacement stays a cluster dispatch with recomputed
# cluster counts.
set(KR_DISPATCH_CLUSTER_DISABLED ON)
foreach(_gfx IN LISTS rocprofiler-sdk-tests-gfx-info)
    if(_gfx MATCHES "^gfx125[0-9]")
        set(KR_DISPATCH_CLUSTER_DISABLED OFF)
        break()
    endif()
endforeach()
add_kr_dispatch_test(test-kernel-replay-dispatch-replace-cluster ENVIRONMENT
                     "KR_DISPATCH_CLUSTER=1" DISABLED ${KR_DISPATCH_CLUSTER_DISABLED})
