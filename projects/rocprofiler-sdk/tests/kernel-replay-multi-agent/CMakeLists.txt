#
# Kernel replay on two GPUs at once. One host thread per GPU launches an accumulate kernel that the
# client tool replays, and the test checks both the result and the replay counts on each GPU. The
# spill variant leaves little free device memory, so each snapshot is cut and partly kept in pinned
# host memory, which GPUs on the same CPU share. Fewer than two GPUs skips the test, and the spill
# variant also skips on integrated GPUs, whose device memory is the host's.
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

project(rocprofiler-sdk-tests-kernel-replay-multi-agent LANGUAGES CXX HIP)

foreach(_TYPE DEBUG MINSIZEREL RELEASE RELWITHDEBINFO)
    if("${CMAKE_HIP_FLAGS_${_TYPE}}" STREQUAL "")
        set(CMAKE_HIP_FLAGS_${_TYPE} "${CMAKE_CXX_FLAGS_${_TYPE}}")
    endif()
endforeach()

find_package(rocprofiler-sdk REQUIRED)
find_package(Threads REQUIRED)

add_library(kernel-replay-multi-agent-client SHARED)
target_sources(kernel-replay-multi-agent-client PRIVATE client.cpp)
target_link_libraries(kernel-replay-multi-agent-client
                      PRIVATE rocprofiler-sdk::rocprofiler-sdk)

set_source_files_properties(main.cpp PROPERTIES LANGUAGE HIP)
add_executable(kernel-replay-multi-agent-testapp)
target_sources(kernel-replay-multi-agent-testapp PRIVATE main.cpp)
target_link_libraries(kernel-replay-multi-agent-testapp PRIVATE Threads::Threads)

set(KR_MULTI_AGENT_LAUNCHES 4)

# Args: launches, MiB per buffer, MiB to leave free on each GPU (0 = no ballast).
function(add_multi_agent_test NAME)
    cmake_parse_arguments(arg "" "BUFFER_MIB;LEAVE_FREE_MIB;TIMEOUT" "ENVIRONMENT"
                          ${ARGN})
    rocprofiler_add_integration_execute_test(
        ${NAME}
        COMMAND $<TARGET_FILE:kernel-replay-multi-agent-testapp>
                ${KR_MULTI_AGENT_LAUNCHES} ${arg_BUFFER_MIB} ${arg_LEAVE_FREE_MIB}
        DEPENDS kernel-replay-multi-agent-client
        TIMEOUT ${arg_TIMEOUT}
        LABELS "integration-tests;kernel-replay"
        PRELOAD "$<TARGET_FILE:kernel-replay-multi-agent-client>"
        ENVIRONMENT "KR_LAUNCHES=${KR_MULTI_AGENT_LAUNCHES}" ${arg_ENVIRONMENT}
        SKIP_REGULAR_EXPRESSION
            "fewer than two GPUs|needs discrete GPUs|device-only snapshot unavailable")
endfunction()

add_multi_agent_test(test-kernel-replay-multi-agent BUFFER_MIB 256 LEAVE_FREE_MIB 0
                     TIMEOUT 120)
add_multi_agent_test(
    test-kernel-replay-multi-agent-host-only
    BUFFER_MIB
    256
    LEAVE_FREE_MIB
    0
    TIMEOUT
    120
    ENVIRONMENT
    "KR_SNAPSHOT=host")
add_multi_agent_test(
    test-kernel-replay-multi-agent-device-only
    BUFFER_MIB
    256
    LEAVE_FREE_MIB
    0
    TIMEOUT
    120
    ENVIRONMENT
    "KR_SNAPSHOT=device")

# Fills both GPUs with fine-grained memory, so nothing else may run on them meanwhile.
add_multi_agent_test(test-kernel-replay-multi-agent-spill BUFFER_MIB 2048 LEAVE_FREE_MIB
                     1024 TIMEOUT 300)
rocprofiler_set_integration_test_name(_spill_test test-kernel-replay-multi-agent-spill
                                      "execute")
set_tests_properties(${_spill_test} PROPERTIES RUN_SERIAL ON)
