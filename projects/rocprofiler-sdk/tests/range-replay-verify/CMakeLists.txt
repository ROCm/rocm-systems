#
# Range replay with verification on, for kernels whose results only a correct replay can
# reproduce. ROCPROF_RANGE_REPLAY_VERIFY makes CLOSE report how many snapshot regions the
# final replayed pass left different from the application's own run of the range. Each test
# checks that count, that the range really was replayed, and, in the application, that its
# own result is intact element for element.
#
# A test is one RR_VERIFY_MODE of the application paired with the dispatch count the tool
# expects.
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

project(rocprofiler-sdk-tests-range-replay-verify LANGUAGES CXX HIP)

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

add_library(range-replay-verify-client SHARED)
target_sources(range-replay-verify-client PRIVATE client.cpp range.hpp)
target_include_directories(range-replay-verify-client PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(
    range-replay-verify-client PRIVATE rocprofiler-sdk::rocprofiler-sdk
                                       rocprofiler-sdk::tests-build-flags)

# The application resolves the range replay entry points with dlsym instead of linking the
# SDK: linking rocprofiler-sdk into a HIP executable makes HIP report
# hipErrorInvalidDeviceFunction on the first launch on gfx942, the same reason the range
# replay samples do it that way.
set_source_files_properties(main.cpp PROPERTIES LANGUAGE HIP)
add_executable(range-replay-verify-testapp)
target_sources(range-replay-verify-testapp PRIVATE main.cpp range.hpp)
target_include_directories(range-replay-verify-testapp
                           PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(
    range-replay-verify-testapp
    PRIVATE rocprofiler-sdk::rocprofiler-sdk-headers rocprofiler-sdk::tests-build-flags
            Threads::Threads ${CMAKE_DL_LIBS})
add_dependencies(range-replay-verify-testapp range-replay-verify-client)

# The tool's verdict alone would let a run pass while the application reported a wrong
# result, so the application's own failure line fails the test too.
function(add_rr_verify_test NAME)
    cmake_parse_arguments(arg "" "MODE;DISPATCHES" "" ${ARGN})
    rocprofiler_add_integration_execute_test(
        ${NAME}
        COMMAND $<TARGET_FILE:range-replay-verify-testapp>
        DEPENDS range-replay-verify-client
        TIMEOUT 120
        LABELS "integration-tests;range-replay"
        PRELOAD "$<TARGET_FILE:range-replay-verify-client>"
        ENVIRONMENT "ROCPROF_RANGE_REPLAY_VERIFY=1" "RR_VERIFY_MODE=${arg_MODE}"
                    "RR_VERIFY_EXPECT_DISPATCHES=${arg_DISPATCHES}"
        PASS_REGULAR_EXPRESSION "\\[rr-verify\\] PASS"
        FAIL_REGULAR_EXPRESSION
            "\\[rr-verify\\] FAIL|\\[rr-verify\\] error|\\[app\\] FAIL|${ROCPROFILER_DEFAULT_FAIL_REGEX}"
        SKIP_REGULAR_EXPRESSION "\\[app\\] SKIP")
endfunction()

# C9: three in-place steps over one 16 MiB buffer. Each pass must start from the restored
# buffer: a step applied to its own output gives a different value, and the steps do not
# commute, so a pass that started anywhere else, or reordered them, diverges.
add_rr_verify_test(test-range-replay-verify-in-place MODE in-place DISPATCHES 3)

# C9: a histogram built with contended global atomicAdd, then folded back into the buffer
# in place. The bins must be restored to zero before every pass or the counts accumulate,
# and integer atomics are order-independent, so verification must report no divergence.
add_rr_verify_test(test-range-replay-verify-atomic-histogram MODE atomic-histogram
                   DISPATCHES 2)
