#
# Range replay against another thread's work on the same agent. The replay window restores
# the agent's whole tracked inventory before each pass and at its end, so work from another
# thread must wait for the window instead of running inside it, or its writes are restored
# away. The tool times a second thread's HIP graph launches into the window, and the
# application checks that its range and every launch left the values its own run computed.
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

project(rocprofiler-sdk-tests-range-replay-concurrency LANGUAGES CXX HIP)

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

add_library(range-replay-concurrency-client SHARED)
target_sources(range-replay-concurrency-client PRIVATE client.cpp range.hpp)
target_include_directories(range-replay-concurrency-client
                           PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(
    range-replay-concurrency-client PRIVATE rocprofiler-sdk::rocprofiler-sdk
                                            rocprofiler-sdk::tests-build-flags)

# The application resolves the range replay entry points and the tool's rr_coord_get
# handshake with dlsym instead of linking either: linking rocprofiler-sdk into a HIP
# executable makes HIP report hipErrorInvalidDeviceFunction on the first launch on gfx942,
# the same reason the range replay samples do it that way.
set_source_files_properties(main.cpp PROPERTIES LANGUAGE HIP)
add_executable(range-replay-concurrency-testapp)
target_sources(range-replay-concurrency-testapp PRIVATE main.cpp range.hpp)
target_include_directories(range-replay-concurrency-testapp
                           PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(
    range-replay-concurrency-testapp
    PRIVATE rocprofiler-sdk::rocprofiler-sdk-headers rocprofiler-sdk::tests-build-flags
            Threads::Threads ${CMAKE_DL_LIBS})
add_dependencies(range-replay-concurrency-testapp range-replay-concurrency-client)

# C8: a second thread launches graphs on the agent while the first thread's range is being
# replayed. Divergence checking is on, so a launch that ran inside the window also shows
# up as a region the final pass changed. The tool's verdict alone would let a run pass
# while the application reported a wrong buffer, so the application's failure line fails
# the test too.
rocprofiler_add_integration_execute_test(
    test-range-replay-concurrent-graph-launch
    COMMAND $<TARGET_FILE:range-replay-concurrency-testapp>
    DEPENDS range-replay-concurrency-client
    TIMEOUT 180
    LABELS "integration-tests;range-replay"
    PRELOAD "$<TARGET_FILE:range-replay-concurrency-client>"
    ENVIRONMENT "ROCPROF_RANGE_REPLAY_VERIFY=1"
    PASS_REGULAR_EXPRESSION "\\[rr-concurrency\\] PASS"
    FAIL_REGULAR_EXPRESSION
        "\\[rr-concurrency\\] FAIL|\\[rr-concurrency\\] error|\\[app\\] FAIL|${ROCPROFILER_DEFAULT_FAIL_REGEX}"
    SKIP_REGULAR_EXPRESSION "\\[app\\] SKIP")
