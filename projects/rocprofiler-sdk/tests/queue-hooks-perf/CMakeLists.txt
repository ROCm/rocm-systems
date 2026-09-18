#
# Queue interposition perf tests — dispatch counter path (callback registry vs direct hooks).
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

project(rocprofiler-sdk-tests-queue-hooks-perf LANGUAGES CXX HIP)

foreach(_TYPE DEBUG MINSIZEREL RELEASE RELWITHDEBINFO)
    if("${CMAKE_HIP_FLAGS_${_TYPE}}" STREQUAL "")
        set(CMAKE_HIP_FLAGS_${_TYPE} "${CMAKE_CXX_FLAGS_${_TYPE}}")
    endif()
endforeach()

find_package(rocprofiler-sdk REQUIRED)
find_package(Python3 REQUIRED)

set_source_files_properties(main.cpp PROPERTIES LANGUAGE HIP)
add_executable(queue-hooks-perf-testapp)
target_sources(queue-hooks-perf-testapp PRIVATE main.cpp)
target_link_libraries(queue-hooks-perf-testapp PRIVATE Threads::Threads)

add_test(
    NAME test-queue-hooks-perf-scaling
    COMMAND
        ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/run_and_validate.py --testapp
        $<TARGET_FILE:queue-hooks-perf-testapp> --preload
        $<TARGET_FILE:rocprofiler-sdk-json-tool> --ballast-mb 8 --launches-low 8
        --launches-high 32 --max-scaling-ratio 6.0)

set_tests_properties(
    test-queue-hooks-perf-scaling
    PROPERTIES TIMEOUT
               180
               LABELS
               "integration-tests;queue-hooks;queue-hooks-perf"
               RESOURCE_LOCK
               GPU
               DEPENDS
               "queue-hooks-perf-testapp;rocprofiler-sdk-json-tool"
               ENVIRONMENT
               "QH_PERF_RESULTS_JSON=${CMAKE_CURRENT_BINARY_DIR}/qh_perf_scaling.json"
               PASS_REGULAR_EXPRESSION
               "\\[qh-perf-run\\] PASS"
               FAIL_REGULAR_EXPRESSION
               "${ROCPROFILER_DEFAULT_FAIL_REGEX}")

add_test(
    NAME test-queue-hooks-perf-launch-sweep
    COMMAND
        ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/run_launch_sweep.py --testapp
        $<TARGET_FILE:queue-hooks-perf-testapp> --preload
        $<TARGET_FILE:rocprofiler-sdk-json-tool> --ballast-mb 8 --launches 4 8 16 32
        --slack 2.5)

set_tests_properties(
    test-queue-hooks-perf-launch-sweep
    PROPERTIES TIMEOUT
               240
               LABELS
               "integration-tests;queue-hooks;queue-hooks-perf"
               RESOURCE_LOCK
               GPU
               DEPENDS
               "queue-hooks-perf-testapp;rocprofiler-sdk-json-tool"
               ENVIRONMENT
               "QH_PERF_SWEEP_JSON=${CMAKE_CURRENT_BINARY_DIR}/qh_perf_launch_sweep.json"
               PASS_REGULAR_EXPRESSION
               "\\[qh-perf-sweep\\] PASS overall"
               FAIL_REGULAR_EXPRESSION
               "${ROCPROFILER_DEFAULT_FAIL_REGEX}")
