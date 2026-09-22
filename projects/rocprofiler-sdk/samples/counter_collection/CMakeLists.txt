#
#
#
cmake_minimum_required(VERSION 3.24 FATAL_ERROR)

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

project(rocprofiler-sdk-samples-counter-collection LANGUAGES CXX HIP)

foreach(_TYPE DEBUG MINSIZEREL RELEASE RELWITHDEBINFO)
    if("${CMAKE_HIP_FLAGS_${_TYPE}}" STREQUAL "")
        set(CMAKE_HIP_FLAGS_${_TYPE} "${CMAKE_CXX_FLAGS_${_TYPE}}")
    endif()
endforeach()

if(ROCPROFILER_MEMCHECK STREQUAL "ThreadSanitizer")
    set(IS_THREAD_SANITIZER ON)
else()
    set(IS_THREAD_SANITIZER OFF)
endif()

find_package(rocprofiler-sdk REQUIRED)
if(BUILD_TESTING)
    find_package(Python3 QUIET COMPONENTS Interpreter)
    if(Python3_Interpreter_FOUND)
        find_package(rocprofiler_sdk_pytest QUIET)
    endif()
endif()

add_library(counter-collection-buffer-client SHARED)
target_sources(counter-collection-buffer-client PRIVATE buffered_client.cpp client.hpp)
target_link_libraries(
    counter-collection-buffer-client
    PUBLIC rocprofiler-sdk::samples-build-flags
    PRIVATE rocprofiler-sdk::rocprofiler-sdk rocprofiler-sdk::samples-common-library)

set_source_files_properties(main.cpp PROPERTIES LANGUAGE HIP)
add_executable(counter-collection-buffer)
target_sources(counter-collection-buffer PRIVATE main.cpp)
target_link_libraries(counter-collection-buffer PRIVATE counter-collection-buffer-client
                                                        Threads::Threads)

rocprofiler_samples_get_ld_library_path_env(LIBRARY_PATH_ENV)
rocprofiler_samples_get_preload_env(PRELOAD_ENV counter-collection-buffer-client)

set(counter-collection-buffer-env "${PRELOAD_ENV}" "${LIBRARY_PATH_ENV}")

add_test(NAME counter-collection-buffer COMMAND $<TARGET_FILE:counter-collection-buffer>)

set_tests_properties(
    counter-collection-buffer
    PROPERTIES TIMEOUT
               120
               LABELS
               "samples"
               ENVIRONMENT
               "${counter-collection-buffer-env}"
               FAIL_REGULAR_EXPRESSION
               "${ROCPROFILER_DEFAULT_FAIL_REGEX}"
               DISABLED
               "${ROCPROFILER_DISABLE_UNSTABLE_CTESTS}")

set_source_files_properties(device_serialized_main.cpp PROPERTIES LANGUAGE HIP)
add_executable(counter-collection-buffer-device-serialization)
target_sources(counter-collection-buffer-device-serialization
               PRIVATE device_serialized_main.cpp)
target_link_libraries(counter-collection-buffer-device-serialization
                      PRIVATE counter-collection-buffer-client Threads::Threads)

rocprofiler_samples_get_ld_library_path_env(LIBRARY_PATH_ENV)
rocprofiler_samples_get_preload_env(PRELOAD_ENV counter-collection-buffer-client)

set(counter-collection-buffer-device-serialization-env "${PRELOAD_ENV}"
                                                       "${LIBRARY_PATH_ENV}")

add_test(NAME counter-collection-buffer-device-serialization
         COMMAND $<TARGET_FILE:counter-collection-buffer-device-serialization>)

set_tests_properties(
    counter-collection-buffer-device-serialization
    PROPERTIES TIMEOUT 120 LABELS "samples" ENVIRONMENT
               "${counter-collection-buffer-device-serialization-env}"
               FAIL_REGULAR_EXPRESSION "${ROCPROFILER_DEFAULT_FAIL_REGEX}")

set(counter-collection-buffer-finalize-at-exit-env
    ${counter-collection-buffer-device-serialization-env}
    "ROCPROFILER_QUEUE_INTERPOSITION=0" "ROCPROFILER_LOG_LEVEL=INFO")

add_test(NAME counter-collection-buffer-finalize-at-exit
         COMMAND $<TARGET_FILE:counter-collection-buffer-device-serialization>)

set_tests_properties(
    counter-collection-buffer-finalize-at-exit
    PROPERTIES TIMEOUT 120 LABELS "samples;repro" ENVIRONMENT
               "${counter-collection-buffer-finalize-at-exit-env}"
               FAIL_REGULAR_EXPRESSION "${ROCPROFILER_DEFAULT_FAIL_REGEX}")

set(counter-collection-buffer-finalize-at-exit-forced-inline-env
    ${counter-collection-buffer-device-serialization-env}
    "ROCPROFILER_QUEUE_INTERPOSITION=1" "ROCPROFILER_LOG_LEVEL=INFO")

add_test(NAME counter-collection-buffer-finalize-at-exit-forced-inline
         COMMAND $<TARGET_FILE:counter-collection-buffer-device-serialization>)

set_tests_properties(
    counter-collection-buffer-finalize-at-exit-forced-inline
    PROPERTIES TIMEOUT 120 LABELS "samples;repro" ENVIRONMENT
               "${counter-collection-buffer-finalize-at-exit-forced-inline-env}"
               FAIL_REGULAR_EXPRESSION "${ROCPROFILER_DEFAULT_FAIL_REGEX}")

add_library(counter-collection-callback-client SHARED)
target_sources(counter-collection-callback-client PRIVATE callback_client.cpp client.hpp)
target_link_libraries(
    counter-collection-callback-client
    PUBLIC rocprofiler-sdk::samples-build-flags
    PRIVATE rocprofiler-sdk::rocprofiler-sdk rocprofiler-sdk::samples-common-library)

set_source_files_properties(main.cpp PROPERTIES LANGUAGE HIP)
add_executable(counter-collection-callback)
target_sources(counter-collection-callback PRIVATE main.cpp)
target_link_libraries(counter-collection-callback
                      PRIVATE counter-collection-callback-client Threads::Threads)

rocprofiler_samples_get_preload_env(PRELOAD_ENV counter-collection-callback-client)

set(counter-collection-callback-env "${PRELOAD_ENV}" "${LIBRARY_PATH_ENV}")

add_test(NAME counter-collection-callback
         COMMAND $<TARGET_FILE:counter-collection-callback>)

set_tests_properties(
    counter-collection-callback
    PROPERTIES TIMEOUT
               120
               LABELS
               "samples"
               ENVIRONMENT
               "${counter-collection-callback-env}"
               FAIL_REGULAR_EXPRESSION
               "${ROCPROFILER_DEFAULT_FAIL_REGEX}"
               DISABLED
               "${ROCPROFILER_DISABLE_UNSTABLE_CTESTS}")

add_library(counter-collection-functional-counter-client SHARED)
target_sources(counter-collection-functional-counter-client
               PRIVATE print_functional_counters_client.cpp client.hpp)
target_link_libraries(
    counter-collection-functional-counter-client
    PUBLIC rocprofiler-sdk::samples-build-flags
    PRIVATE rocprofiler-sdk::rocprofiler-sdk rocprofiler-sdk::samples-common-library)

add_executable(counter-collection-print-functional-counters)
target_sources(counter-collection-print-functional-counters PRIVATE main.cpp)
target_link_libraries(
    counter-collection-print-functional-counters
    PRIVATE counter-collection-functional-counter-client Threads::Threads)

rocprofiler_samples_get_preload_env(PRELOAD_ENV
                                    counter-collection-functional-counter-client)

set(counter-collection-functional-counter-env "${PRELOAD_ENV}" "${LIBRARY_PATH_ENV}")

add_test(NAME counter-collection-print-functional-counters
         COMMAND $<TARGET_FILE:counter-collection-print-functional-counters>)

set_tests_properties(
    counter-collection-print-functional-counters
    PROPERTIES TIMEOUT 120 LABELS "samples" ENVIRONMENT
               "${counter-collection-functional-counter-env}" FAIL_REGULAR_EXPRESSION
               "${ROCPROFILER_DEFAULT_FAIL_REGEX}")

add_library(counter-collection-device-profiling-client SHARED)
target_sources(counter-collection-device-profiling-client
               PRIVATE device_counting_async_client.cpp client.hpp)
target_link_libraries(
    counter-collection-device-profiling-client
    PUBLIC rocprofiler-sdk::samples-build-flags
    PRIVATE rocprofiler-sdk::rocprofiler-sdk rocprofiler-sdk::samples-common-library)

add_executable(counter-collection-device-profiling)
target_sources(counter-collection-device-profiling PRIVATE main.cpp)
target_link_libraries(counter-collection-device-profiling
                      PRIVATE counter-collection-device-profiling-client Threads::Threads)

rocprofiler_samples_get_preload_env(PRELOAD_ENV
                                    counter-collection-device-profiling-client)

set(counter-collection-device-profiling-output
    "${CMAKE_CURRENT_BINARY_DIR}/counter-collection-device-profiling-async.log")
set(counter-collection-device-profiling-env
    "${PRELOAD_ENV}" "${LIBRARY_PATH_ENV}"
    "ROCPROFILER_SAMPLE_OUTPUT_FILE=${counter-collection-device-profiling-output}"
    "ROCR_VISIBLE_DEVICES=0" "HIP_VISIBLE_DEVICES=0")

# The sample configures the first visible agent, so a single visible device keeps that
# agent and the workload's device the same one. The sampler thread runs on a fixed
# interval, so the iteration count must keep the run long enough to clear the validator's
# minimum sample count.
add_test(NAME counter-collection-device-profiling
         COMMAND $<TARGET_FILE:counter-collection-device-profiling> 50000 50 1)

set_tests_properties(
    counter-collection-device-profiling
    PROPERTIES TIMEOUT
               120
               LABELS
               "samples"
               ENVIRONMENT
               "${counter-collection-device-profiling-env}"
               FAIL_REGULAR_EXPRESSION
               "${ROCPROFILER_DEFAULT_FAIL_REGEX}"
               FIXTURES_SETUP
               "counter-collection-device-profiling-async"
               RESOURCE_LOCK
               "rocprofiler_device_counting_pmc"
               ATTACHED_FILES_ON_FAIL
               "${counter-collection-device-profiling-output}"
               DISABLED
               "${IS_THREAD_SANITIZER}")

add_library(counter-collection-device-profiling-sync-client SHARED)
target_sources(counter-collection-device-profiling-sync-client
               PRIVATE device_counting_sync_client.cpp client.hpp)
target_link_libraries(
    counter-collection-device-profiling-sync-client
    PUBLIC rocprofiler-sdk::samples-build-flags
    PRIVATE rocprofiler-sdk::rocprofiler-sdk rocprofiler-sdk::samples-common-library)

add_executable(counter-collection-device-profiling-sync)
target_sources(counter-collection-device-profiling-sync PRIVATE main.cpp)
target_link_libraries(
    counter-collection-device-profiling-sync
    PRIVATE counter-collection-device-profiling-sync-client Threads::Threads)

rocprofiler_samples_get_preload_env(PRELOAD_ENV
                                    counter-collection-device-profiling-sync-client)

set(counter-collection-device-profiling-sync-output
    "${CMAKE_CURRENT_BINARY_DIR}/counter-collection-device-profiling-sync.log")
set(counter-collection-device-profiling-sync-env
    "${PRELOAD_ENV}" "${LIBRARY_PATH_ENV}"
    "ROCPROFILER_SAMPLE_OUTPUT_FILE=${counter-collection-device-profiling-sync-output}"
    "ROCR_VISIBLE_DEVICES=0" "HIP_VISIBLE_DEVICES=0")

# The sample configures the first visible agent, so a single visible device keeps that
# agent and the workload's device the same one. The sampler thread runs on a fixed
# interval, so the iteration count must keep the run long enough to clear the validator's
# minimum sample count.
add_test(NAME counter-collection-device-profiling-sync
         COMMAND $<TARGET_FILE:counter-collection-device-profiling-sync> 50000 50 1)

set_tests_properties(
    counter-collection-device-profiling-sync
    PROPERTIES TIMEOUT
               120
               LABELS
               "samples"
               ENVIRONMENT
               "${counter-collection-device-profiling-sync-env}"
               FAIL_REGULAR_EXPRESSION
               "${ROCPROFILER_DEFAULT_FAIL_REGEX}"
               FIXTURES_SETUP
               "counter-collection-device-profiling-sync"
               RESOURCE_LOCK
               "rocprofiler_device_counting_pmc"
               ATTACHED_FILES_ON_FAIL
               "${counter-collection-device-profiling-sync-output}"
               DISABLED
               "${IS_THREAD_SANITIZER}")

if(BUILD_TESTING AND Python3_Interpreter_FOUND)
    configure_file(test_device_counting.py test_device_counting.py COPYONLY)

    if(rocprofiler_sdk_pytest_FOUND)
        set(counter-collection-device-profiling-validator
            "${ROCPSDK_PYTEST_EXECUTABLE}" -x -v -rs
            "${CMAKE_CURRENT_BINARY_DIR}/test_device_counting.py")
    else()
        set(counter-collection-device-profiling-validator
            "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_BINARY_DIR}/test_device_counting.py")
    endif()

    add_test(NAME counter-collection-device-profiling-validate
             COMMAND ${counter-collection-device-profiling-validator})

    set_tests_properties(
        counter-collection-device-profiling-validate
        PROPERTIES
            TIMEOUT
            120
            LABELS
            "samples;validation"
            WORKING_DIRECTORY
            "${CMAKE_CURRENT_BINARY_DIR}"
            ENVIRONMENT
            "ROCPROFILER_SAMPLE_ASYNC_OUTPUT_FILE=${counter-collection-device-profiling-output};ROCPROFILER_SAMPLE_SYNC_OUTPUT_FILE=${counter-collection-device-profiling-sync-output}"
            FIXTURES_REQUIRED
            "counter-collection-device-profiling-async;counter-collection-device-profiling-sync"
            FAIL_REGULAR_EXPRESSION
            "${ROCPROFILER_DEFAULT_FAIL_REGEX}"
            ATTACHED_FILES_ON_FAIL
            "${counter-collection-device-profiling-output};${counter-collection-device-profiling-sync-output}"
            DISABLED
            "${IS_THREAD_SANITIZER}")
elseif(BUILD_TESTING)
    message(
        WARNING
            "Python3 interpreter not found; counter-collection device-profiling validation will not be registered"
        )
endif()
