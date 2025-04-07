#
# rocprofv3 tool tests for negating tracing options
#
cmake_minimum_required(VERSION 3.21.0 FATAL_ERROR)

project(
    rocprofiler-tests-rocprofv3-negate-aggregate-tracing-options
    LANGUAGES CXX
    VERSION 0.0.0)

find_package(rocprofiler-sdk REQUIRED)

string(REPLACE "LD_PRELOAD=" "ROCPROF_PRELOAD=" PRELOAD_ENV
               "${ROCPROFILER_MEMCHECK_PRELOAD_ENV}")

set(negate-aggregate-tracing-options-env "${PRELOAD_ENV}")

rocprofiler_configure_pytest_files(CONFIG input-negate-aggregate-tracing-options.yml)

##########################################################################################
#
#   Command line input
#
##########################################################################################

add_test(
    NAME rocprofv3-test-negate-aggregate-tracing-options-cmd-line
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> -M --sys-trace -d
        ${CMAKE_CURRENT_BINARY_DIR} -o out --log-level info --hip-trace=no --hsa-trace=no
        --marker-trace=no --kernel-trace=no --memory-copy-trace=no
        --memory-allocation-trace=no --scratch-memory-trace=no --rccl-trace=no
        --kokkos-trace=no --rocdecode-trace=no --rocjpeg-trace=no --hip-runtime-trace=no
        --hip-compiler-trace=no --hsa-core-trace=no --hsa-amd-trace=no
        --hsa-image-trace=no --hsa-finalizer-trace=no --advanced-thread-trace=no
        --kernel-rename=no --summary --echo -- ${CMAKE_COMMAND})

set_tests_properties(
    rocprofv3-test-negate-aggregate-tracing-options-cmd-line
    PROPERTIES TIMEOUT
               45
               LABELS
               "integration-tests"
               ENVIRONMENT
               "${negate-aggregate-tracing-options-env}"
               PASS_REGULAR_EXPRESSION
               "command:.*${CMAKE_COMMAND}"
               WILL_FAIL
               ON)

##########################################################################################
#
#   YAML input
#
##########################################################################################

add_test(
    NAME rocprofv3-test-negate-aggregate-tracing-options-inp-yaml
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> -i
        ${CMAKE_CURRENT_BINARY_DIR}/input-negate-aggregate-tracing-options.yml --
        ${CMAKE_COMMAND})

set_tests_properties(
    rocprofv3-test-negate-aggregate-tracing-options-inp-yaml
    PROPERTIES TIMEOUT
               45
               LABELS
               "integration-tests"
               ENVIRONMENT
               "${negate-aggregate-tracing-options-env}"
               PASS_REGULAR_EXPRESSION
               "command:.*${CMAKE_COMMAND}"
               WILL_FAIL
               ON)
