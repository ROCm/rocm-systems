#
# SDMA copy profiling tests (ROCM-30729)
#
cmake_minimum_required(VERSION 3.24 FATAL_ERROR)

project(
    rocprofiler-sdk-tests-sdma-copy
    LANGUAGES CXX
    VERSION 0.0.0)

find_package(rocprofiler-sdk REQUIRED)

# tests/bin/sdma-copy holds a dependency signal on the host while profiled SDMA copies
# wait on it, then checks that no copy completes before the release, that every copy
# starts no earlier than the release, has a positive duration, reaches the expected
# completion-signal value and copies the right data. The application prints "SKIP: ..."
# and exits with 77 when the route has no SDMA engine or there are too few agents.
function(rocprofiler_sdk_add_sdma_copy_test NAME)
    rocprofiler_add_integration_execute_test(
        ${NAME}
        TARGET sdma-copy
        ARGS ${ARGN}
        TIMEOUT 60
        WSL_TIMEOUT_SCALE
        LABELS "integration-tests;sdma"
        ENVIRONMENT "HSA_ENABLE_SDMA=1"
        SKIP_REGULAR_EXPRESSION "SKIP:")
endfunction()

rocprofiler_sdk_add_sdma_copy_test(sdma-copy-profiling)
rocprofiler_sdk_add_sdma_copy_test(sdma-copy-profiling-counted --completion-value 2)
rocprofiler_sdk_add_sdma_copy_test(sdma-copy-profiling-counted-64bit --completion-value
                                   4294967297)
rocprofiler_sdk_add_sdma_copy_test(sdma-copy-profiling-no-dependencies --dependencies 0)
rocprofiler_sdk_add_sdma_copy_test(sdma-copy-profiling-multiple-dependencies
                                   --dependencies 3)
