# MIT License
#
# Copyright (c) 2023-2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

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

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/input-negate-aggregate-tracing-options.yml.in
    ${CMAKE_CURRENT_BINARY_DIR}/input-negate-aggregate-tracing-options.yml @ONLY)

##########################################################################################
#
#   Command line input
#
##########################################################################################

rocprofiler_add_integration_execute_test(
    rocprofv3-test-negate-aggregate-tracing-options-cmd-line
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> -M --sys-trace -d
        ${CMAKE_CURRENT_BINARY_DIR} -o out --log-level info --hip-trace=no --hsa-trace=no
        --marker-trace=no --pytorch-trace=no --kernel-trace=no --memory-copy-trace=no
        --memory-allocation-trace=no --scratch-memory-trace=no --rccl-trace=no
        --kokkos-trace=no --ompt-trace=no --rocdecode-trace=no --rocjpeg-trace=no
        --rocshmem-trace=no --hip-runtime-trace=no --hip-compiler-trace=no
        --hsa-core-trace=no --hsa-amd-trace=no --hsa-image-trace=no
        --hsa-finalizer-trace=no --advanced-thread-trace=no --kfd-trace=no
        --kfd-page-migration-trace=no --kfd-page-mapping-trace=no --kfd-queue-trace=no
        --kfd-dropped-events-trace=no --kernel-rename=no --summary --echo
        --hipfile-trace=no -- ${CMAKE_COMMAND}
    DEPENDS rocprofiler-sdk::rocprofv3
    TIMEOUT 45
    LABELS "integration-tests"
    PRELOAD "${PRELOAD_ENV}"
    WILL_FAIL)

rocprofiler_add_integration_execute_test(
    rocprofv3-test-pytorch-trace-environment
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> --pytorch-trace --marker-trace=no
        --selected-regions=no --selected-regions-ref-count=no -- ${CMAKE_COMMAND} -P
        ${CMAKE_CURRENT_SOURCE_DIR}/validate-pytorch-trace-env.cmake
    DEPENDS rocprofiler-sdk::rocprofv3
    TIMEOUT 45
    LABELS "integration-tests"
    PRELOAD "${PRELOAD_ENV}"
    ENVIRONMENT
        "TORCH_PROFILER_EMIT_ROCTX=0;TORCH_PROFILER_ROCTX_SELECTED_REGIONS=0;ROCPROF_MARKER_API_TRACE=0;ROCPROF_SELECTED_REGIONS=0;ROCPROF_SELECTED_REGIONS_REF_COUNT=0"
    )

rocprofiler_add_integration_execute_test(
    rocprofv3-test-pytorch-trace-disabled-environment
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> --pytorch-trace=no -- ${CMAKE_COMMAND}
        -DEXPECTED_ROCTX=0 -P ${CMAKE_CURRENT_SOURCE_DIR}/validate-pytorch-trace-env.cmake
    DEPENDS rocprofiler-sdk::rocprofv3
    TIMEOUT 45
    LABELS "integration-tests"
    PRELOAD "${PRELOAD_ENV}"
    ENVIRONMENT "TORCH_PROFILER_EMIT_ROCTX=1;TORCH_PROFILER_ROCTX_SELECTED_REGIONS=1")

rocprofiler_add_integration_execute_test(
    rocprofv3-test-pytorch-trace-attach-will-fail
    COMMAND ${CMAKE_COMMAND} -DROCPROFV3=$<TARGET_FILE:rocprofiler-sdk::rocprofv3> -P
            ${CMAKE_CURRENT_SOURCE_DIR}/validate-pytorch-trace-attach.cmake
    DEPENDS rocprofiler-sdk::rocprofv3
    TIMEOUT 45
    LABELS "integration-tests"
    PRELOAD "${PRELOAD_ENV}")

rocprofiler_add_integration_execute_test(
    rocprofv3-test-pytorch-trace-collection-period-will-fail
    COMMAND
        ${CMAKE_COMMAND} -DROCPROFV3=$<TARGET_FILE:rocprofiler-sdk::rocprofv3>
        -DCONFLICT=collection-period -P
        ${CMAKE_CURRENT_SOURCE_DIR}/validate-pytorch-trace-conflict.cmake
    DEPENDS rocprofiler-sdk::rocprofv3
    TIMEOUT 45
    LABELS "integration-tests"
    PRELOAD "${PRELOAD_ENV}")

rocprofiler_add_integration_execute_test(
    rocprofv3-test-pytorch-trace-att-consecutive-kernels-will-fail
    COMMAND
        ${CMAKE_COMMAND} -DROCPROFV3=$<TARGET_FILE:rocprofiler-sdk::rocprofv3>
        -DCONFLICT=att-consecutive-kernels -P
        ${CMAKE_CURRENT_SOURCE_DIR}/validate-pytorch-trace-conflict.cmake
    DEPENDS rocprofiler-sdk::rocprofv3
    TIMEOUT 45
    LABELS "integration-tests"
    PRELOAD "${PRELOAD_ENV}")

rocprofiler_add_integration_execute_test(
    rocprofv3-test-pytorch-trace-att-no-intercept-will-fail
    COMMAND
        ${CMAKE_COMMAND} -DROCPROFV3=$<TARGET_FILE:rocprofiler-sdk::rocprofv3>
        -DCONFLICT=att-no-intercept -P
        ${CMAKE_CURRENT_SOURCE_DIR}/validate-pytorch-trace-conflict.cmake
    DEPENDS rocprofiler-sdk::rocprofv3
    TIMEOUT 45
    LABELS "integration-tests"
    PRELOAD "${PRELOAD_ENV}")

##########################################################################################
#
#   YAML input
#
##########################################################################################

rocprofiler_add_integration_execute_test(
    rocprofv3-test-negate-aggregate-tracing-options-inp-yaml
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> -i
        ${CMAKE_CURRENT_BINARY_DIR}/input-negate-aggregate-tracing-options.yml --
        ${CMAKE_COMMAND}
    DEPENDS rocprofiler-sdk::rocprofv3
    TIMEOUT 45
    LABELS "integration-tests"
    PRELOAD "${PRELOAD_ENV}"
    WILL_FAIL)
