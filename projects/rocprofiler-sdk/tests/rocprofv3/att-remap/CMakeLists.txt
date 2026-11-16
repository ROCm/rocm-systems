#
# rocprofv3 --att-remap: the remapper (projects/remapper) runs a dispatch's kernel replay loop and
# traces each SIMD of one WGP on its own pass. Needs gfx1250, the trace decoder and libremapper.so.
#
cmake_minimum_required(VERSION 3.21.0 FATAL_ERROR)

project(rocprofiler-sdk-tests-rocprofv3-att-remap LANGUAGES CXX)

find_package(rocprofiler-sdk REQUIRED)

rocprofiler_configure_pytest_files(CONFIG pytest.ini COPY validate.py conftest.py)

rocprofiler_sdk_find_trace_decoder()

find_library(
    remapper_LIBRARY
    NAMES remapper
    HINTS ${remapper_ROOT} ${ROCM_PATH} ENV ROCM_PATH /opt/rocm
    PATH_SUFFIXES lib)
find_file(
    remapper_LAYOUT_SCRIPT layout.py
    HINTS ${remapper_ROOT} ${ROCM_PATH} ENV ROCM_PATH /opt/rocm
    PATH_SUFFIXES libexec/remapper/att_merge)
mark_as_advanced(remapper_LIBRARY remapper_LAYOUT_SCRIPT)

set(ATT_REMAP_DISABLED ON)
if(attdecoder_FOUND AND remapper_LIBRARY)
    set(ATT_REMAP_DISABLED OFF)
endif()

set(ATT_REMAP_OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/att-remap)

rocprofiler_add_integration_execute_test(
    rocprofv3-test-att-remap-generate
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> --att --att-library-path
        ${attdecoder_LIB_DIR} --att-remap --att-remap-library ${remapper_LIBRARY}
        --att-remap-options repetitions=1 --att-remap-merge False --att-target-cu 0
        --kernel-include-regex tiled_k_loop --replay-mode kernel
        --kernel-replay-beta-enabled --output-format json -d ${ATT_REMAP_OUTPUT}/%pid% -o
        out -- $<TARGET_FILE:att-remap> 2048 16
    DEPENDS att-remap
    TIMEOUT 600
    LABELS "kernel-replay;att-remap"
    DISABLED ${ATT_REMAP_DISABLED}
    FIXTURES_SETUP rocprofv3-test-att-remap-data)

rocprofiler_add_integration_validate_test(
    rocprofv3-test-att-remap-validate BUNDLE_TESTS
    TEST_PATHS validate.py
    COPY conftest.py
    ARGS --output-root ${ATT_REMAP_OUTPUT} --layout-script "${remapper_LAYOUT_SCRIPT}" -x
    TIMEOUT 120
    LABELS "kernel-replay;att-remap"
    DISABLED ${ATT_REMAP_DISABLED}
    FIXTURES_REQUIRED rocprofv3-test-att-remap-data)
