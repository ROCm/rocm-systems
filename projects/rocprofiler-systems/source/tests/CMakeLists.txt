# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

set(UNIT_TEST_OBJECTS
    $<TARGET_OBJECTS:lib-common-tests>
    $<TARGET_OBJECTS:trace-cache-tests>
    $<TARGET_OBJECTS:demangler-tests>
    $<TARGET_OBJECTS:logger-tests>
    $<TARGET_OBJECTS:lib-components-tests>
    $<$<TARGET_EXISTS:rccl-unit-tests>:$<TARGET_OBJECTS:rccl-unit-tests>>
)

list(APPEND UNIT_TEST_OBJECTS $<TARGET_OBJECTS:pmc-common-tests>)
list(APPEND UNIT_TEST_OBJECTS $<TARGET_OBJECTS:pmc-gpu-collector-tests>)
if(ROCPROFSYS_BUILD_AINIC)
    list(APPEND UNIT_TEST_OBJECTS $<TARGET_OBJECTS:pmc-nic-collector-tests>)
endif()

add_executable(rocprof-sys-unit-tests ${UNIT_TEST_OBJECTS})

target_link_libraries(
    rocprof-sys-unit-tests
    PRIVATE
        rocprofiler-systems-googletest-library
        rocprofiler-systems-common-library
        rocprofiler-systems-core-library
        rocprofiler-systems-object-library
        rocprofiler-systems-pmc-library
        rocprofiler-systems::rocprofiler-systems-binary
)
