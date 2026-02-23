# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

add_executable(
    rocprof-sys-unit-tests
    $<TARGET_OBJECTS:lib-common-tests>
    $<TARGET_OBJECTS:trace-cache-tests>
    $<TARGET_OBJECTS:demangler-tests>
    $<TARGET_OBJECTS:logger-tests>
    $<$<TARGET_EXISTS:rccl-unit-tests>:$<TARGET_OBJECTS:rccl-unit-tests>>
)

target_link_libraries(
    rocprof-sys-unit-tests
    PRIVATE
        rocprofiler-systems-googletest-library
        rocprofiler-systems-common-library
        rocprofiler-systems-core-library
)
