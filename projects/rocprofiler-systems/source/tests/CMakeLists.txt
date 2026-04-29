# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

set(UNIT_TEST_OBJECTS
    $<TARGET_OBJECTS:lib-common-tests>
    $<TARGET_OBJECTS:trace-cache-tests>
    $<TARGET_OBJECTS:demangler-tests>
    $<TARGET_OBJECTS:logger-tests>
    $<TARGET_OBJECTS:lib-components-tests>
    $<TARGET_OBJECTS:rocprofiler-sdk-components-unit-tests>
    $<$<TARGET_EXISTS:rccl-unit-tests>:$<TARGET_OBJECTS:rccl-unit-tests>>
    $<$<TARGET_EXISTS:bin-common-tests>:$<TARGET_OBJECTS:bin-common-tests>>
)

list(APPEND UNIT_TEST_OBJECTS $<TARGET_OBJECTS:pmc-common-tests>)
list(APPEND UNIT_TEST_OBJECTS $<TARGET_OBJECTS:pmc-gpu-collector-tests>)
list(APPEND UNIT_TEST_OBJECTS $<TARGET_OBJECTS:pmc-nic-collector-tests>)
list(APPEND UNIT_TEST_OBJECTS $<TARGET_OBJECTS:pmc-cpu-collector-tests>)
list(APPEND UNIT_TEST_OBJECTS $<TARGET_OBJECTS:procfs-driver-tests>)

add_executable(rocprof-sys-unit-tests ${UNIT_TEST_OBJECTS})

target_link_libraries(
    rocprof-sys-unit-tests
    PRIVATE
        rocprofiler-systems-googletest-library
        rocprofiler-systems-common-library
        rocprofiler-systems-core-library
        rocprofiler-systems::rocprofiler-systems-binary
        $<$<TARGET_EXISTS:rocprofiler-systems-bin-common>:rocprofiler-systems::rocprofiler-systems-bin-common>
        rocprofiler-systems::rocprofiler-systems-interface-library
        # Main static archive: pulls thread_info, pmc::postfork_*, runtime, and
        # the rest of the rocprof-sys library symbols. The sampling lifecycle
        # methods (folded out of real_production_hooks) reference these
        # unconditionally, so the unit-tests binary now links them just like
        # the sampling integration test binaries already do. The TSAN stress
        # test which previously motivated keeping this archive out has been
        # retired.
        rocprofiler-systems::librocprofiler-systems-static
        # Sampling unit + integration tests folded in; bench stays separate.
        rocprof-sys-sampling-tests-objs
)

target_include_directories(
    rocprof-sys-unit-tests
    PRIVATE ${PROJECT_SOURCE_DIR}/source/lib/rocprof-sys ${PROJECT_SOURCE_DIR}/source/lib
)
