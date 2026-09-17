# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

set(UNIT_TEST_OBJECTS
    $<TARGET_OBJECTS:lib-common-tests>
    $<TARGET_OBJECTS:lib-common-units-tests>
    $<TARGET_OBJECTS:trace-cache-tests>
    $<TARGET_OBJECTS:demangler-tests>
    $<TARGET_OBJECTS:tracing-config-tests>
    $<TARGET_OBJECTS:control-posix-clock-tests>
    $<TARGET_OBJECTS:control-steady-clock-tests>
    $<TARGET_OBJECTS:control-session-scope-tests>
    $<TARGET_OBJECTS:trace-period-config-tests>
    $<TARGET_OBJECTS:track-registry-tests>
    $<TARGET_OBJECTS:packet-framing-tests>
    $<TARGET_OBJECTS:perfetto-engine-tests>
    $<TARGET_OBJECTS:perfetto-sinks-tests>
    $<TARGET_OBJECTS:emission-handshake-tls-contract-tests>
    $<TARGET_OBJECTS:output-format-tests>
    $<TARGET_OBJECTS:parse-numeric-range-tests>
    $<TARGET_OBJECTS:rank-passes-filter-tests>
    $<TARGET_OBJECTS:state-tests>
    $<TARGET_OBJECTS:progress-tests>
    $<TARGET_OBJECTS:logger-tests>
    $<TARGET_OBJECTS:lib-components-tests>
    $<TARGET_OBJECTS:rocprofiler-sdk-components-unit-tests>
    $<TARGET_OBJECTS:buffered-domains-unit-tests>
    $<TARGET_OBJECTS:callback-domains-unit-tests>
    $<$<TARGET_EXISTS:rccl-unit-tests>:$<TARGET_OBJECTS:rccl-unit-tests>>
    $<$<TARGET_EXISTS:bin-common-tests>:$<TARGET_OBJECTS:bin-common-tests>>
)

list(APPEND UNIT_TEST_OBJECTS $<TARGET_OBJECTS:amd-smi-backend-tests>)
list(APPEND UNIT_TEST_OBJECTS $<TARGET_OBJECTS:pmc-amd-smi-provider-tests>)
list(APPEND UNIT_TEST_OBJECTS $<TARGET_OBJECTS:pmc-common-tests>)
list(APPEND UNIT_TEST_OBJECTS $<TARGET_OBJECTS:pmc-gpu-collector-tests>)
list(APPEND UNIT_TEST_OBJECTS $<TARGET_OBJECTS:pmc-nic-collector-tests>)
list(APPEND UNIT_TEST_OBJECTS $<TARGET_OBJECTS:pmc-cpu-collector-tests>)
list(APPEND UNIT_TEST_OBJECTS $<TARGET_OBJECTS:procfs-backend-tests>)
list(
    APPEND UNIT_TEST_OBJECTS
    $<$<TARGET_EXISTS:rocprofiler-sdk-backend-tests>:$<TARGET_OBJECTS:rocprofiler-sdk-backend-tests>>
)
set(_rocprofsys_have_sdk_pmc_tests FALSE)

if(
    (ROCPROFSYS_ROCM_VERSION_MAJOR GREATER 6)
    OR (
        ROCPROFSYS_ROCM_VERSION_MAJOR EQUAL 6
        AND ROCPROFSYS_ROCM_VERSION_MINOR GREATER_EQUAL 4
    )
)
    set(_rocprofsys_have_sdk_pmc_tests TRUE)
endif()

list(
    APPEND UNIT_TEST_OBJECTS
    $<$<BOOL:${_rocprofsys_have_sdk_pmc_tests}>:$<TARGET_OBJECTS:pmc-sdk-pmc-collector-tests>>
)

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
)

target_include_directories(
    rocprof-sys-unit-tests
    PRIVATE ${PROJECT_SOURCE_DIR}/source/lib/rocprof-sys ${PROJECT_SOURCE_DIR}/source/lib
)

# Do NOT link rocprofiler-systems-object-library into unit tests.
# It contains library.cpp whose static initializers instantiate thread_data /
# thread_deleter, causing a TSan deadlock during atexit.
get_target_property(_ut_libs rocprof-sys-unit-tests LINK_LIBRARIES)

if(_ut_libs MATCHES "rocprofiler-systems-object-library")
    message(
        FATAL_ERROR
        "rocprof-sys-unit-tests must NOT link rocprofiler-systems-object-library. "
    )
endif()

# ------------------------------------------------------------------------------#
#
# Host-ASAN payload
#
# ------------------------------------------------------------------------------#

# tests/pytest/test_units.py marks these build_only, so the packaged test tree
# skips rocprof-sys-unit-tests and none of its cases run in CI. The CPU-only
# host-ASAN matrix consumes the installed binary together with
# host_asan_tests.json, which pins the exact cases admitted on a runner that has
# no GPU device nodes.
if(ROCPROFSYS_INSTALL_HOST_ASAN_TESTS)
    if(NOT ROCPROFSYS_INSTALL_TESTING)
        message(
            FATAL_ERROR
            "ROCPROFSYS_INSTALL_HOST_ASAN_TESTS requires ROCPROFSYS_INSTALL_TESTING"
        )
    endif()

    if(ROCPROFSYS_USE_SANITIZER AND ROCPROFSYS_SANITIZER_TYPE MATCHES "address")
        # -fsanitize=address alone links the runtime statically into an
        # executable, leaving no DT_NEEDED entry. The consumer verifies direct
        # shared linkage before it trusts a result, so request it explicitly.
        target_link_options(rocprof-sys-unit-tests PRIVATE -shared-libasan)

        # The runtime is libclang_rt.asan.so under a per-target runtime
        # directory and libclang_rt.asan-<arch>.so under the legacy layout.
        unset(_asan_runtime_dir)
        foreach(
            _asan_lib
            libclang_rt.asan.so
            libclang_rt.asan-${CMAKE_SYSTEM_PROCESSOR}.so
        )
            execute_process(
                COMMAND ${CMAKE_CXX_COMPILER} -print-file-name=${_asan_lib}
                OUTPUT_VARIABLE _asan_runtime
                OUTPUT_STRIP_TRAILING_WHITESPACE
                ERROR_QUIET
            )
            if(EXISTS "${_asan_runtime}")
                get_filename_component(_asan_runtime_dir "${_asan_runtime}" DIRECTORY)
                break()
            endif()
        endforeach()

        if(NOT _asan_runtime_dir)
            message(
                FATAL_ERROR
                "ROCPROFSYS_INSTALL_HOST_ASAN_TESTS requires the shared ASAN runtime, "
                "which ${CMAKE_CXX_COMPILER} did not resolve"
            )
        endif()

        set_target_properties(
            rocprof-sys-unit-tests
            PROPERTIES INSTALL_RPATH "${_asan_runtime_dir}"
        )
    endif()

    install(
        TARGETS rocprof-sys-unit-tests
        RUNTIME
            DESTINATION share/rocprofiler-systems/tests/host-asan
            COMPONENT rocprofiler-systems-tests
    )
    install(
        FILES ${CMAKE_CURRENT_LIST_DIR}/host_asan_tests.json
        DESTINATION share/rocprofiler-systems/tests/host-asan
        COMPONENT rocprofiler-systems-tests
    )
endif()
