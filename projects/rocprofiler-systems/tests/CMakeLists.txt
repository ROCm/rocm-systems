# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

#
# rocprofiler-systems tests
#
include_guard(GLOBAL)

if(ROCPROFSYS_USE_PYTESTS)
    include(${CMAKE_CURRENT_LIST_DIR}/pytest/CMakeLists.txt)
else()
    include(${CMAKE_CURRENT_LIST_DIR}/rocprof-sys-testing.cmake)

    # test groups
    include(${CMAKE_CURRENT_LIST_DIR}/rocprof-sys-unit-tests.cmake)
    include(${CMAKE_CURRENT_LIST_DIR}/rocprof-sys-config-tests.cmake)
    include(${CMAKE_CURRENT_LIST_DIR}/rocprof-sys-instrument-tests.cmake)
    include(${CMAKE_CURRENT_LIST_DIR}/rocprof-sys-preset-tests.cmake)
    include(${CMAKE_CURRENT_LIST_DIR}/rocprof-sys-pthread-tests.cmake)
    include(${CMAKE_CURRENT_LIST_DIR}/rocprof-sys-rocm-tests.cmake)
    include(${CMAKE_CURRENT_LIST_DIR}/rocprof-sys-user-api-tests.cmake)
    include(${CMAKE_CURRENT_LIST_DIR}/rocprof-sys-mpi-tests.cmake)
    include(${CMAKE_CURRENT_LIST_DIR}/rocprof-sys-ucx-tests.cmake)
    include(${CMAKE_CURRENT_LIST_DIR}/rocprof-sys-kokkos-tests.cmake)
    include(${CMAKE_CURRENT_LIST_DIR}/rocprof-sys-openmp-tests.cmake)
    include(${CMAKE_CURRENT_LIST_DIR}/rocprof-sys-code-coverage-tests.cmake)
    include(${CMAKE_CURRENT_LIST_DIR}/rocprof-sys-fork-tests.cmake)
    include(${CMAKE_CURRENT_LIST_DIR}/rocprof-sys-time-window-tests.cmake)
    include(${CMAKE_CURRENT_LIST_DIR}/rocprof-sys-attach-tests.cmake)
    include(${CMAKE_CURRENT_LIST_DIR}/rocprof-sys-rccl-tests.cmake)
    include(${CMAKE_CURRENT_LIST_DIR}/rocprof-sys-overflow-tests.cmake)
    include(${CMAKE_CURRENT_LIST_DIR}/rocprof-sys-annotate-tests.cmake)
    include(${CMAKE_CURRENT_LIST_DIR}/rocprof-sys-causal-tests.cmake)
    include(${CMAKE_CURRENT_LIST_DIR}/rocprof-sys-python-tests.cmake)
    include(${CMAKE_CURRENT_LIST_DIR}/rocprof-sys-decode-tests.cmake)
    include(${CMAKE_CURRENT_LIST_DIR}/rocprof-sys-gpu-connect-tests.cmake)
    include(${CMAKE_CURRENT_LIST_DIR}/rocprof-sys-nic-perf.cmake)
    include(${CMAKE_CURRENT_LIST_DIR}/rocprof-sys-roctx-tests.cmake)
    include(${CMAKE_CURRENT_LIST_DIR}/rocprof-sys-rocm-hip-stream.cmake)
    include(${CMAKE_CURRENT_LIST_DIR}/rocprof-sys-binary-tests.cmake)
    include(${CMAKE_CURRENT_LIST_DIR}/rocprof-sys-thread-limit-tests.cmake)

    # -------------------------------------------------------------------------------------- #
    #
    # Global cleanup test for temporary files
    # This runs once after ALL tests complete to clean up trace cache temporary files
    # Uses FIXTURES_CLEANUP to ensure it runs after all tests requiring the fixture
    #
    # -------------------------------------------------------------------------------------- #

    #delete temp files created by rocprofiler-sys tests in /tmp owned by the current user. Always return success.
    add_test(
        NAME rocprofsys-cleanup-tmp-files
        COMMAND
            sh -c
            "find /tmp -maxdepth 1 -user $(whoami) \\( -name 'buffered_storage*.bin' -o -name 'metadata*.json' \\) -delete 2>/dev/null || true"
        WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
    )

    set_tests_properties(
        rocprofsys-cleanup-tmp-files
        PROPERTIES
            FIXTURES_CLEANUP rocprofsys-global-tmp-files
            LABELS "cleanup;global"
            TIMEOUT 30
    )
endif()

# ------------------------------------------------------------------------------#
#
# Move test files to build directory
#
# ------------------------------------------------------------------------------#

set(ROCPROFSYS_PYTHON_VALIDATION_FILES
    ${CMAKE_CURRENT_LIST_DIR}/validate-causal-json.py
    ${CMAKE_CURRENT_LIST_DIR}/validate-perfetto-proto.py
    ${CMAKE_CURRENT_LIST_DIR}/validate-rocpd.py
    ${CMAKE_CURRENT_LIST_DIR}/validate-timemory-json.py
)

set(ROCPROFSYS_TEST_SCRIPTS
    ${CMAKE_CURRENT_LIST_DIR}/run-rocprof-sys-pid.sh
    ${CMAKE_CURRENT_LIST_DIR}/get_default_nic.sh
    ${CMAKE_CURRENT_LIST_DIR}/generate_papi_nic_events.sh
)

add_custom_target(
    copy-test-files
    ALL
    COMMAND
        ${CMAKE_COMMAND} -E make_directory
        ${CMAKE_BINARY_DIR}/share/rocprofiler-systems/tests
    COMMAND
        ${CMAKE_COMMAND} -E copy_if_different ${ROCPROFSYS_PYTHON_VALIDATION_FILES}
        ${CMAKE_BINARY_DIR}/share/rocprofiler-systems/tests
    # copy_directory for directories (copy_if_different only works for files)
    COMMAND
        ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_CURRENT_LIST_DIR}/rocpd-validation-rules
        ${CMAKE_BINARY_DIR}/share/rocprofiler-systems/tests/rocpd-validation-rules
    COMMAND
        ${CMAKE_COMMAND} -E copy_if_different ${ROCPROFSYS_TEST_SCRIPTS}
        ${CMAKE_BINARY_DIR}/share/rocprofiler-systems/tests
)

# ------------------------------------------------------------------------------#
#
# Pytests install
#
# ------------------------------------------------------------------------------#

if(ROCPROFSYS_INSTALL_TESTING)
    # Python Validation scripts
    install(
        PROGRAMS ${ROCPROFSYS_PYTHON_VALIDATION_FILES}
        DESTINATION share/rocprofiler-systems/tests
        COMPONENT rocprofiler-systems-tests
    )
    install(
        DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/rocpd-validation-rules
        DESTINATION share/rocprofiler-systems/tests
        COMPONENT rocprofiler-systems-tests
    )
    # Scripts
    install(
        PROGRAMS ${ROCPROFSYS_TEST_SCRIPTS}
        DESTINATION share/rocprofiler-systems/tests
        COMPONENT rocprofiler-systems-tests
    )
endif()
