# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

#
# rocprofiler-systems tests
#
include_guard(GLOBAL)

include(${CMAKE_CURRENT_LIST_DIR}/pytest/CMakeLists.txt)
include(${CMAKE_CURRENT_LIST_DIR}/rocprof-sys-pytest.cmake)

# ------------------------------------------------------------------------------#
#
# Move test files to build directory
#
# ------------------------------------------------------------------------------#

set(ROCPROFSYS_PYTHON_VALIDATION_FILES
    ${CMAKE_CURRENT_LIST_DIR}/check_amd_smi_metrics.py
    ${CMAKE_CURRENT_LIST_DIR}/validate-causal-json.py
    ${CMAKE_CURRENT_LIST_DIR}/validate-perfetto-proto.py
    ${CMAKE_CURRENT_LIST_DIR}/validate-rocpd.py
    ${CMAKE_CURRENT_LIST_DIR}/validate-timemory-json.py
    ${CMAKE_CURRENT_LIST_DIR}/validate-unified-memory.py
)

set(ROCPROFSYS_TEST_SCRIPTS
    ${CMAKE_CURRENT_LIST_DIR}/get_default_nic.sh
    ${CMAKE_CURRENT_LIST_DIR}/generate_papi_nic_events.sh
    ${CMAKE_CURRENT_LIST_DIR}/run_rocprofiler_systems.py
)

set(_capchk_dir ${CMAKE_BINARY_DIR}/share/rocprofiler-systems/tests)
add_executable(
    rocprof-sys-capchk
    EXCLUDE_FROM_ALL
    ${CMAKE_CURRENT_LIST_DIR}/rocprof-sys-capchk.cpp
)
set_target_properties(
    rocprof-sys-capchk
    PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
        CXX_EXTENSIONS OFF
        RUNTIME_OUTPUT_DIRECTORY "${_capchk_dir}"
)

# test_categories.yaml drives tier (quick/standard/comprehensive/full) label
# injection at CTest-generate time (see conftest.py:_load_test_categories).
set(ROCPROFSYS_TEST_CATEGORIES_YAML ${CMAKE_CURRENT_LIST_DIR}/test_categories.yaml)
configure_file(
    ${ROCPROFSYS_TEST_CATEGORIES_YAML}
    ${CMAKE_BINARY_DIR}/share/rocprofiler-systems/tests/test_categories.yaml
    COPYONLY
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
    COMMAND
        ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_LIST_DIR}/README.md
        ${CMAKE_BINARY_DIR}/share/rocprofiler-systems/tests
    COMMAND
        ${CMAKE_COMMAND} -E copy_if_different ${ROCPROFSYS_TEST_CATEGORIES_YAML}
        ${CMAKE_BINARY_DIR}/share/rocprofiler-systems/tests
    COMMAND
        ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_CURRENT_LIST_DIR}/rocpd-validation-rules
        ${CMAKE_BINARY_DIR}/share/rocprofiler-systems/tests/rocpd-validation-rules
    COMMAND
        ${CMAKE_COMMAND} -E copy_if_different ${ROCPROFSYS_TEST_SCRIPTS}
        ${CMAKE_BINARY_DIR}/share/rocprofiler-systems/tests
)
add_dependencies(copy-test-files rocprof-sys-capchk)

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
    # Capability check binary
    install(
        TARGETS rocprof-sys-capchk
        RUNTIME
            DESTINATION share/rocprofiler-systems/tests
            COMPONENT rocprofiler-systems-tests
    )
    # Test categories policy
    install(
        FILES ${ROCPROFSYS_TEST_CATEGORIES_YAML}
        DESTINATION share/rocprofiler-systems/tests
        COMPONENT rocprofiler-systems-tests
    )
endif()
