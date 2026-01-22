# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

#
# rocprofiler-systems pytest install
#
include_guard(GLOBAL)

# ------------------------------------------------------------------------------#
# Pytest specific
# ------------------------------------------------------------------------------#

set(ROCPROFSYS_PYTEST_PACKAGE_FILES
    ${CMAKE_CURRENT_LIST_DIR}/rocprofsys/__init__.py
    ${CMAKE_CURRENT_LIST_DIR}/rocprofsys/config.py
    ${CMAKE_CURRENT_LIST_DIR}/rocprofsys/gpu.py
    ${CMAKE_CURRENT_LIST_DIR}/rocprofsys/runners.py
    ${CMAKE_CURRENT_LIST_DIR}/rocprofsys/validators.py
)

file(GLOB ROCPROFSYS_PYTEST_TEST_FILES "${CMAKE_CURRENT_LIST_DIR}/test_*.py")
set(ROCPROFSYS_PYTEST_FILES
    ${CMAKE_CURRENT_LIST_DIR}/conftest.py
    ${ROCPROFSYS_PYTEST_TEST_FILES}
)

add_custom_target(
    copy-pytest-files
    ALL
    DEPENDS ${ROCPROFSYS_PYTEST_PACKAGE_FILES} ${ROCPROFSYS_PYTEST_FILES}
    COMMAND
        ${CMAKE_COMMAND} -E make_directory
        ${CMAKE_BINARY_DIR}/share/rocprofiler-systems/tests/pytest/rocprofsys
    COMMAND
        ${CMAKE_COMMAND} -E copy_if_different ${ROCPROFSYS_PYTEST_PACKAGE_FILES}
        ${CMAKE_BINARY_DIR}/share/rocprofiler-systems/tests/pytest/rocprofsys/
    COMMAND
        ${CMAKE_COMMAND} -E copy_if_different ${ROCPROFSYS_PYTEST_FILES}
        ${CMAKE_BINARY_DIR}/share/rocprofiler-systems/tests/pytest/
    COMMAND
        ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_SOURCE_DIR}/requirements.txt
        ${CMAKE_BINARY_DIR}/share/rocprofiler-systems/tests/
    COMMAND
        ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_LIST_DIR}/README.md
        ${CMAKE_BINARY_DIR}/share/rocprofiler-systems/tests/
    COMMENT "Copying pytest files to build directory"
)

if(ROCPROFSYS_INSTALL_TESTING)
    # Under install mode, run build_standalone.sh under default mode
    add_custom_command(
        OUTPUT ${CMAKE_BINARY_DIR}/share/rocprofiler-systems/tests/rocprofsys-tests.pyz
        COMMAND
            ${CMAKE_CURRENT_LIST_DIR}/build_standalone.sh --output-dir
            ${CMAKE_BINARY_DIR}/share/rocprofiler-systems/tests
        DEPENDS
            ${CMAKE_CURRENT_LIST_DIR}/build_standalone.sh
            ${ROCPROFSYS_PYTEST_PACKAGE_FILES}
            ${ROCPROFSYS_PYTEST_FILES}
        COMMENT "Building standalone pytest binary"
        VERBATIM
    )

    add_custom_target(
        build-standalone-pytest
        ALL
        DEPENDS ${CMAKE_BINARY_DIR}/share/rocprofiler-systems/tests/rocprofsys-tests.pyz
    )

    install(
        FILES ${CMAKE_BINARY_DIR}/share/rocprofiler-systems/tests/rocprofsys-tests.pyz
        DESTINATION share/rocprofiler-systems/tests
        COMPONENT rocprofiler-systems-tests
    )
    install(
        FILES ${CMAKE_BINARY_DIR}/share/rocprofiler-systems/tests/requirements.txt
        DESTINATION share/rocprofiler-systems/tests
        COMPONENT rocprofiler-systems-tests
    )
    install(
        FILES ${CMAKE_BINARY_DIR}/share/rocprofiler-systems/tests/README.md
        DESTINATION share/rocprofiler-systems/tests
        COMPONENT rocprofiler-systems-tests
    )
endif()
