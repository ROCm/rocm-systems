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
    COMMAND
        ${CMAKE_COMMAND} -E make_directory
        ${CMAKE_BINARY_DIR}/share/rocprofiler-systems/tests/pytest/rocprofsys
    COMMAND
        ${CMAKE_COMMAND} -E copy_if_different ${ROCPROFSYS_PYTEST_PACKAGE_FILES}
        ${CMAKE_BINARY_DIR}/share/rocprofiler-systems/tests/pytest/rocprofsys/
    COMMAND
        ${CMAKE_COMMAND} -E copy_if_different ${ROCPROFSYS_PYTEST_FILES}
        ${CMAKE_BINARY_DIR}/share/rocprofiler-systems/tests/pytest/
    COMMENT "Copying pytest files to build directory"
)

if(ROCPROFSYS_INSTALL_TESTING)
    # Under install mode, run build_standalone.sh under default mode
    execute_process(
        COMMAND
            ${CMAKE_CURRENT_LIST_DIR}/build_standalone.sh --output-dir
            ${CMAKE_BINARY_DIR}/share/rocprofiler-systems
        WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
        RESULT_VARIABLE BUILD_STANDALONE_RESULT
        OUTPUT_VARIABLE BUILD_STANDALONE_OUTPUT
        ERROR_VARIABLE BUILD_STANDALONE_ERROR
    )

    if(NOT BUILD_STANDALONE_RESULT EQUAL 0)
        rocprofiler_systems_message(FATAL_ERROR "Build standalone binary failed (${BUILD_STANDALONE_RESULT}):\n${BUILD_STANDALONE_OUTPUT}\n${BUILD_STANDALONE_ERROR}")
    endif()

    if(BUILD_STANDALONE_RESULT EQUAL 0)
        install(
            FILES ${CMAKE_BINARY_DIR}/share/rocprofiler-systems/rocprofsys-tests.pyz
            DESTINATION share/rocprofiler-systems
        )
        install(
            FILES ${CMAKE_SOURCE_DIR}/requirements.txt
            DESTINATION share/rocprofiler-systems
        )
    endif()
endif()
