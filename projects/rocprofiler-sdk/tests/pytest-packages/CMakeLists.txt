#
#
#

set(PACKAGE_OUTPUT_DIR
    ${ROCPROFILER_SDK_TESTS_BINARY_DIR}/pytest-packages/rocprofiler_sdk)

if(NOT EXISTS ${CMAKE_CURRENT_BINARY_DIR}/trace_processor_shell.linux_clang_release)
    file(
        DOWNLOAD
        https://github.com/jrmadsen/perfetto-trace-processor-shell/releases/download/v0.0.4/trace_processor_shell.linux_clang_release
        ${CMAKE_CURRENT_BINARY_DIR}/trace_processor_shell.linux_clang_release
        EXPECTED_MD5 c25390cd99aa96622e328efa6699c6c6
        INACTIVITY_TIMEOUT 30
        TIMEOUT 300
        SHOW_PROGRESS)
    configure_file(
        ${CMAKE_CURRENT_BINARY_DIR}/trace_processor_shell.linux_clang_release
        ${PACKAGE_OUTPUT_DIR}/pytest_utils/trace_processor_shell
        COPYONLY
        FILE_PERMISSIONS
        OWNER_READ
        OWNER_WRITE
        OWNER_EXECUTE
        GROUP_READ
        GROUP_EXECUTE
        WORLD_READ
        WORLD_EXECUTE)
endif()

set(PYTEST_UTILS_PYTHON_SOURCES __init__.py dotdict.py perfetto_reader.py otf2_reader.py
                                rocpd_reader.py)
set(TESTS_PYTHON_SOURCES __init__.py rocprofv3.py)

foreach(_FILE ${PYTEST_UTILS_PYTHON_SOURCES})
    configure_file(${CMAKE_CURRENT_LIST_DIR}/pytest_utils/${_FILE}
                   ${PACKAGE_OUTPUT_DIR}/pytest_utils/${_FILE} COPYONLY)
endforeach()

foreach(_FILE ${TESTS_PYTHON_SOURCES})
    configure_file(${CMAKE_CURRENT_LIST_DIR}/tests/${_FILE}
                   ${PACKAGE_OUTPUT_DIR}/tests/${_FILE} COPYONLY)
endforeach()

file(
    WRITE "${PACKAGE_OUTPUT_DIR}/__init__.py"
    "#
from __future__ import absolute_import

from . import pytest_utils
")

add_subdirectory(pc_sampling)
