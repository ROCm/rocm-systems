#
#
#

set(PACKAGE_OUTPUT_DIR
    ${ROCPROFILER_SDK_TESTS_BINARY_DIR}/pytest-packages/rocprofiler_sdk)

set(PYTEST_UTILS_PYTHON_SOURCES __init__.py dotdict.py perfetto_reader.py otf2_reader.py)
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
