#
#
#

set(PACKAGE_OUTPUT_DIR
    ${ROCPROFILER_SDK_TESTS_BINARY_DIR}/pytest-packages/rocprofiler_sdk/pc_sampling)

file(
    WRITE "${PACKAGE_OUTPUT_DIR}/__init__.py"
    "#
from __future__ import absolute_import

from . import exec_mask_manipulation
")

add_subdirectory(exec_mask_manipulation)
add_subdirectory(stochastic)
add_subdirectory(transpose_multiple_agents)
