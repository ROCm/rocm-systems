# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

add_test(
    NAME rocprof-sys-unit-tests
    COMMAND rocprof-sys-unit-tests
    WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
)
