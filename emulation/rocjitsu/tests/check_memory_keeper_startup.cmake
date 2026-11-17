# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Missing arguments must reach main and return 1. Loader failure returns 127;
# sanitizer startup failures likewise cannot masquerade as a successful check.
execute_process(
    COMMAND env -i TMPDIR=/dev/null "${KEEPER}"
    RESULT_VARIABLE result
    ERROR_VARIABLE diagnostic
)
if(NOT result STREQUAL "1")
    message(
        FATAL_ERROR
        "Scrubbed memory keeper startup returned ${result}: ${diagnostic}"
    )
endif()
