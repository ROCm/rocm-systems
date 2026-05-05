# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT

include(AISAddExecutable)

ais_add_executable(
    NAME read-write-check
    DEPS hipfile
    SRCS "read-write-check.cpp"
    SYSINCLS ${HIPFILE_INCLUDE_PATH}
)
