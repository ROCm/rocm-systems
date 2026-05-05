# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT

include(AISAddExecutable)

ais_add_executable(
    NAME offsetcp
    DEPS hipfile
    SRCS "offsetcp.cpp"
    SYSINCLS ${HIPFILE_INCLUDE_PATH}
)

