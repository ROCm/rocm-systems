# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT

include(HipFileAddExecutable)

hipfile_add_executable(
    NAME ais-stats
    DEPS hipfile
    SRCS "ais-stats.cpp"
    SYSINCLS ${HIPFILE_AMD_SOURCE_PATH} ${HIPFILE_INCLUDE_PATH}
)
