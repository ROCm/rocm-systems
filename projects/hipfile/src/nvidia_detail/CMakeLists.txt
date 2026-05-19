# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT

# CUDA-backed hipFile

set(HIPFILE_SOURCES
    "${HIPFILE_SRC_COMMON_PATH}/hipfile-common.cpp"
    hipfile.cpp
    hipfile-cufile.cpp
)

# Create hipfile target
ais_add_libraries(
    NAME    hipfile
    DETAIL  nvidia
    SRCS    ${HIPFILE_SOURCES}
    LIBINCL ${HIPFILE_INCLUDE_PATH}
)
