# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT

# ROCm-backed hipFile

set(HIPFILE_SOURCES
    "${HIPFILE_SRC_COMMON_PATH}/hipfile-common.cpp"
    api_trace/api-dispatch-interface.cpp
    api_trace/api-trace.cpp
    async.cpp
    backend.cpp
    backend/asyncop-fallback.cpp
    backend/memcpy-kernel.hip
    backend/fallback.cpp
    backend/fastpath.cpp
    batch/batch.cpp
    buffer.cpp
    configuration.cpp
    context.cpp
    environment.cpp
    file.cpp
    file-descriptor.cpp
    hip.cpp
    hipfile.cpp
    hipfile-stats.cpp
    io.cpp
    mountinfo.cpp
    state.cpp
    stats.cpp
    stream.cpp
    sys.cpp
)

# Create hipfile target
ais_add_libraries(
    NAME     hipfile
    DETAIL   amd
    LIBS     mount
    SRCS     ${HIPFILE_SOURCES}
    LIBINCL  ${HIPFILE_INCLUDE_PATH}
    SYSINCLS ${HIPFILE_AMD_SOURCE_PATH}
)
