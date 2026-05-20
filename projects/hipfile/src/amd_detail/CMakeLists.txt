# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT

# ROCm-backed hipFile

set(HIPFILE_SOURCES
    "${HIPFILE_SRC_COMMON_PATH}/hipfile-common.cpp"
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

if (HIPFILE_ROCPROFILER_REGISTER AND rocprofiler-register_FOUND)
    set(HIPFILE_SOURCES
        api_trace/api-dispatch-interface.cpp
        api_trace/api-trace.cpp
        ${HIPFILE_SOURCES}
    )
endif()

# Create hipfile target
ais_add_libraries(
    NAME     hipfile
    DETAIL   amd
    LIBS     mount
    SRCS     ${HIPFILE_SOURCES}
    LIBINCL  ${HIPFILE_INCLUDE_PATH}
    SYSINCLS ${HIPFILE_AMD_SOURCE_PATH}
)

if (HIPFILE_ROCPROFILER_REGISTER AND rocprofiler-register_FOUND)
    target_link_libraries(hipfile PRIVATE rocprofiler-register::rocprofiler-register)
    target_compile_definitions(
        hipfile
        PRIVATE
            HIPFILE_ROCPROFILER_REGISTER=1
            HIPFILE_ROCP_REG_VERSION_MAJOR=${PROJECT_VERSION_MAJOR}
            HIPFILE_ROCP_REG_VERSION_MINOR=${PROJECT_VERSION_MINOR}
            HIPFILE_ROCP_REG_VERSION_PATCH=${PROJECT_VERSION_PATCH}
    )
endif()
