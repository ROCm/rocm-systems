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
    thread-pool.cpp
)

# Create hipfile target
ais_add_libraries(
    NAME     hipfile
    DETAIL   amd
    LIBS     mount
    SRCS     ${HIPFILE_SOURCES}
    LIBINCL  ${HIPFILE_INCLUDE_PATH}
    SYSINCLS ${HIPFILE_AMD_SOURCE_PATH} ${HIPFILE_THIRD_PARTY_PATH}
)

# There are some warnings that appear while compiling thread-pool.cpp due
# to inlining from Taskflow, so we see these warnings even with third-party
# set as a system include directory. Disable these warnings.
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set_source_files_properties(
        thread-pool.cpp
        PROPERTIES
            COMPILE_OPTIONS
                "-Wno-null-dereference;-Wno-array-bounds"
    )
endif()

if(HIPFILE_ROCPROFILER_REGISTER)
    target_compile_definitions(
        hipfile
        PRIVATE
            HIPFILE_ROCPROFILER_REGISTER=1
            HIPFILE_ROCP_REG_VERSION_MAJOR=${AIS_LIBRARY_MAJOR}
            HIPFILE_ROCP_REG_VERSION_MINOR=${AIS_LIBRARY_MINOR}
            HIPFILE_ROCP_REG_VERSION_PATCH=${AIS_LIBRARY_PATCH})
    target_link_libraries(hipfile PRIVATE rocprofiler-register::rocprofiler-register)
    set_target_properties(hipfile PROPERTIES INSTALL_RPATH "\$ORIGIN")
endif()
