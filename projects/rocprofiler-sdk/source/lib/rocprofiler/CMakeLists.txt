#
#
#
set(ROCPROFILER_LIB_HEADERS buffer.hpp internal_threading.hpp registration.hpp)
set(ROCPROFILER_LIB_SOURCES
    buffer.cpp buffer_tracing.cpp callback_tracing.cpp context.cpp internal_threading.cpp
    rocprofiler.cpp registration.cpp)

add_library(rocprofiler-library SHARED)
add_library(rocprofiler::rocprofiler-library ALIAS rocprofiler-library)

target_sources(rocprofiler-library PRIVATE ${ROCPROFILER_LIB_SOURCES}
                                           ${ROCPROFILER_LIB_HEADERS})

add_subdirectory(hsa)
add_subdirectory(context)

target_link_libraries(
    rocprofiler-library
    PUBLIC rocprofiler::rocprofiler-headers rocprofiler::rocprofiler-hsa-runtime
           rocprofiler::rocprofiler-hip
    PRIVATE rocprofiler::rocprofiler-build-flags rocprofiler::rocprofiler-memcheck
            rocprofiler::rocprofiler-common-library rocprofiler::rocprofiler-stdcxxfs
            rocprofiler::rocprofiler-dl rocprofiler::rocprofiler-amd-comgr)

set_target_properties(
    rocprofiler-library
    PROPERTIES OUTPUT_NAME rocprofiler64
               SOVERSION ${PROJECT_VERSION_MAJOR}
               VERSION ${PROJECT_VERSION}
               DEFINE_SYMBOL rocprofiler_EXPORTS)
