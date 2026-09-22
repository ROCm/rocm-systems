#
set(ROCPROFILER_LIB_TRACING_SOURCES profiling_time.cpp)
set(ROCPROFILER_LIB_TRACING_HEADERS fwd.hpp profiling_time.hpp tracing.hpp)

target_sources(rocprofiler-sdk-object-library PRIVATE ${ROCPROFILER_LIB_TRACING_SOURCES}
                                                      ${ROCPROFILER_LIB_TRACING_HEADERS})
