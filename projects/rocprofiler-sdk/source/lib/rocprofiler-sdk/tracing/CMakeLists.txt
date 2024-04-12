#
set(ROCPROFILER_LIB_TRACING_SOURCES)
set(ROCPROFILER_LIB_TRACING_HEADERS fwd.hpp tracing.hpp)

target_sources(rocprofiler-object-library PRIVATE ${ROCPROFILER_LIB_TRACING_SOURCES}
                                                  ${ROCPROFILER_LIB_TRACING_HEADERS})
