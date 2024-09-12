#
#
#
set(ROCPROFILER_LIB_RCCL_DETAILS_SOURCES)
set(ROCPROFILER_LIB_RCCL_DETAILS_HEADERS api_trace.h)

target_sources(rocprofiler-object-library PRIVATE ${ROCPROFILER_LIB_RCCL_DETAILS_SOURCES}
                                                  ${ROCPROFILER_LIB_RCCL_DETAILS_HEADERS})
