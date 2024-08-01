#
#
#   Installation of amd_detail headers
#
#
set(ROCPROFILER_AMD_DETAIL_HEADER_FILES thread_trace.h thread_trace_core.h
                                        thread_trace_dispatch.h thread_trace_agent.h)

install(
    FILES ${ROCPROFILER_AMD_DETAIL_HEADER_FILES}
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/rocprofiler-sdk/amd_detail/
    COMPONENT development)
