set(ROCPROFILER_EXPERIMENTAL_THREAD_TRACE_HEADER_FILES
    core.h agent.h dispatch.h trace_decoder_types.h trace_decoder.h)

install(
    FILES ${ROCPROFILER_EXPERIMENTAL_THREAD_TRACE_HEADER_FILES}
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/rocprofiler-sdk/experimental/thread-trace
    COMPONENT development)
