set(ROCPROFILER_EXPERIMENTAL_HEADER_FILES counters.h thread_trace.h spm.h)

install(
    FILES ${ROCPROFILER_EXPERIMENTAL_HEADER_FILES}
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/rocprofiler-sdk/experimental
    COMPONENT development)

add_subdirectory(thread-trace)
add_subdirectory(spm)
