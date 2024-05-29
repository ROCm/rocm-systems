#
#
#   Installation of amd_detail headers
#
#
set(ROCPROFILER_AMD_DETAIL_HEADER_FILES thread_trace.h)

install(
    FILES ${ROCPROFILER_AMD_DETAIL_HEADER_FILES}
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/rocprofiler-sdk/amd_detail/
    COMPONENT development)

add_subdirectory(rocprofiler-sdk-codeobj)
