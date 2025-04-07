#
#
#   Installation of public HSA headers
#
#
set(ROCPROFILER_MARKER_HEADER_FILES api_args.h api_id.h table_id.h)

install(
    FILES ${ROCPROFILER_MARKER_HEADER_FILES}
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/rocprofiler-sdk/marker
    COMPONENT development)
