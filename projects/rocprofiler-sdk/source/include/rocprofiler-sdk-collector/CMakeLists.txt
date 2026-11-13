#
#
#   Installation of public headers
#
#
configure_file(${CMAKE_CURRENT_LIST_DIR}/version.h.in
               ${CMAKE_CURRENT_BINARY_DIR}/version.h @ONLY)

set(ROCPROFILER_COLLECTOR_HEADER_FILES
    # core headers
    collector.h records.h
    # secondary headers
    defines.h types.h ${CMAKE_CURRENT_BINARY_DIR}/version.h)

install(
    FILES ${ROCPROFILER_COLLECTOR_HEADER_FILES}
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/rocprofiler-sdk-collector
    COMPONENT development)
