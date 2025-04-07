#
#
#   Installation of public headers
#
#
configure_file(${CMAKE_CURRENT_LIST_DIR}/version.h.in
               ${CMAKE_CURRENT_BINARY_DIR}/version.h @ONLY)

set(ROCTX_HEADER_FILES
    # core headers
    roctx.h
    # secondary headers
    api_trace.h defines.h types.h ${CMAKE_CURRENT_BINARY_DIR}/version.h)

install(
    FILES ${ROCTX_HEADER_FILES}
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/rocprofiler-sdk-roctx
    COMPONENT roctx)
