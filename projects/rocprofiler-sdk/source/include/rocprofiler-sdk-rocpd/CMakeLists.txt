#
#
#   Installation of public headers
#
#
configure_file(${CMAKE_CURRENT_LIST_DIR}/version.h.in
               ${CMAKE_CURRENT_BINARY_DIR}/version.h @ONLY)

set(ROCPD_HEADER_FILES
    # core headers
    rocpd.h
    # secondary headers
    defines.h sql.h types.h ${CMAKE_CURRENT_BINARY_DIR}/version.h)

install(
    FILES ${ROCPD_HEADER_FILES}
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/rocprofiler-sdk-rocpd
    COMPONENT rocpd)
