#
#
#   Installation of public headers
#
#
configure_file(${CMAKE_CURRENT_LIST_DIR}/version.h.in
               ${CMAKE_CURRENT_BINARY_DIR}/version.h @ONLY)

set(ROCATTACH_HEADER_FILES
    # core headers
    rocattach.h
    # secondary headers
    defines.h types.h ${CMAKE_CURRENT_BINARY_DIR}/version.h)

install(
    FILES ${ROCATTACH_HEADER_FILES}
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/rocprofiler-sdk-rocattach
    COMPONENT rocattach)
