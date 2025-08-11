#
#
#   Installation of public headers
#
#
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/version.h.in
               ${CMAKE_CURRENT_BINARY_DIR}/version.h @ONLY)

set(ROCPROFILER_REGISTER_INCLUDE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/rocprofiler-register.h
                                       ${CMAKE_CURRENT_BINARY_DIR}/version.h)

install(
    FILES ${ROCPROFILER_REGISTER_INCLUDE_FILES}
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/rocprofiler-register
    COMPONENT core)
