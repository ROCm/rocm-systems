#
#
#   Installation of public headers
#
#
configure_file(${CMAKE_CURRENT_LIST_DIR}/version.h.in
               ${CMAKE_CURRENT_BINARY_DIR}/version.h @ONLY)

set(ROCPROFILER_HEADER_FILES config.h defines.h hip.h hsa.h marker.h rocprofiler.h
                             rocprofiler_plugin.h ${CMAKE_CURRENT_BINARY_DIR}/version.h)

install(FILES ${ROCPROFILER_HEADER_FILES}
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/rocprofiler)

add_subdirectory(hip)
add_subdirectory(hsa)
add_subdirectory(marker)
