#
#
#   Installation of public headers
#
#
set(ROCPROFILER_INCLUDE_FILES config.h rocprofiler.h rocprofiler_plugin.h)
install(FILES ${ROCPROFILER_INCLUDE_FILES}
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/rocprofiler)
