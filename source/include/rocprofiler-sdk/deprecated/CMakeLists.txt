#
#  headers containing deprecated functions, types, etc.
#
set(ROCPROFILER_DEPRECATED_HEADER_FILES counters.h profile_config.h)

install(
    FILES ${ROCPROFILER_DEPRECATED_HEADER_FILES}
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/rocprofiler-sdk/deprecated
    COMPONENT development)
