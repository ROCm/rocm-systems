#
#
# Installation of public KFD headers
#
#
set(ROCPROFILER_KFD_HEADER_FILES kfd_id.h)

install(
    FILES ${ROCPROFILER_KFD_HEADER_FILES}
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/rocprofiler-sdk/kfd
    COMPONENT development)
