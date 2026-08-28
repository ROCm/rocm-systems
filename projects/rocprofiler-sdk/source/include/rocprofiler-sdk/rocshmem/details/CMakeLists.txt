#
#
# Installation of vendored rocSHMEM dispatch table header
#
#
set(ROCPROFILER_ROCSHMEM_DETAILS_HEADER_FILES api_trace.h)

install(
    FILES ${ROCPROFILER_ROCSHMEM_DETAILS_HEADER_FILES}
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/rocprofiler-sdk/rocshmem/details
    COMPONENT development)
