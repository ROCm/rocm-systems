#
#
# Installation of public RCCL headers
#
#
set(ROCPROFILER_RCCL_DETAILS_HEADER_FILES api_trace.h rccl.h)

install(
    FILES ${ROCPROFILER_RCCL_DETAILS_HEADER_FILES}
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/rocprofiler-sdk/rccl/details
    COMPONENT development)
