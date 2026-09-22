#
#
# Installation of public RCCL headers
#
#
set(ROCPROFILER_RCCL_HEADER_FILES api_args.h api_id.h table_id.h)

install(
    FILES ${ROCPROFILER_RCCL_HEADER_FILES}
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/rocprofiler-sdk/rccl
    COMPONENT development)

add_subdirectory(details)
