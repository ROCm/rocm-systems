#
#
#   Installation of public HSA headers
#
#
set(ROCPROFILER_HSA_HEADER_FILES api_args.h api_id.h table_api_id.h)

install(
    FILES ${ROCPROFILER_HSA_HEADER_FILES}
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/rocprofiler-sdk/hsa
    COMPONENT development)
