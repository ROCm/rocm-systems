#
#
#   Installation of public HIP headers
#
#
set(ROCPROFILER_HIP_HEADER_FILES api_args.h api_id.h compiler_api_id.h runtime_api_id.h
                                 table_id.h)

install(
    FILES ${ROCPROFILER_HIP_HEADER_FILES}
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/rocprofiler-sdk/hip
    COMPONENT development)
