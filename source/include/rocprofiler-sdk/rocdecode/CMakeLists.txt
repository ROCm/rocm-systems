#
#
# Installation of public rocDecode headers
#
#
set(ROCPROFILER_ROCDECODE_HEADER_FILES api_args.h api_id.h api_trace.h table_id.h)

install(
    FILES ${ROCPROFILER_ROCDECODE_HEADER_FILES}
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/rocprofiler-sdk/rocdecode
    COMPONENT development)

add_subdirectory(details)
