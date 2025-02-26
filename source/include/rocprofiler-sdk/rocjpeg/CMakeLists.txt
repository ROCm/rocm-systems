#
#
# Installation of public rocJPEG headers
#
#
set(ROCPROFILER_ROCJPEG_HEADER_FILES api_args.h api_id.h table_id.h)

install(
    FILES ${ROCPROFILER_ROCJPEG_HEADER_FILES}
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/rocprofiler-sdk/rocjpeg
    COMPONENT development)

add_subdirectory(details)
