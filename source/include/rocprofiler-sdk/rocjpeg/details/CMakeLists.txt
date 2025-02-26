#
#
# Installation of public rocJPEG headers
#
#
set(ROCPROFILER_ROCJPEG_DETAILS_HEADER_FILES rocjpeg_api_trace.h rocjpeg.h
                                             rocjpeg_version.h rocjpeg_headers.h)

install(
    FILES ${ROCPROFILER_ROCJPEG_DETAILS_HEADER_FILES}
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/rocprofiler-sdk/rocjpeg/details
    COMPONENT development)
