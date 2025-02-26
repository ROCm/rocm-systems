#
#
# Installation of public rocDecode headers
#
#
set(ROCPROFILER_ROCDECODE_DETAILS_HEADER_FILES
    rocdecode_api_trace.h rocdecode.h rocparser.h rocdecode_version.h
    roc_bitstream_reader.h rocdecode_headers.h)

install(
    FILES ${ROCPROFILER_ROCDECODE_DETAILS_HEADER_FILES}
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/rocprofiler-sdk/rocdecode/details
    COMPONENT development)
