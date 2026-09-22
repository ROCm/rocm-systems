#
#
#   Installation of public OMPT headers
#
#
set(ROCPROFILER_OMPT_HEADER_FILES api_args.h api_id.h omp-tools.h)

install(
    FILES ${ROCPROFILER_OMPT_HEADER_FILES}
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/rocprofiler-sdk/ompt
    COMPONENT development)
