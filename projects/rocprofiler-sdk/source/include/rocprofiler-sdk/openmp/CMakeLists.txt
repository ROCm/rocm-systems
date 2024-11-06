#
#
#   Installation of public OpenMP headers
#
#
set(ROCPROFILER_OPENMP_HEADER_FILES api_args.h api_id.h omp-tools.h)

install(
    FILES ${ROCPROFILER_OPENMP_HEADER_FILES}
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/rocprofiler-sdk/openmp
    COMPONENT development)
