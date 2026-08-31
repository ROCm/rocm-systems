#
#
# Installation of public rocSHMEM headers
#
#
set(ROCPROFILER_ROCSHMEM_HEADER_FILES api_args.h api_id.h table_id.h)

install(
    FILES ${ROCPROFILER_ROCSHMEM_HEADER_FILES}
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/rocprofiler-sdk/rocshmem
    COMPONENT development)

add_subdirectory(details)
