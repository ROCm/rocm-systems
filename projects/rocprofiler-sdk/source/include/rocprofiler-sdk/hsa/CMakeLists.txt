#
#
# Installation of public HSA headers
#
#
set(ROCPROFILER_HSA_HEADER_FILES
    amd_ext_api_id.h
    api_args.h
    api_id.h
    api_trace_version.h
    core_api_id.h
    finalize_ext_api_id.h
    image_ext_api_id.h
    scratch_memory_args.h
    scratch_memory_id.h
    table_id.h)

install(
    FILES ${ROCPROFILER_HSA_HEADER_FILES}
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/rocprofiler-sdk/hsa
    COMPONENT development)
