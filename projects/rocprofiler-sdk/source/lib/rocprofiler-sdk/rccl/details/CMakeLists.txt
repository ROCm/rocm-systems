#
#
#
set(ROCPROFILER_LIB_RCCL_DETAILS_SOURCES)
set(ROCPROFILER_LIB_RCCL_DETAILS_HEADERS fmt.hpp)

target_sources(
    rocprofiler-sdk-object-library PRIVATE ${ROCPROFILER_LIB_RCCL_DETAILS_SOURCES}
                                           ${ROCPROFILER_LIB_RCCL_DETAILS_HEADERS})
