#
#
#
set(ROCPROFILER_LIB_ROCPD_DETAILS_SOURCES)
set(ROCPROFILER_LIB_ROCPD_DETAILS_HEADERS format.hpp)

target_sources(
    rocprofiler-sdk-rocpd-shared-library PRIVATE ${ROCPROFILER_LIB_HIP_DETAILS_SOURCES}
                                                 ${ROCPROFILER_LIB_HIP_DETAILS_HEADERS})
