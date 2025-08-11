#
#
#
set(ROCPROFILER_LIB_OMPT_DETAILS_SOURCES)
set(ROCPROFILER_LIB_OMPT_DETAILS_HEADERS format.hpp)

target_sources(
    rocprofiler-sdk-object-library PRIVATE ${ROCPROFILER_LIB_OMPT_DETAILS_SOURCES}
                                           ${ROCPROFILER_LIB_OMPT_DETAILS_HEADERS})
