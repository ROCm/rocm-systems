#
#
#
set(ROCPROFILER_LIB_HIP_DETAILS_SOURCES)
set(ROCPROFILER_LIB_HIP_DETAILS_HEADERS format.hpp ostream.hpp)

target_sources(rocprofiler-object-library PRIVATE ${ROCPROFILER_LIB_HIP_DETAILS_SOURCES}
                                                  ${ROCPROFILER_LIB_HIP_DETAILS_HEADERS})
