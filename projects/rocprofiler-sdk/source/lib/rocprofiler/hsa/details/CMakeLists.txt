#
#
#
set(ROCPROFILER_LIB_HSA_DETAILS_SOURCES)
set(ROCPROFILER_LIB_HSA_DETAILS_HEADERS ostream.hpp)

target_sources(rocprofiler-library PRIVATE ${ROCPROFILER_LIB_HSA_DETAILS_SOURCES}
                                           ${ROCPROFILER_LIB_HSA_DETAILS_HEADERS})
