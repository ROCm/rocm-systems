#
#
set(ROCPROFILER_LIB_MARKER_SOURCES marker.cpp)
set(ROCPROFILER_LIB_MARKER_HEADERS defines.hpp marker.hpp utils.hpp)

target_sources(rocprofiler-object-library PRIVATE ${ROCPROFILER_LIB_MARKER_SOURCES}
                                                  ${ROCPROFILER_LIB_MARKER_HEADERS})
