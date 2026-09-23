#
#
if(WIN32)
    # roctx is not built on Windows, so its API table is never captured.
    set(ROCPROFILER_LIB_MARKER_SOURCES windows_stubs.cpp)
else()
    set(ROCPROFILER_LIB_MARKER_SOURCES marker.cpp range_marker.cpp)
endif()
set(ROCPROFILER_LIB_MARKER_HEADERS defines.hpp marker.hpp utils.hpp)

target_sources(rocprofiler-sdk-object-library PRIVATE ${ROCPROFILER_LIB_MARKER_SOURCES}
                                                      ${ROCPROFILER_LIB_MARKER_HEADERS})
