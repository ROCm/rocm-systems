set(ROCPROFILER_LIB_COUNTERS_SOURCES metrics.cpp)
set(ROCPROFILER_LIB_COUNTERS_HEADERS metrics.hpp)

target_sources(rocprofiler-object-library PRIVATE ${ROCPROFILER_LIB_COUNTERS_SOURCES}
                                                  ${ROCPROFILER_LIB_COUNTERS_HEADERS})

add_subdirectory(xml)

if(ROCPROFILER_BUILD_TESTS)
    add_subdirectory(tests)
endif()
