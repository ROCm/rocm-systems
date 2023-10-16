set(ROCPROFILER_LIB_COUNTERS_SOURCES metrics.cpp evaluate_ast.cpp core.cpp)
set(ROCPROFILER_LIB_COUNTERS_HEADERS metrics.hpp evaluate_ast.hpp core.hpp)
target_sources(rocprofiler-object-library PRIVATE ${ROCPROFILER_LIB_COUNTERS_SOURCES}
                                                  ${ROCPROFILER_LIB_COUNTERS_HEADERS})

add_subdirectory(xml)
add_subdirectory(parser)

if(ROCPROFILER_BUILD_TESTS)
    add_subdirectory(tests)
endif()
