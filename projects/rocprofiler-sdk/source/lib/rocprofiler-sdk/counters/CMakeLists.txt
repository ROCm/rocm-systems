set(ROCPROFILER_LIB_COUNTERS_SOURCES
    metrics.cpp dimensions.cpp evaluate_ast.cpp core.cpp id_decode.cpp
    dispatch_handlers.cpp controller.cpp device_counting.cpp)
set(ROCPROFILER_LIB_COUNTERS_HEADERS
    metrics.hpp dimensions.hpp evaluate_ast.hpp core.hpp id_decode.hpp
    dispatch_handlers.hpp controller.hpp device_counting.hpp)
target_sources(rocprofiler-sdk-object-library PRIVATE ${ROCPROFILER_LIB_COUNTERS_SOURCES}
                                                      ${ROCPROFILER_LIB_COUNTERS_HEADERS})

add_subdirectory(xml)
add_subdirectory(parser)
add_subdirectory(yaml)

if(ROCPROFILER_BUILD_TESTS)
    add_subdirectory(tests)
endif()
