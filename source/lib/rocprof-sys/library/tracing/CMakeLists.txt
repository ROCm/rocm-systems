#
set(tracing_sources ${CMAKE_CURRENT_LIST_DIR}/annotation.cpp)
set(tracing_headers ${CMAKE_CURRENT_LIST_DIR}/annotation.hpp)

target_sources(
    rocprofiler-systems-object-library
    PRIVATE ${tracing_sources} ${tracing_headers}
)
