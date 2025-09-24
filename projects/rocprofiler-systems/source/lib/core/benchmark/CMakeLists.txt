set(benchmark_sources ${CMAKE_CURRENT_LIST_DIR}/benchmark.cpp)

set(benchmark_headers
    ${CMAKE_CURRENT_LIST_DIR}/benchmark.hpp
    ${CMAKE_CURRENT_LIST_DIR}/category.hpp
)

target_sources(
    rocprofiler-systems-core-library
    PRIVATE ${benchmark_sources} ${benchmark_headers}
)
