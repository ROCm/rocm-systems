#
set(coverage_sources)
set(coverage_headers ${CMAKE_CURRENT_LIST_DIR}/impl.hpp)

target_sources(
    rocprofiler-systems-object-library
    PRIVATE ${coverage_sources} ${coverage_headers}
)
