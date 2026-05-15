#
set(containers_sources)

set(containers_headers
    ${CMAKE_CURRENT_LIST_DIR}/aligned_static_vector.hpp
    ${CMAKE_CURRENT_LIST_DIR}/c_array.hpp
    ${CMAKE_CURRENT_LIST_DIR}/operators.hpp
    ${CMAKE_CURRENT_LIST_DIR}/stable_vector.hpp
    ${CMAKE_CURRENT_LIST_DIR}/static_vector.hpp
)

target_sources(
    rocprofiler-systems-core-library
    PRIVATE ${containers_sources} ${containers_headers}
)
