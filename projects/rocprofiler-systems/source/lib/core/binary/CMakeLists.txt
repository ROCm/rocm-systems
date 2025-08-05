#
set(binary_sources ${CMAKE_CURRENT_LIST_DIR}/address_range.cpp)

set(binary_headers
    ${CMAKE_CURRENT_LIST_DIR}/address_range.hpp
    ${CMAKE_CURRENT_LIST_DIR}/fwd.hpp
)

target_sources(
    rocprofiler-systems-core-library
    PRIVATE ${binary_sources} ${binary_headers}
)
