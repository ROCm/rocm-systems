#
set(binary_sources
    ${CMAKE_CURRENT_LIST_DIR}/address_multirange.cpp
    ${CMAKE_CURRENT_LIST_DIR}/analysis.cpp
    ${CMAKE_CURRENT_LIST_DIR}/dwarf_entry.cpp
    ${CMAKE_CURRENT_LIST_DIR}/link_map.cpp
    ${CMAKE_CURRENT_LIST_DIR}/scope_filter.cpp
    ${CMAKE_CURRENT_LIST_DIR}/symbol.cpp
)

set(binary_headers
    ${CMAKE_CURRENT_LIST_DIR}/address_multirange.hpp
    ${CMAKE_CURRENT_LIST_DIR}/analysis.hpp
    ${CMAKE_CURRENT_LIST_DIR}/dwarf_entry.hpp
    ${CMAKE_CURRENT_LIST_DIR}/binary_info.hpp
    ${CMAKE_CURRENT_LIST_DIR}/link_map.hpp
    ${CMAKE_CURRENT_LIST_DIR}/scope_filter.hpp
    ${CMAKE_CURRENT_LIST_DIR}/symbol.hpp
)

add_library(rocprofiler-systems-binary-library STATIC)
add_library(
    rocprofiler-systems::rocprofiler-systems-binary
    ALIAS rocprofiler-systems-binary-library
)

target_sources(
    rocprofiler-systems-binary-library
    PRIVATE ${binary_sources} ${binary_headers}
)

target_link_libraries(
    rocprofiler-systems-binary-library
    PRIVATE
        rocprofiler-systems::rocprofiler-systems-interface-library
        rocprofiler-systems::rocprofiler-systems-core
)

set_target_properties(
    rocprofiler-systems-binary-library
    PROPERTIES OUTPUT_NAME ${BINARY_NAME_PREFIX}-binary
)
