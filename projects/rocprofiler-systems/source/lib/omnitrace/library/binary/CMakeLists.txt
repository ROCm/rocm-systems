#
set(binary_sources
    ${CMAKE_CURRENT_LIST_DIR}/address_multirange.cpp
    ${CMAKE_CURRENT_LIST_DIR}/address_range.cpp
    ${CMAKE_CURRENT_LIST_DIR}/analysis.cpp
    ${CMAKE_CURRENT_LIST_DIR}/dwarf_entry.cpp
    ${CMAKE_CURRENT_LIST_DIR}/link_map.cpp
    ${CMAKE_CURRENT_LIST_DIR}/scope_filter.cpp
    ${CMAKE_CURRENT_LIST_DIR}/symbol.cpp)

set(binary_headers
    ${CMAKE_CURRENT_LIST_DIR}/address_multirange.hpp
    ${CMAKE_CURRENT_LIST_DIR}/address_range.hpp
    ${CMAKE_CURRENT_LIST_DIR}/analysis.hpp
    ${CMAKE_CURRENT_LIST_DIR}/dwarf_entry.hpp
    ${CMAKE_CURRENT_LIST_DIR}/fwd.hpp
    ${CMAKE_CURRENT_LIST_DIR}/binary_info.hpp
    ${CMAKE_CURRENT_LIST_DIR}/link_map.hpp
    ${CMAKE_CURRENT_LIST_DIR}/scope_filter.hpp
    ${CMAKE_CURRENT_LIST_DIR}/symbol.hpp)

target_sources(omnitrace-object-library PRIVATE ${binary_sources} ${binary_headers})
