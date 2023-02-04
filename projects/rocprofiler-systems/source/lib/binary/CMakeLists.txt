#
set(binary_sources
    ${CMAKE_CURRENT_LIST_DIR}/address_multirange.cpp
    ${CMAKE_CURRENT_LIST_DIR}/analysis.cpp
    ${CMAKE_CURRENT_LIST_DIR}/dwarf_entry.cpp
    ${CMAKE_CURRENT_LIST_DIR}/link_map.cpp
    ${CMAKE_CURRENT_LIST_DIR}/scope_filter.cpp
    ${CMAKE_CURRENT_LIST_DIR}/symbol.cpp)

set(binary_headers
    ${CMAKE_CURRENT_LIST_DIR}/address_multirange.hpp
    ${CMAKE_CURRENT_LIST_DIR}/analysis.hpp
    ${CMAKE_CURRENT_LIST_DIR}/dwarf_entry.hpp
    ${CMAKE_CURRENT_LIST_DIR}/binary_info.hpp
    ${CMAKE_CURRENT_LIST_DIR}/link_map.hpp
    ${CMAKE_CURRENT_LIST_DIR}/scope_filter.hpp
    ${CMAKE_CURRENT_LIST_DIR}/symbol.hpp)

add_library(omnitrace-binary-library STATIC)
add_library(omnitrace::omnitrace-binary ALIAS omnitrace-binary-library)

target_sources(omnitrace-binary-library PRIVATE ${binary_sources} ${binary_headers})

target_link_libraries(
    omnitrace-binary-library PRIVATE omnitrace::omnitrace-interface-library
                                     omnitrace::omnitrace-core)

set_target_properties(omnitrace-binary-library PROPERTIES OUTPUT_NAME omnitrace-binary)
