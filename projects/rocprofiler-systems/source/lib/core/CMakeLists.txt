#
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/defines.hpp.in
               ${CMAKE_CURRENT_BINARY_DIR}/defines.hpp @ONLY)

set(core_sources
    ${CMAKE_CURRENT_LIST_DIR}/categories.cpp
    ${CMAKE_CURRENT_LIST_DIR}/config.cpp
    ${CMAKE_CURRENT_LIST_DIR}/constraint.cpp
    ${CMAKE_CURRENT_LIST_DIR}/debug.cpp
    ${CMAKE_CURRENT_LIST_DIR}/dynamic_library.cpp
    ${CMAKE_CURRENT_LIST_DIR}/exception.cpp
    ${CMAKE_CURRENT_LIST_DIR}/gpu.cpp
    ${CMAKE_CURRENT_LIST_DIR}/locking.cpp
    ${CMAKE_CURRENT_LIST_DIR}/mproc.cpp
    ${CMAKE_CURRENT_LIST_DIR}/perfetto.cpp
    ${CMAKE_CURRENT_LIST_DIR}/state.cpp
    ${CMAKE_CURRENT_LIST_DIR}/timemory.cpp)

set(core_headers
    ${CMAKE_CURRENT_LIST_DIR}/categories.hpp
    ${CMAKE_CURRENT_LIST_DIR}/common.hpp
    ${CMAKE_CURRENT_LIST_DIR}/concepts.hpp
    ${CMAKE_CURRENT_LIST_DIR}/config.hpp
    ${CMAKE_CURRENT_LIST_DIR}/constraint.hpp
    ${CMAKE_CURRENT_LIST_DIR}/debug.hpp
    ${CMAKE_CURRENT_LIST_DIR}/dynamic_library.hpp
    ${CMAKE_CURRENT_LIST_DIR}/exception.hpp
    ${CMAKE_CURRENT_LIST_DIR}/gpu.hpp
    ${CMAKE_CURRENT_LIST_DIR}/locking.hpp
    ${CMAKE_CURRENT_LIST_DIR}/mproc.hpp
    ${CMAKE_CURRENT_LIST_DIR}/perfetto.hpp
    ${CMAKE_CURRENT_LIST_DIR}/redirect.hpp
    ${CMAKE_CURRENT_LIST_DIR}/state.hpp
    ${CMAKE_CURRENT_LIST_DIR}/timemory.hpp
    ${CMAKE_CURRENT_LIST_DIR}/utility.hpp)

add_library(omnitrace-core-library STATIC)
add_library(omnitrace::omnitrace-core ALIAS omnitrace-core-library)

target_sources(omnitrace-core-library PRIVATE ${core_sources} ${core_headers}
                                              ${CMAKE_CURRENT_BINARY_DIR}/defines.hpp)
add_subdirectory(binary)
add_subdirectory(components)
add_subdirectory(containers)

target_include_directories(omnitrace-core-library BEFORE
                           PRIVATE ${CMAKE_CURRENT_LIST_DIR})

target_link_libraries(omnitrace-core-library
                      PRIVATE omnitrace::omnitrace-interface-library)

set_target_properties(omnitrace-core-library PROPERTIES OUTPUT_NAME omnitrace-core)
