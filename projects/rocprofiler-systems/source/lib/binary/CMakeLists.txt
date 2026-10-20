# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

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

# Detect system libunwind >= 1.6 which has a regression in _ULx86_64_get_proc_name
# (null-deref via __sigaction). The bundled libunwind (ROCPROFSYS_BUILD_LIBUNWIND=ON)
# is pre-1.6 and unaffected. When the bug is present, analysis.cpp uses dladdr()
# as a crash-safe fallback for name resolution.
if(NOT ROCPROFSYS_BUILD_LIBUNWIND)
    find_package(PkgConfig QUIET)
    if(PkgConfig_FOUND)
        pkg_check_modules(SYSTEM_LIBUNWIND QUIET libunwind)
        if(SYSTEM_LIBUNWIND_FOUND AND SYSTEM_LIBUNWIND_VERSION VERSION_GREATER_EQUAL 1.6)
            target_compile_definitions(
                rocprofiler-systems-binary-library
                PRIVATE ROCPROFSYS_LIBUNWIND_BROKEN_GET_PROC_NAME=1
            )
        endif()
    endif()
endif()
