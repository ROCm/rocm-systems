# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

set(containers_headers
    ${CMAKE_CURRENT_LIST_DIR}/aligned_static_vector.hpp
    ${CMAKE_CURRENT_LIST_DIR}/c_array.hpp
    ${CMAKE_CURRENT_LIST_DIR}/operators.hpp
    ${CMAKE_CURRENT_LIST_DIR}/stable_vector.hpp
    ${CMAKE_CURRENT_LIST_DIR}/static_vector.hpp
)

add_library(rocprofiler-systems-containers-library INTERFACE)
add_library(
    rocprofiler-systems::rocprofiler-systems-containers
    ALIAS rocprofiler-systems-containers-library
)

target_sources(rocprofiler-systems-containers-library INTERFACE ${containers_headers})

target_link_libraries(
    rocprofiler-systems-containers-library
    INTERFACE rocprofiler-systems::rocprofiler-systems-headers
)

target_include_directories(
    rocprofiler-systems-containers-library
    INTERFACE ${PROJECT_SOURCE_DIR}/source/lib
)

if(ROCPROFSYS_BUILD_TESTING)
    add_subdirectory(tests)
endif()
