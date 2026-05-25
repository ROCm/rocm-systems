# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

if(ROCPROFSYS_DISABLE_EXAMPLES)
    if("openmp-lu" IN_LIST ROCPROFSYS_DISABLE_EXAMPLES)
        return()
    endif()
endif()

include(${CMAKE_CURRENT_LIST_DIR}/../openmp-helper.cmake)

add_executable(openmp-lu lu.cpp)
target_link_libraries(openmp-lu PRIVATE openmp-common)

rocprofiler_systems_custom_compilation(
    COMPILER ${amdclangpp_EXECUTABLE}
    TARGET openmp-lu
)

if(ROCPROFSYS_INSTALL_EXAMPLES AND TARGET openmp-lu)
    install(
        TARGETS openmp-lu
        DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/rocprofiler-systems/examples
        COMPONENT rocprofiler-systems-examples
    )
endif()
