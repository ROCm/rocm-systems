# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

check_rocprofsys_disable_examples("openmp-cg" _OPENMP_CG_DISABLED)
if(_OPENMP_CG_DISABLED)
    return()
endif()

include(${CMAKE_CURRENT_LIST_DIR}/../openmp-helper.cmake)

add_executable(openmp-cg cg.cpp)
target_link_libraries(openmp-cg PRIVATE openmp-common)

rocprofiler_systems_custom_compilation(
    COMPILER ${amdclangpp_EXECUTABLE}
    TARGET openmp-cg
)

if(ROCPROFSYS_INSTALL_EXAMPLES AND TARGET openmp-cg)
    install(
        TARGETS openmp-cg
        DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/rocprofiler-systems/examples
        COMPONENT rocprofiler-systems-examples
    )
endif()
