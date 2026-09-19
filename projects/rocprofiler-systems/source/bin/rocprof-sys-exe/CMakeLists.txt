# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# ------------------------------------------------------------------------------#
#
# rocprofiler-systems-exe target — unified CLI entry point (`rocsys`)
#
# ------------------------------------------------------------------------------#

add_executable(
    rocprofiler-systems-exe
    ${CMAKE_CURRENT_LIST_DIR}/rocprof-sys.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../common/tool_runner.cpp
)

target_compile_definitions(rocprofiler-systems-exe PRIVATE TIMEMORY_CMAKE=1)
target_include_directories(
    rocprofiler-systems-exe
    PRIVATE ${CMAKE_CURRENT_LIST_DIR} ${CMAKE_CURRENT_LIST_DIR}/..
)
target_link_libraries(
    rocprofiler-systems-exe
    PRIVATE
        rocprofiler-systems::rocprofiler-systems-compile-definitions
        rocprofiler-systems::rocprofiler-systems-headers
        rocprofiler-systems::rocprofiler-systems-common-library
        rocprofiler-systems::rocprofiler-systems-bin-common
        rocprofiler-systems::rocprofiler-systems-core
        rocprofiler-systems::rocprofiler-systems-json
        rocprofiler-systems::rocprofiler-systems-sanitizer
)
set_target_properties(
    rocprofiler-systems-exe
    PROPERTIES
        BUILD_RPATH "\$ORIGIN:\$ORIGIN/../${CMAKE_INSTALL_LIBDIR}"
        INSTALL_RPATH "${ROCPROFSYS_EXE_INSTALL_RPATH}"
        OUTPUT_NAME rocsys
)

rocprofiler_systems_strip_target(rocprofiler-systems-exe)

install(TARGETS rocprofiler-systems-exe DESTINATION ${CMAKE_INSTALL_BINDIR} OPTIONAL)

foreach(
    _dep
    rocprofiler-systems-sample
    rocprofiler-systems-run
    rocprofiler-systems-instrument
    rocprofiler-systems-causal
    rocprofiler-systems-avail
    rocprofiler-systems-attach
)
    if(TARGET ${_dep})
        add_dependencies(rocprofiler-systems-exe ${_dep})
    endif()
endforeach()
