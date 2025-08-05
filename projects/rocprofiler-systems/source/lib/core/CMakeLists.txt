# MIT License
#
# Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

#
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/defines.hpp.in
    ${CMAKE_CURRENT_BINARY_DIR}/defines.hpp
    @ONLY
)

set(core_sources
    ${CMAKE_CURRENT_LIST_DIR}/agent_manager.cpp
    ${CMAKE_CURRENT_LIST_DIR}/amd_smi.cpp
    ${CMAKE_CURRENT_LIST_DIR}/argparse.cpp
    ${CMAKE_CURRENT_LIST_DIR}/categories.cpp
    ${CMAKE_CURRENT_LIST_DIR}/config.cpp
    ${CMAKE_CURRENT_LIST_DIR}/constraint.cpp
    ${CMAKE_CURRENT_LIST_DIR}/cpu.cpp
    ${CMAKE_CURRENT_LIST_DIR}/debug.cpp
    ${CMAKE_CURRENT_LIST_DIR}/dynamic_library.cpp
    ${CMAKE_CURRENT_LIST_DIR}/exception.cpp
    ${CMAKE_CURRENT_LIST_DIR}/gpu.cpp
    ${CMAKE_CURRENT_LIST_DIR}/mproc.cpp
    ${CMAKE_CURRENT_LIST_DIR}/node_info.cpp
    ${CMAKE_CURRENT_LIST_DIR}/perf.cpp
    ${CMAKE_CURRENT_LIST_DIR}/perfetto.cpp
    ${CMAKE_CURRENT_LIST_DIR}/rocprofiler-sdk.cpp
    ${CMAKE_CURRENT_LIST_DIR}/state.cpp
    ${CMAKE_CURRENT_LIST_DIR}/timemory.cpp
    ${CMAKE_CURRENT_LIST_DIR}/utility.cpp
    ${CMAKE_CURRENT_LIST_DIR}/agent_manager.cpp
    ${CMAKE_CURRENT_LIST_DIR}/node_info.cpp
)

set(core_headers
    ${CMAKE_CURRENT_LIST_DIR}/agent.hpp
    ${CMAKE_CURRENT_LIST_DIR}/agent_manager.hpp
    ${CMAKE_CURRENT_LIST_DIR}/amd_smi.hpp
    ${CMAKE_CURRENT_LIST_DIR}/argparse.hpp
    ${CMAKE_CURRENT_LIST_DIR}/categories.hpp
    ${CMAKE_CURRENT_LIST_DIR}/common.hpp
    ${CMAKE_CURRENT_LIST_DIR}/concepts.hpp
    ${CMAKE_CURRENT_LIST_DIR}/config.hpp
    ${CMAKE_CURRENT_LIST_DIR}/constraint.hpp
    ${CMAKE_CURRENT_LIST_DIR}/cpu.hpp
    ${CMAKE_CURRENT_LIST_DIR}/debug.hpp
    ${CMAKE_CURRENT_LIST_DIR}/dynamic_library.hpp
    ${CMAKE_CURRENT_LIST_DIR}/exception.hpp
    ${CMAKE_CURRENT_LIST_DIR}/gpu.hpp
    ${CMAKE_CURRENT_LIST_DIR}/locking.hpp
    ${CMAKE_CURRENT_LIST_DIR}/mpi.hpp
    ${CMAKE_CURRENT_LIST_DIR}/mproc.hpp
    ${CMAKE_CURRENT_LIST_DIR}/node_info.hpp
    ${CMAKE_CURRENT_LIST_DIR}/perf.hpp
    ${CMAKE_CURRENT_LIST_DIR}/perfetto.hpp
    ${CMAKE_CURRENT_LIST_DIR}/redirect.hpp
    ${CMAKE_CURRENT_LIST_DIR}/rocprofiler-sdk.hpp
    ${CMAKE_CURRENT_LIST_DIR}/state.hpp
    ${CMAKE_CURRENT_LIST_DIR}/timemory.hpp
    ${CMAKE_CURRENT_LIST_DIR}/utility.hpp
    ${CMAKE_CURRENT_LIST_DIR}/agent.hpp
    ${CMAKE_CURRENT_LIST_DIR}/agent_manager.hpp
    ${CMAKE_CURRENT_LIST_DIR}/node_info.hpp
)

add_library(rocprofiler-systems-core-library STATIC)
add_library(
    rocprofiler-systems::rocprofiler-systems-core
    ALIAS rocprofiler-systems-core-library
)

target_sources(
    rocprofiler-systems-core-library
    PRIVATE ${core_sources} ${core_headers} ${CMAKE_CURRENT_BINARY_DIR}/defines.hpp
)
add_subdirectory(binary)
add_subdirectory(components)
add_subdirectory(containers)
add_subdirectory(rocpd)
add_subdirectory(trace_cache)

target_include_directories(
    rocprofiler-systems-core-library
    BEFORE
    PRIVATE ${CMAKE_CURRENT_LIST_DIR}
)

target_include_directories(
    rocprofiler-systems-core-library
    PRIVATE ${PROJECT_SOURCE_DIR}/external/timemory/source/timemory/tpls/cereal
)

target_link_libraries(
    rocprofiler-systems-core-library
    PRIVATE rocprofiler-systems::rocprofiler-systems-interface-library
)
target_link_libraries(
    rocprofiler-systems-core-library
    PRIVATE
        $<BUILD_INTERFACE:rocprofiler-systems::rocprofiler-systems-headers>
        $<BUILD_INTERFACE:rocprofiler-systems::rocprofiler-systems-threading>
        $<BUILD_INTERFACE:rocprofiler-systems::rocprofiler-systems-common-library>
        $<BUILD_INTERFACE:rocprofiler-systems::rocprofiler-systems-compile-definitions>
        $<BUILD_INTERFACE:rocprofiler-systems::rocprofiler-systems-compile-options>
        $<BUILD_INTERFACE:rocprofiler-systems::rocprofiler-systems-perfetto>
        $<BUILD_INTERFACE:rocprofiler-systems::rocprofiler-systems-sqlite3>
        $<BUILD_INTERFACE:rocprofiler-systems::rocprofiler-systems-timemory>
        $<BUILD_INTERFACE:rocprofiler-systems::rocprofiler-systems-mpi>
        $<BUILD_INTERFACE:rocprofiler-systems::rocprofiler-systems-rocm>
        $<BUILD_INTERFACE:rocprofiler-systems::rocprofiler-systems-static-libgcc-optional>
        $<BUILD_INTERFACE:rocprofiler-systems::rocprofiler-systems-static-libstdcxx-optional>
        $<BUILD_INTERFACE:rocprofiler-systems::rocprofiler-systems-sanitizer>
        $<BUILD_INTERFACE:$<IF:$<BOOL:${ROCPROFSYS_BUILD_LTO}>,rocprofiler-systems::rocprofiler-systems-lto,>>
)

file(GLOB ROCPD_SCHEMA_FILES "${CMAKE_CURRENT_LIST_DIR}/rocpd/data_storage/schema/*.sql")

foreach(_SRC ${ROCPD_SCHEMA_FILES})
    cmake_path(GET _SRC FILENAME _BASE)
    configure_file(
        ${_SRC}
        ${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_NAME}/${_BASE}
        COPYONLY
    )
    install(
        FILES ${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_NAME}/${_BASE}
        DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_NAME}
        COMPONENT core
    )
endforeach()

set_target_properties(
    rocprofiler-systems-core-library
    PROPERTIES OUTPUT_NAME ${BINARY_NAME_PREFIX}-core
)
