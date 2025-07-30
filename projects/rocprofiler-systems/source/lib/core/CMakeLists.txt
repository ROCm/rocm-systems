#
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/defines.hpp.in
    ${CMAKE_CURRENT_BINARY_DIR}/defines.hpp
    @ONLY
)

set(core_sources
    ${CMAKE_CURRENT_LIST_DIR}/argparse.cpp
    ${CMAKE_CURRENT_LIST_DIR}/categories.cpp
    ${CMAKE_CURRENT_LIST_DIR}/config.cpp
    ${CMAKE_CURRENT_LIST_DIR}/constraint.cpp
    ${CMAKE_CURRENT_LIST_DIR}/debug.cpp
    ${CMAKE_CURRENT_LIST_DIR}/dynamic_library.cpp
    ${CMAKE_CURRENT_LIST_DIR}/exception.cpp
    ${CMAKE_CURRENT_LIST_DIR}/gpu.cpp
    ${CMAKE_CURRENT_LIST_DIR}/mproc.cpp
    ${CMAKE_CURRENT_LIST_DIR}/perf.cpp
    ${CMAKE_CURRENT_LIST_DIR}/perfetto.cpp
    ${CMAKE_CURRENT_LIST_DIR}/rocprofiler-sdk.cpp
    ${CMAKE_CURRENT_LIST_DIR}/amd_smi.cpp
    ${CMAKE_CURRENT_LIST_DIR}/state.cpp
    ${CMAKE_CURRENT_LIST_DIR}/timemory.cpp
    ${CMAKE_CURRENT_LIST_DIR}/utility.cpp
)

set(core_headers
    ${CMAKE_CURRENT_LIST_DIR}/argparse.hpp
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
    ${CMAKE_CURRENT_LIST_DIR}/mpi.hpp
    ${CMAKE_CURRENT_LIST_DIR}/mproc.hpp
    ${CMAKE_CURRENT_LIST_DIR}/perf.hpp
    ${CMAKE_CURRENT_LIST_DIR}/perfetto.hpp
    ${CMAKE_CURRENT_LIST_DIR}/redirect.hpp
    ${CMAKE_CURRENT_LIST_DIR}/rocprofiler-sdk.hpp
    ${CMAKE_CURRENT_LIST_DIR}/amd_smi.hpp
    ${CMAKE_CURRENT_LIST_DIR}/state.hpp
    ${CMAKE_CURRENT_LIST_DIR}/timemory.hpp
    ${CMAKE_CURRENT_LIST_DIR}/utility.hpp
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
        $<BUILD_INTERFACE:rocprofiler-systems::rocprofiler-systems-timemory>
        $<BUILD_INTERFACE:rocprofiler-systems::rocprofiler-systems-mpi>
        $<BUILD_INTERFACE:rocprofiler-systems::rocprofiler-systems-rocm>
        $<BUILD_INTERFACE:rocprofiler-systems::rocprofiler-systems-static-libgcc-optional>
        $<BUILD_INTERFACE:rocprofiler-systems::rocprofiler-systems-static-libstdcxx-optional>
        $<BUILD_INTERFACE:rocprofiler-systems::rocprofiler-systems-sanitizer>
        $<BUILD_INTERFACE:$<IF:$<BOOL:${ROCPROFSYS_BUILD_LTO}>,rocprofiler-systems::rocprofiler-systems-lto,>>
)

set_target_properties(
    rocprofiler-systems-core-library
    PROPERTIES OUTPUT_NAME ${BINARY_NAME_PREFIX}-core
)
