#
# ATT decoder wrapper library for use by the rocprofv3 tool
#
set(ATT_TOOL_SOURCE_FILES
    waitcnt/analysis.cpp
    waitcnt/gfx9.cpp
    waitcnt/gfx10.cpp
    waitcnt/gfx12.cpp
    att_lib_wrapper.cpp
    wave.cpp
    code.cpp
    filenames.cpp
    occupancy.cpp
    wstates.cpp
    profile_interface.cpp
    dl.cpp)

find_package(
    amd_comgr
    REQUIRED
    CONFIG
    HINTS
    ${rocm_version_DIR}
    ${ROCM_PATH}
    PATHS
    ${rocm_version_DIR}
    ${ROCM_PATH}
    PATH_SUFFIXES
    lib/cmake/amd_comgr)

add_library(rocprofiler-sdk-att-parser STATIC)
add_library(rocprofiler-sdk::rocprofiler-sdk-att-parser ALIAS rocprofiler-sdk-att-parser)
target_sources(rocprofiler-sdk-att-parser PRIVATE ${ATT_TOOL_SOURCE_FILES})

target_link_libraries(
    rocprofiler-sdk-att-parser
    PRIVATE rocprofiler-sdk::rocprofiler-sdk-codeobj
            rocprofiler-sdk::rocprofiler-sdk-headers
            rocprofiler-sdk::rocprofiler-sdk-json
            rocprofiler-sdk::rocprofiler-sdk-common-library
            amd_comgr
            dw)

add_subdirectory(waitcnt)

if(ROCPROFILER_BUILD_TESTS)
    add_subdirectory(tests)
endif()
