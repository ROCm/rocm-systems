#
# ATT decoder wrapper library for use by the rocprofv3 tool
#
rocprofiler_activate_clang_tidy()

set(ATT_TOOL_SOURCE_FILES att_lib_wrapper.cpp code.cpp filenames.cpp occupancy.cpp
                          perfcounter.cpp profile_interface.cpp wave.cpp wstates.cpp)

set(ATT_TOOL_HEADER_FILES
    att_lib_wrapper.hpp
    code.hpp
    filenames.hpp
    occupancy.hpp
    outputfile.hpp
    perfcounter.hpp
    profile_interface.hpp
    util.hpp
    wave.hpp
    wstates.hpp)

add_library(rocprofiler-sdk-att-parser STATIC)
add_library(rocprofiler-sdk::rocprofiler-sdk-att-parser ALIAS rocprofiler-sdk-att-parser)
target_sources(rocprofiler-sdk-att-parser PRIVATE ${ATT_TOOL_SOURCE_FILES}
                                                  ${ATT_TOOL_HEADER_FILES})

target_link_libraries(
    rocprofiler-sdk-att-parser
    PRIVATE rocprofiler-sdk::rocprofiler-sdk-shared-library
            rocprofiler-sdk::rocprofiler-sdk-headers
            rocprofiler-sdk::rocprofiler-sdk-json
            rocprofiler-sdk::rocprofiler-sdk-common-library
            rocprofiler-sdk::rocprofiler-sdk-amd-comgr
            rocprofiler-sdk::rocprofiler-sdk-dw
            rocprofiler-sdk::rocprofiler-sdk-elf)

add_subdirectory(waitcnt)
