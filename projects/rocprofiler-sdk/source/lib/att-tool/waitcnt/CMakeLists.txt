#
# ATT waitcnt ISA
#
set(ATT_TOOL_WAITCNT_SOURCE_FILES analysis.cpp gfx9.cpp gfx10.cpp gfx12.cpp)

set(ATT_TOOL_WAITCNT_HEADER_FILES analysis.hpp)

target_sources(rocprofiler-sdk-att-parser PRIVATE ${ATT_TOOL_WAITCNT_SOURCE_FILES}
                                                  ${ATT_TOOL_WAITCNT_HEADER_FILES})

if(ROCPROFILER_BUILD_TESTS)
    add_subdirectory(tests)
endif()
