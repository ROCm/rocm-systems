#
# CMakeLists.txt for aqlprofile core source files
#

target_sources(
    rocprofiler-sdk-aqlprofile
    PRIVATE aql_profile.cpp
            counters.cpp
            gfx10_factory.cpp
            gfx115x_factory.cpp
            gfx11_factory.cpp
            gfx12_factory.cpp
            gfx908_factory.cpp
            gfx90a_factory.cpp
            gfx940_factory.cpp
            gfx9_factory.cpp
            ip_offset_table_init.cpp
            memorymanager.cpp
            navi_reg_init.cpp
            parse_ip_discovery.cpp
            pm4_factory.cpp
            populate_aql.cpp
            spm_data.cpp
            spm_decode.cpp
            spm_v2.cpp
            threadtrace.cpp
            vega20_reg_init.cpp)

if(ROCPROFILER_BUILD_TESTS)
    add_subdirectory(tests)
endif()
