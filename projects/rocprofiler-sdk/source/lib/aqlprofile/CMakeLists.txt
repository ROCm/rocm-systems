#
# Build dynamic Library object
#
# set ( TARGET_LIB "${TARGET_NAME}" )
# set ( LIB_SRC
# ${LIB_DIR}/core/aql_profile.cpp
# ${LIB_DIR}/core/counters.cpp
# ${LIB_DIR}/core/threadtrace.cpp
# ${LIB_DIR}/core/spm_data.cpp
# ${LIB_DIR}/core/spm_decode.cpp
# ${LIB_DIR}/core/spm_v2.cpp
# ${LIB_DIR}/core/populate_aql.cpp
# ${LIB_DIR}/core/memorymanager.cpp
# ${LIB_DIR}/core/pm4_factory.cpp
# ${LIB_DIR}/core/gfx9_factory.cpp
# ${LIB_DIR}/core/gfx908_factory.cpp
# ${LIB_DIR}/core/gfx90a_factory.cpp
# ${LIB_DIR}/core/gfx940_factory.cpp
# ${LIB_DIR}/core/gfx10_factory.cpp
# ${LIB_DIR}/core/gfx11_factory.cpp
# ${LIB_DIR}/core/gfx115x_factory.cpp
# ${LIB_DIR}/core/gfx12_factory.cpp
# ${LIB_DIR}/core/vega20_reg_init.cpp
# ${LIB_DIR}/core/parse_ip_discovery.cpp
# ${LIB_DIR}/core/navi_reg_init.cpp
# ${LIB_DIR}/core/ip_offset_table_init.cpp
# ${LIB_DIR}/util/hsa_rsrc_factory.cpp
# )

configure_file(${CMAKE_CURRENT_LIST_DIR}/version.h.in
               ${CMAKE_CURRENT_BINARY_DIR}/version.h @ONLY)

add_library(rocprofiler-sdk-aqlprofile STATIC)
add_library(rocprofiler-sdk::rocprofiler-sdk-aqlprofile ALIAS rocprofiler-sdk-aqlprofile)
target_link_libraries(
    rocprofiler-sdk-aqlprofile
    PUBLIC $<BUILD_INTERFACE:rocprofiler-sdk::rocprofiler-sdk-headers>
           # $<BUILD_INTERFACE:rocprofiler-sdk::rocprofiler-sdk-build-flags>
           $<BUILD_INTERFACE:rocprofiler-sdk::rocprofiler-sdk-threading>
           $<BUILD_INTERFACE:rocprofiler-sdk::rocprofiler-sdk-memcheck>
           $<BUILD_INTERFACE:rocprofiler-sdk::rocprofiler-sdk-cxx-filesystem>
           $<BUILD_INTERFACE:rocprofiler-sdk::rocprofiler-sdk-glog>
           $<BUILD_INTERFACE:rocprofiler-sdk::rocprofiler-sdk-fmt>
           $<BUILD_INTERFACE:rocprofiler-sdk::rocprofiler-sdk-dl>
           $<BUILD_INTERFACE:rocprofiler-sdk::rocprofiler-sdk-atomic>
           $<BUILD_INTERFACE:rocprofiler-sdk::rocprofiler-sdk-hsakmt-nolink>
           $<BUILD_INTERFACE:rocprofiler-sdk::rocprofiler-sdk-hsa-runtime-nolink>)
set_target_properties(rocprofiler-sdk-aqlprofile PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_subdirectory(core)
add_subdirectory(util)
add_subdirectory(pm4)
