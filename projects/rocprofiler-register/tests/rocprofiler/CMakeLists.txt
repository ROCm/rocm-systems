#
#   mock rocprofiler library
#

find_package(rocprofiler-register REQUIRED)
if(TARGET rocprofiler-register::rocprofiler-register-headers)
    get_property(
        rocp_reg_INCLUDE_DIR
        TARGET rocprofiler-register::rocprofiler-register-headers
        PROPERTY INTERFACE_INCLUDE_DIRECTORIES)
else()
    get_property(
        rocp_reg_INCLUDE_DIR
        TARGET rocprofiler-register::rocprofiler-register
        PROPERTY INTERFACE_INCLUDE_DIRECTORIES)
endif()

add_library(rocprofiler SHARED)
add_library(rocprofiler::rocprofiler ALIAS rocprofiler)
target_sources(rocprofiler PRIVATE rocprofiler.cpp)
target_include_directories(
    rocprofiler
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}>
           $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>
           $<BUILD_INTERFACE:${rocp_reg_INCLUDE_DIR}>)
set_target_properties(rocprofiler PROPERTIES OUTPUT_NAME rocprofiler-sdk)
# WINDOWS-DIVERGENCE: SOVERSION on Windows produces rocprofiler-sdk-0.dll
# which does not match what rocprofiler_register's rocp_load_rocprofiler_lib
# searches for ("rocprofiler-sdk.dll"). The Linux side keeps SOVERSION 0 to
# match librocprofiler-sdk.so.0.
if(NOT WIN32)
    set_target_properties(rocprofiler PROPERTIES SOVERSION 0 VERSION 0.0.0)
endif()
rocp_register_strip_target(rocprofiler)
