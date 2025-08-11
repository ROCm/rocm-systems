#
#
#

if(NOT TARGET rocprofiler-register::rocprofiler-register)
    find_package(rocprofiler-register REQUIRED)
endif()

add_library(amdhip SHARED)
add_library(amdhip::amdhip ALIAS amdhip)
target_sources(amdhip PRIVATE amdhip.cpp amdhip.hpp)
target_include_directories(amdhip PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}>)
target_link_libraries(
    amdhip
    PRIVATE rocprofiler-register::rocprofiler-register
    PUBLIC hsa-runtime::hsa-runtime)
set_target_properties(
    amdhip
    PROPERTIES OUTPUT_NAME amdhip64
               SOVERSION 6
               VERSION 6.0.1)
rocp_register_strip_target(amdhip)

add_library(amdhip-invalid SHARED)
add_library(amdhip::amdhip-invalid ALIAS amdhip-invalid)
target_sources(amdhip-invalid PRIVATE amdhip.cpp amdhip.hpp)
target_include_directories(amdhip-invalid
                           PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}>)
target_link_libraries(
    amdhip-invalid
    PRIVATE rocprofiler-register::rocprofiler-register
    PUBLIC hsa-runtime::hsa-runtime-invalid)
set_target_properties(
    amdhip-invalid
    PROPERTIES OUTPUT_NAME amdhip64
               SOVERSION 6
               VERSION 6.0.1
               LIBRARY_OUTPUT_DIRECTORY "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/invalid")
rocp_register_strip_target(amdhip-invalid)
