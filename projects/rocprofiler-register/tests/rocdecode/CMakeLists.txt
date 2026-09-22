#
#
#

if(NOT TARGET rocprofiler-register::rocprofiler-register)
    # find_package(rocprofiler-register REQUIRED)
endif()

add_library(rocdecode SHARED)
add_library(rocdecode::rocdecode ALIAS rocdecode)
target_sources(rocdecode PRIVATE rocdecode.cpp rocdecode.hpp)
target_include_directories(rocdecode PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}>)
target_link_libraries(rocdecode PRIVATE rocprofiler-register::rocprofiler-register)
set_target_properties(
    rocdecode
    PROPERTIES OUTPUT_NAME rocdecode
               SOVERSION 1
               VERSION 1.0)
rocp_register_strip_target(rocdecode)

add_library(rocdecode-invalid SHARED)
add_library(rocdecode::rocdecode-invalid ALIAS rocdecode-invalid)
target_sources(rocdecode-invalid PRIVATE rocdecode.cpp rocdecode.hpp)
target_include_directories(rocdecode-invalid
                           PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}>)
target_link_libraries(rocdecode-invalid
                      PRIVATE rocprofiler-register::rocprofiler-register)
set_target_properties(
    rocdecode-invalid
    PROPERTIES OUTPUT_NAME rocdecode
               SOVERSION 1
               VERSION 1.0
               LIBRARY_OUTPUT_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/invalid)
rocp_register_strip_target(rocdecode-invalid)
