#
#
#

if(NOT TARGET rocprofiler-register::rocprofiler-register)
    # find_package(rocprofiler-register REQUIRED)
endif()

add_library(hsa-runtime SHARED)
add_library(hsa-runtime::hsa-runtime ALIAS hsa-runtime)
target_sources(hsa-runtime PRIVATE hsa-runtime.cpp hsa-runtime.hpp)
target_include_directories(hsa-runtime
                           PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}>)
target_link_libraries(hsa-runtime PRIVATE rocprofiler-register::rocprofiler-register)
set_target_properties(
    hsa-runtime
    PROPERTIES OUTPUT_NAME hsa-runtime64
               SOVERSION 2
               VERSION 2.1.0)
rocp_register_strip_target(hsa-runtime)

add_library(hsa-runtime-invalid SHARED)
add_library(hsa-runtime::hsa-runtime-invalid ALIAS hsa-runtime-invalid)
target_sources(hsa-runtime-invalid PRIVATE hsa-runtime.cpp hsa-runtime.hpp)
target_include_directories(hsa-runtime-invalid
                           PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}>)
target_link_libraries(hsa-runtime-invalid
                      PRIVATE rocprofiler-register::rocprofiler-register)
set_target_properties(
    hsa-runtime-invalid
    PROPERTIES OUTPUT_NAME hsa-runtime64
               SOVERSION 1
               VERSION 1.2.3
               LIBRARY_OUTPUT_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/invalid)
rocp_register_strip_target(hsa-runtime-invalid)
