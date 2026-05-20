#
#
#

if(NOT TARGET rocprofiler-register::rocprofiler-register)
    # find_package(rocprofiler-register REQUIRED)
endif()

add_library(hipfile SHARED)
add_library(hipfile::hipfile ALIAS hipfile)
target_sources(hipfile PRIVATE hipfile.cpp hipfile.hpp)
target_include_directories(hipfile PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}>)
target_link_libraries(hipfile PRIVATE rocprofiler-register::rocprofiler-register)
set_target_properties(
    hipfile
    PROPERTIES OUTPUT_NAME hipfile
               SOVERSION 1
               VERSION 1.0)
rocp_register_strip_target(hipfile)

add_library(hipfile-invalid SHARED)
add_library(hipfile::hipfile-invalid ALIAS hipfile-invalid)
target_sources(hipfile-invalid PRIVATE hipfile.cpp hipfile.hpp)
target_include_directories(hipfile-invalid
                           PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}>)
target_link_libraries(hipfile-invalid PRIVATE rocprofiler-register::rocprofiler-register)
set_target_properties(
    hipfile-invalid
    PROPERTIES OUTPUT_NAME hipfile
               SOVERSION 1
               VERSION 1.0
               LIBRARY_OUTPUT_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/invalid)
rocp_register_strip_target(hipfile-invalid)
