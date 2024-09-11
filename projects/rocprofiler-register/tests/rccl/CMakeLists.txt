#
#
#

if(NOT TARGET rocprofiler-register::rocprofiler-register)
    # find_package(rocprofiler-register REQUIRED)
endif()

add_library(rccl SHARED)
add_library(rccl::rccl ALIAS rccl)
target_sources(rccl PRIVATE rccl.cpp rccl.hpp)
target_include_directories(rccl PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}>)
target_link_libraries(rccl PRIVATE rocprofiler-register::rocprofiler-register)
set_target_properties(
    rccl
    PROPERTIES OUTPUT_NAME rccl
               SOVERSION 1
               VERSION 1.0)
rocp_register_strip_target(rccl)

add_library(rccl-invalid SHARED)
add_library(rccl::rccl-invalid ALIAS rccl-invalid)
target_sources(rccl-invalid PRIVATE rccl.cpp rccl.hpp)
target_include_directories(rccl-invalid
                           PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}>)
target_link_libraries(rccl-invalid PRIVATE rocprofiler-register::rocprofiler-register)
set_target_properties(
    rccl-invalid
    PROPERTIES OUTPUT_NAME rccl
               SOVERSION 1
               VERSION 1.0
               LIBRARY_OUTPUT_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/invalid)
rocp_register_strip_target(rccl-invalid)
