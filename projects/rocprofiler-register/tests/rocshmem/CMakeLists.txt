#
#
#

if(NOT TARGET rocprofiler-register::rocprofiler-register)
    # find_package(rocprofiler-register REQUIRED)
endif()

add_library(rocshmem SHARED)
add_library(rocshmem::rocshmem ALIAS rocshmem)
target_sources(rocshmem PRIVATE rocshmem.cpp rocshmem.hpp)
target_include_directories(rocshmem PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}>)
target_link_libraries(rocshmem PRIVATE rocprofiler-register::rocprofiler-register)
set_target_properties(
    rocshmem
    PROPERTIES OUTPUT_NAME rocshmem
               SOVERSION 3
               VERSION 3.0)
rocp_register_strip_target(rocshmem)

add_library(rocshmem-invalid SHARED)
add_library(rocshmem::rocshmem-invalid ALIAS rocshmem-invalid)
target_sources(rocshmem-invalid PRIVATE rocshmem.cpp rocshmem.hpp)
target_include_directories(rocshmem-invalid
                           PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}>)
target_link_libraries(rocshmem-invalid PRIVATE rocprofiler-register::rocprofiler-register)
set_target_properties(
    rocshmem-invalid
    PROPERTIES OUTPUT_NAME rocshmem
               SOVERSION 3
               VERSION 3.0
               LIBRARY_OUTPUT_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/invalid)
rocp_register_strip_target(rocshmem-invalid)
