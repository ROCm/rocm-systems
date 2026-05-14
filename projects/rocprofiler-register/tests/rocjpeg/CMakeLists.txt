#
#
#

if(NOT TARGET rocprofiler-register::rocprofiler-register)
    # find_package(rocprofiler-register REQUIRED)
endif()

add_library(rocjpeg SHARED)
add_library(rocjpeg::rocjpeg ALIAS rocjpeg)
target_sources(rocjpeg PRIVATE rocjpeg.cpp rocjpeg.hpp)
target_include_directories(rocjpeg PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}>)
target_link_libraries(rocjpeg PRIVATE rocprofiler-register::rocprofiler-register)
set_target_properties(rocjpeg PROPERTIES OUTPUT_NAME rocjpeg)
# WINDOWS-DIVERGENCE: see tests/hsa-runtime/CMakeLists.txt for rationale.
if(NOT WIN32)
    set_target_properties(rocjpeg PROPERTIES SOVERSION 1 VERSION 1.0)
endif()
rocp_register_strip_target(rocjpeg)

add_library(rocjpeg-invalid SHARED)
add_library(rocjpeg::rocjpeg-invalid ALIAS rocjpeg-invalid)
target_sources(rocjpeg-invalid PRIVATE rocjpeg.cpp rocjpeg.hpp)
target_include_directories(rocjpeg-invalid
                           PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}>)
target_link_libraries(rocjpeg-invalid PRIVATE rocprofiler-register::rocprofiler-register)
set_target_properties(rocjpeg-invalid PROPERTIES OUTPUT_NAME rocjpeg)
if(WIN32)
    set_target_properties(
        rocjpeg-invalid
        PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                   ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/invalid
                   LIBRARY_OUTPUT_DIRECTORY
                   ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/invalid
                   ARCHIVE_OUTPUT_DIRECTORY
                   ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/invalid)
else()
    set_target_properties(
        rocjpeg-invalid
        PROPERTIES SOVERSION 1
                   VERSION 1.0
                   LIBRARY_OUTPUT_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/invalid)
endif()
rocp_register_strip_target(rocjpeg-invalid)
