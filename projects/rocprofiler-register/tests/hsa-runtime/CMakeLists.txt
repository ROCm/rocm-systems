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
set_target_properties(hsa-runtime PROPERTIES OUTPUT_NAME hsa-runtime64)
# WINDOWS-DIVERGENCE: PE/COFF DLL names do not encode SOVERSION; a target
# with SOVERSION 2 produces hsa-runtime64-2.dll on Windows which the test
# binary cannot LoadLibrary by base name. Linux retains the symlink chain.
if(NOT WIN32)
    set_target_properties(hsa-runtime PROPERTIES SOVERSION 2 VERSION 2.1.0)
endif()
rocp_register_strip_target(hsa-runtime)

add_library(hsa-runtime-invalid SHARED)
add_library(hsa-runtime::hsa-runtime-invalid ALIAS hsa-runtime-invalid)
target_sources(hsa-runtime-invalid PRIVATE hsa-runtime.cpp hsa-runtime.hpp)
target_include_directories(hsa-runtime-invalid
                           PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}>)
target_link_libraries(hsa-runtime-invalid
                      PRIVATE rocprofiler-register::rocprofiler-register)
set_target_properties(hsa-runtime-invalid PROPERTIES OUTPUT_NAME hsa-runtime64)
# WINDOWS-DIVERGENCE: on Windows, the SHARED library .dll lands in
# RUNTIME_OUTPUT_DIRECTORY (not LIBRARY_OUTPUT_DIRECTORY which holds the
# import .lib). The "invalid" variant tests rely on filesystem isolation
# from the valid variant, so place the .dll in invalid/ and the import .lib
# alongside it for symmetry with Linux's libdir layout.
if(WIN32)
    set_target_properties(
        hsa-runtime-invalid
        PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                   ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/invalid
                   LIBRARY_OUTPUT_DIRECTORY
                   ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/invalid
                   ARCHIVE_OUTPUT_DIRECTORY
                   ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/invalid)
else()
    set_target_properties(
        hsa-runtime-invalid
        PROPERTIES SOVERSION 1
                   VERSION 1.2.3
                   LIBRARY_OUTPUT_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/invalid)
endif()
rocp_register_strip_target(hsa-runtime-invalid)
