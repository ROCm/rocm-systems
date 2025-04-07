################################################################################
## Copyright (c) 2022 Advanced Micro Devices, Inc.
##
## Permission is hereby granted, free of charge, to any person obtaining a copy
## of this software and associated documentation files (the "Software"), to
## deal in the Software without restriction, including without limitation the
## rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
## sell copies of the Software, and to permit persons to whom the Software is
## furnished to do so, subject to the following conditions:
##
## The above copyright notice and this permission notice shall be included in
## all copies or substantial portions of the Software.
##
## THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
## IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
## FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
## AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
## LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
## FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
## IN THE SOFTWARE.
################################################################################

find_package(amd_comgr REQUIRED CONFIG
  PATHS
    /opt/rocm/
  PATH_SUFFIXES
    lib/cmake/amd_comgr
)
MESSAGE(STATUS "Code Object Manager found at ${amd_comgr_DIR}.")

file(GLOB FILE_SOURCES "*.cpp")
add_library(file_plugin ${LIBRARY_TYPE} ${FILE_SOURCES})

set_target_properties(file_plugin PROPERTIES
                      CXX_VISIBILITY_PRESET hidden
                      LINK_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/../exportmap
                      LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}
                      INSTALL_RPATH "${ROCM_APPEND_PRIVLIB_RPATH}")

target_compile_definitions(file_plugin
  PRIVATE HIP_PROF_HIP_API_STRING=1 __HIP_PLATFORM_AMD__=1)

target_include_directories(file_plugin PRIVATE ${PROJECT_SOURCE_DIR}/inc)

target_link_options(file_plugin PRIVATE -Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/../exportmap -Wl,--no-undefined)

target_link_libraries(file_plugin PRIVATE util roctracer  amd_comgr hsa-runtime64::hsa-runtime64 stdc++fs amd_comgr)

install(TARGETS file_plugin LIBRARY
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/${PROJECT_NAME}
        COMPONENT runtime)

install(TARGETS file_plugin LIBRARY
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/${PROJECT_NAME}
        COMPONENT asan)
