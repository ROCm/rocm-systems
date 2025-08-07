# ###############################################################################
# # Copyright (c) 2022 Advanced Micro Devices, Inc.
# #
# # Permission is hereby granted, free of charge, to any person obtaining a copy
# # of this software and associated documentation files (the "Software"), to
# # deal in the Software without restriction, including without limitation the
# # rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
# # sell copies of the Software, and to permit persons to whom the Software is
# # furnished to do so, subject to the following conditions:
# #
# # The above copyright notice and this permission notice shall be included in
# # all copies or substantial portions of the Software.
# #
# # THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# # IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# # FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# # AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# # LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# # FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# # IN THE SOFTWARE.
# ###############################################################################

file(GLOB ROCPROFILER_UTIL_SRC_FILES ${PROJECT_SOURCE_DIR}/src/utils/helper.cpp)

file(GLOB FILE_SOURCES_V1 "file_v1.cpp")
add_library(file_plugin_v1 SHARED ${FILE_SOURCES_V1} ${ROCPROFILER_UTIL_SRC_FILES})

set_target_properties(
    file_plugin_v1
    PROPERTIES CXX_VISIBILITY_PRESET hidden
               DEFINE_SYMBOL "ROCPROFILER_EXPORTS"
               LINK_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/../exportmap
               LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib/rocprofiler/v1
               INSTALL_RPATH "${ROCM_APPEND_PRIVLIB_RPATH}"
               OUTPUT_NAME file_plugin
               VERSION 1.0.0
               SOVERSION 1)

target_compile_definitions(file_plugin_v1 PRIVATE HIP_PROF_HIP_API_STRING=1
                                               __HIP_PLATFORM_HCC__=1)

target_include_directories(file_plugin_v1 PRIVATE ${PROJECT_SOURCE_DIR})

target_link_options(
    file_plugin_v1 PRIVATE -Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/../exportmap
    -Wl,--no-undefined)

target_link_libraries(file_plugin_v1 PRIVATE rocprofiler-v2 hsa-runtime64::hsa-runtime64
                                          stdc++fs amd_comgr dl)

# Prepare Name Link SO files for V1 file plugin Library
add_custom_command(
    TARGET file_plugin_v1
    POST_BUILD
    COMMAND
        ${CMAKE_COMMAND} -E copy
        ${PROJECT_BINARY_DIR}/lib/rocprofiler/v1/libfile_plugin.so.1*
        ${PROJECT_BINARY_DIR}/lib/rocprofiler/.)

install(TARGETS file_plugin_v1 LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}/${PROJECT_NAME}
                                    COMPONENT asan NAMELINK_SKIP)
install(TARGETS file_plugin_v1 LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}/${PROJECT_NAME}
                                    COMPONENT runtime NAMELINK_SKIP)

file(GLOB FILE_SOURCES "file.cpp")
add_library(file_plugin SHARED ${FILE_SOURCES} ${ROCPROFILER_UTIL_SRC_FILES})

set_target_properties(
    file_plugin
    PROPERTIES CXX_VISIBILITY_PRESET hidden
               DEFINE_SYMBOL "ROCPROFILER_EXPORTS"
               LINK_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/../exportmap
               LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib/rocprofiler
               INSTALL_RPATH "${ROCM_APPEND_PRIVLIB_RPATH}"
               VERSION 2.0.0
               SOVERSION 2)

target_compile_definitions(file_plugin PRIVATE HIP_PROF_HIP_API_STRING=1
                                               __HIP_PLATFORM_AMD__=1)

target_include_directories(file_plugin PRIVATE ${PROJECT_SOURCE_DIR})

target_link_options(
    file_plugin PRIVATE -Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/../exportmap
    -Wl,--no-undefined)

target_link_libraries(file_plugin PRIVATE rocprofiler-v2 hsa-runtime64::hsa-runtime64
                                          stdc++fs amd_comgr dl)

install(TARGETS file_plugin LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}/${PROJECT_NAME}
                                    COMPONENT asan)
install(TARGETS file_plugin LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}/${PROJECT_NAME}
                                    COMPONENT runtime)
