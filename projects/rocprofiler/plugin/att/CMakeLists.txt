# ##############################################################################
# # Copyright (c) 2022 Advanced Micro Devices, Inc. # # Permission is hereby
# granted, free of charge, to any person obtaining a copy # of this software and
# associated documentation files (the "Software"), to # deal in the Software
# without restriction, including without limitation the # rights to use, copy,
# modify, merge, publish, distribute, sublicense, and/or # sell copies of the
# Software, and to permit persons to whom the Software is # furnished to do so,
# subject to the following conditions: # # The above copyright notice and this
# permission notice shall be included in # all copies or substantial portions of
# the Software. # # THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY
# KIND, EXPRESS OR # IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, # FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO
# EVENT SHALL THE # AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER # LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING # FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
# USE OR OTHER DEALINGS # IN THE SOFTWARE.
# ##############################################################################

find_library(
  ROCPROFV2_ATT rocprofv2_att
  HINTS ${CMAKE_INSTALL_PREFIX}
  PATHS ${ROCM_PATH}
  PATH_SUFFIXES hsa-amd-aqlprofile)

set(ENV{ROCPROFV2_ATT_LIB_PATH} $ROCPROFV2_ATT)

# Building att plugin library
file(GLOB ROCPROFILER_UTIL_SRC_FILES ${PROJECT_SOURCE_DIR}/src/utils/helper.cpp)
file(GLOB FILE_SOURCES att.cpp)
add_library(att_plugin SHARED ${FILE_SOURCES} ${ROCPROFILER_UTIL_SRC_FILES})

set_target_properties(
  att_plugin
  PROPERTIES CXX_VISIBILITY_PRESET hidden
             LINK_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/../exportmap
             LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})

target_compile_definitions(att_plugin PRIVATE HIP_PROF_HIP_API_STRING=1
                                              __HIP_PLATFORM_HCC__=1)

target_include_directories(
  att_plugin PRIVATE ${PROJECT_SOURCE_DIR}
                     ${CMAKE_CURRENT_SOURCE_DIR})
target_link_options(
  att_plugin PRIVATE
  -Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/../exportmap
  -Wl,--no-undefined)
target_link_libraries(att_plugin PRIVATE rocprofiler-v2
                                         hsa-runtime64::hsa-runtime64 stdc++fs)

install(TARGETS att_plugin
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}/${PROJECT_NAME}
                COMPONENT asan)
install(TARGETS att_plugin
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}/${PROJECT_NAME}
                COMPONENT runtime)

configure_file(att.py att/att.py COPYONLY)
configure_file(trace_view.py att/trace_view.py COPYONLY)
# configure_file(t.db att/t.db COPYONLY)
configure_file(ui/index.html att/ui/index.html COPYONLY)
configure_file(ui/logo.svg att/ui/logo.svg COPYONLY)
configure_file(ui/styles.css att/ui/styles.css COPYONLY)
# configure_file(ui/trace.json att/ui/trace.json COPYONLY)
install(
  DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/att
  DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/rocprofiler
  USE_SOURCE_PERMISSIONS
  COMPONENT runtime)
