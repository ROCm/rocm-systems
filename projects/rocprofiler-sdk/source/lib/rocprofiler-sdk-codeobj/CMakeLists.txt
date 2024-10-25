# ##############################################################################
# # Copyright (c) 2024 Advanced Micro Devices, Inc. # # Permission is hereby
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

set(CMAKE_INSTALL_DEFAULT_COMPONENT_NAME "development")

add_library(rocprofiler-sdk-codeobj INTERFACE)
add_library(rocprofiler-sdk::rocprofiler-sdk-codeobj ALIAS rocprofiler-sdk-codeobj)

target_include_directories(
    rocprofiler-sdk-codeobj
    INTERFACE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/source/include>
              $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

target_link_libraries(
    rocprofiler-sdk-codeobj
    INTERFACE rocprofiler-sdk::rocprofiler-sdk-amd-comgr
              rocprofiler-sdk::rocprofiler-sdk-dw rocprofiler-sdk::rocprofiler-sdk-elf)

if(ROCPROFILER_BUILD_TESTS)
    add_subdirectory(tests)
endif()
