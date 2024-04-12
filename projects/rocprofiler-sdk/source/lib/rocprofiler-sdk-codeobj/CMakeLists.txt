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

set(CMAKE_INSTALL_DEFAULT_COMPONENT_NAME "codeobj")

set(LIB_CODEOBJ_PARSER_SOURCES code_object_track.cpp code_printing.cpp disassembly.cpp)
set(LIB_CODEOBJ_PARSER_HEADERS code_object_track.hpp code_printing.hpp disassembly.hpp
                               segment.hpp)

add_library(rocprofiler-sdk-codeobj STATIC)
target_sources(rocprofiler-sdk-codeobj PRIVATE ${LIB_CODEOBJ_PARSER_SOURCES})
target_link_libraries(
    rocprofiler-sdk-codeobj
    PRIVATE rocprofiler::rocprofiler-amd-comgr rocprofiler::rocprofiler-dw
            rocprofiler::rocprofiler-elf rocprofiler::rocprofiler-build-flags
            rocprofiler::rocprofiler-memcheck rocprofiler::rocprofiler-common-library)

target_include_directories(rocprofiler-sdk-codeobj PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

set_target_properties(
    rocprofiler-sdk-codeobj
    PROPERTIES LIBRARY_OUTPUT_DIRECTORY
               ${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}/rocprofiler-sdk
               POSITION_INDEPENDENT_CODE ON)

install(
    TARGETS rocprofiler-sdk-codeobj
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/rocprofiler-sdk
    COMPONENT tools
    EXPORT rocprofiler-sdk-codeobj-targets)

install(
    FILES ${LIB_CODEOBJ_PARSER_HEADERS}
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/rocprofiler-sdk-codeobj
    COMPONENT tools)
