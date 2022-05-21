################################################################################
## Copyright (c) 2018-2022 Advanced Micro Devices, Inc.
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

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})

if(${LIBRARY_TYPE} STREQUAL STATIC)
  add_compile_definitions(STATIC_BUILD=1)
endif()

option(DEBUG_TRACE "Enable debug tracing")
if(DEBUG_TRACE)
  add_compile_definitions(DEBUG_TRACE_ON=1)
endif()

find_package(Python3 COMPONENTS Interpreter REQUIRED)

execute_process(COMMAND ${Python3_EXECUTABLE} -c "import CppHeaderParser"
                RESULT_VARIABLE CPP_HEADER_PARSER
                OUTPUT_QUIET)

if(NOT ${CPP_HEADER_PARSER} EQUAL 0)
  message(FATAL_ERROR "\
The \"CppHeaderParser\" Python3 package is not installed. \
Please install it using the following command: \"pip3 install CppHeaderParser\".\
")
endif()

get_property(HSA_RUNTIME_INCLUDE_DIRECTORIES TARGET hsa-runtime64::hsa-runtime64 PROPERTY INTERFACE_INCLUDE_DIRECTORIES)
find_file(HSA_H hsa.h
          PATHS ${HSA_RUNTIME_INCLUDE_DIRECTORIES}
          PATH_SUFFIXES hsa
          NO_DEFAULT_PATH
          REQUIRED)
get_filename_component(HSA_RUNTIME_INC_PATH ${HSA_H} DIRECTORY)

## Generate the HSA wrapper functions header
add_custom_command(
  OUTPUT hsa_prof_str.h
  COMMAND ${Python3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/script/hsaap.py ${CMAKE_CURRENT_BINARY_DIR} "${HSA_RUNTIME_INC_PATH}"  > /dev/null
  DEPENDS ${PROJECT_SOURCE_DIR}/script/hsaap.py
          "${HSA_RUNTIME_INC_PATH}/hsa.h" "${HSA_RUNTIME_INC_PATH}/hsa_ext_amd.h"
          "${HSA_RUNTIME_INC_PATH}/hsa_ext_image.h" "${HSA_RUNTIME_INC_PATH}/hsa_api_trace.h"
  COMMENT "Generating hsa_prof_str.h...")

## Generate the HSA pretty printers
add_custom_command(
  OUTPUT hsa_ostream_ops.h
  COMMAND ${CMAKE_C_COMPILER} -E "${HSA_RUNTIME_INC_PATH}/hsa.h" -o hsa.h.i
  COMMAND ${CMAKE_C_COMPILER} -E "${HSA_RUNTIME_INC_PATH}/hsa_ext_amd.h" -o hsa_ext_amd.h.i
  BYPRODUCTS hsa.h.i hsa_ext_amd.h.i
  COMMAND ${Python3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/script/gen_ostream_ops.py
          -in hsa.h.i,hsa_ext_amd.h.i -out hsa_ostream_ops.h  > /dev/null
  DEPENDS ${PROJECT_SOURCE_DIR}/script/gen_ostream_ops.py
          "${HSA_RUNTIME_INC_PATH}/hsa.h" "${HSA_RUNTIME_INC_PATH}/hsa_ext_amd.h"
  COMMENT "Generating hsa_ostream_ops.h...")

get_property(HIP_INCLUDE_DIRECTORIES TARGET hip::amdhip64 PROPERTY INTERFACE_INCLUDE_DIRECTORIES)
find_file(HIP_RUNTIME_API_H hip_runtime_api.h
          PATHS ${HIP_INCLUDE_DIRECTORIES}
          PATH_SUFFIXES hip
          NO_DEFAULT_PATH
          REQUIRED)

## Generate the HIP pretty printers
add_custom_command(
  OUTPUT hip_ostream_ops.h
  COMMAND ${CMAKE_C_COMPILER} "$<$<BOOL:${HIP_INCLUDE_DIRECTORIES}>:-I$<JOIN:${HIP_INCLUDE_DIRECTORIES},$<SEMICOLON>-I>>"
          -E "${HIP_RUNTIME_API_H}" -D__HIP_PLATFORM_HCC__=1 -D__HIP_ROCclr__=1 -o hip_runtime_api.h.i
  BYPRODUCTS hip_runtime_api.h.i
  COMMAND ${Python3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/script/gen_ostream_ops.py
          -in hip_runtime_api.h.i -out hip_ostream_ops.h > /dev/null
  DEPENDS ${PROJECT_SOURCE_DIR}/script/gen_ostream_ops.py "${HIP_RUNTIME_API_H}"
  COMMENT "Generating hip_ostream_ops.h..."
  COMMAND_EXPAND_LISTS)

set(PUBLIC_HEADERS
  roctx.h
  roctracer.h
  roctracer_ext.h
  roctracer_hip.h
  roctracer_hcc.h
  roctracer_hsa.h
  roctracer_roctx.h
  ext/prof_protocol.h)

foreach(header ${PUBLIC_HEADERS})
  get_filename_component(header_subdir ${header} DIRECTORY)
  install(FILES ${PROJECT_SOURCE_DIR}/inc/${header} DESTINATION include/${ROCTRACER_NAME}/${header_subdir})
endforeach()

set(GENERATED_HEADERS
  hip_ostream_ops.h
  hsa_prof_str.h
  hsa_ostream_ops.h)

foreach(header ${GENERATED_HEADERS})
  install(FILES ${CMAKE_CURRENT_BINARY_DIR}/${header} DESTINATION include/${ROCTRACER_NAME})
endforeach()

## Build the ROCtracer library
file(GLOB ROCTRACER_SOURCES "roctracer/*.cpp" "util/*.cpp")
add_library(roctracer ${LIBRARY_TYPE} ${ROCTRACER_SOURCES} ${GENERATED_HEADERS})

set_target_properties(roctracer PROPERTIES
                      CXX_VISIBILITY_PRESET hidden
                      OUTPUT_NAME "roctracer64"
                      DEFINE_SYMBOL "ROCTRACER_EXPORTS"
                      LINK_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/roctracer/exportmap
                      VERSION ${ROCTRACER_VERSION}
                      SOVERSION ${ROCTRACER_VERSION_MAJOR})

target_compile_definitions(roctracer
  PUBLIC AMD_INTERNAL_BUILD
  PRIVATE PROF_API_IMPL HIP_PROF_HIP_API_STRING=1 __HIP_PLATFORM_AMD__=1)

target_include_directories(roctracer
  PUBLIC
    ${HIP_INCLUDE_DIRECTORIES} ${HSA_RUNTIME_INCLUDE_DIRECTORIES}
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/inc>
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/roctracer ${CMAKE_CURRENT_SOURCE_DIR})

target_link_options(roctracer PRIVATE -Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/roctracer/exportmap -Wl,--no-undefined)
target_link_libraries(roctracer PRIVATE hsa-runtime64::hsa-runtime64 Threads::Threads dl)

install(TARGETS roctracer LIBRARY DESTINATION lib)

## Build the ROCTX library
file(GLOB ROCTX_SOURCES "roctx/*.cpp")
add_library(roctx ${LIBRARY_TYPE} ${ROCTX_SOURCES})

set_target_properties(roctx PROPERTIES
                      CXX_VISIBILITY_PRESET hidden
                      OUTPUT_NAME "roctx64"
                      DEFINE_SYMBOL "ROCTX_EXPORTS"
                      LINK_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/roctx/exportmap
                      VERSION ${ROCTRACER_VERSION}
                      SOVERSION ${ROCTRACER_VERSION_MAJOR})

target_include_directories(roctx
  PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/inc>
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/roctracer ${CMAKE_CURRENT_SOURCE_DIR})

target_link_options(roctx PRIVATE -Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/roctx/exportmap -Wl,--no-undefined)

install(TARGETS roctx LIBRARY DESTINATION lib)

## Build the tracer_tool library
if (${LIBRARY_TYPE} STREQUAL SHARED)

file(GLOB TRACER_TOOL_SOURCES "tracer_tool/*.cpp" "util/*.cpp")
add_library(roctracer_tool SHARED ${TRACER_TOOL_SOURCES})

set_target_properties(roctracer_tool PROPERTIES
                      CXX_VISIBILITY_PRESET hidden
                      LINK_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/tracer_tool/exportmap)

target_compile_definitions(roctracer_tool
  PRIVATE HIP_PROF_HIP_API_STRING=1 __HIP_PLATFORM_HCC__=1)

target_include_directories(roctracer_tool
  PRIVATE
    ${PROJECT_SOURCE_DIR}/inc ${CMAKE_CURRENT_SOURCE_DIR}/roctracer
    ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(roctracer_tool roctracer hsa-runtime64::hsa-runtime64 Threads::Threads atomic dl)
target_link_options(roctracer_tool PRIVATE -Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/tracer_tool/exportmap -Wl,--no-undefined)

install(TARGETS roctracer_tool LIBRARY DESTINATION lib/${ROCTRACER_NAME})
endif()

option(FILE_REORG_BACKWARD_COMPATIBILITY "Enable File Reorg with backward compatibility" ON)
if(FILE_REORG_BACKWARD_COMPATIBILITY)
  include(${PROJECT_SOURCE_DIR}/roctracer-backward-compat.cmake)
endif()
