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

find_package(Python3 COMPONENTS Interpreter REQUIRED)

execute_process(
  COMMAND ${Python3_EXECUTABLE} -c "import CppHeaderParser"
  RESULT_VARIABLE CPPHEADERPARSER
  OUTPUT_QUIET
)

if(NOT ${CPPHEADERPARSER} EQUAL 0)
  message(FATAL_ERROR
    "The \"CppHeaderParser\" Python3 package is not installed. Please install it using the following command: \"pip3 install CppHeaderParser\"."
  )
endif()

if(${LIBRARY_TYPE} STREQUAL STATIC)
  add_definitions(-DSTATIC_BUILD=1)
endif()

option(DEBUG_TRACE "Enable debug tracing")
if(DEBUG_TRACE)
    add_definitions(-DDEBUG_TRACE_ON=1)
endif()

set(GEN_INC_DIR ${PROJECT_BINARY_DIR}/inc)

# FIXME!!! should not depend on another project's internal files
execute_process(COMMAND ${CMAKE_COMMAND} -E make_directory "${GEN_INC_DIR}/rocprofiler")
execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink "${PROJECT_SOURCE_DIR}/../rocprofiler/inc/rocprofiler.h" "${GEN_INC_DIR}/rocprofiler/rocprofiler.h")
execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink "${PROJECT_SOURCE_DIR}/../rocprofiler/src/core/activity.h" "${GEN_INC_DIR}/rocprofiler/activity.h")

get_property(HSA_RUNTIME_INCLUDE_DIRECTORIES TARGET hsa-runtime64::hsa-runtime64 PROPERTY INTERFACE_INCLUDE_DIRECTORIES)
find_file(hsa_h_file hsa.h
    PATHS ${HSA_RUNTIME_INCLUDE_DIRECTORIES}
    PATH_SUFFIXES hsa
    NO_DEFAULT_PATH
    REQUIRED
)
get_filename_component(HSA_RUNTIME_INC_PATH ${hsa_h_file} DIRECTORY)

add_custom_command(
    OUTPUT "${GEN_INC_DIR}/hsa_prof_str.h"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${GEN_INC_DIR}"
    COMMAND ${Python3_EXECUTABLE}
        "${PROJECT_SOURCE_DIR}/script/hsaap.py"
        "${PROJECT_BINARY_DIR}" "${HSA_RUNTIME_INC_PATH}"  > /dev/null
    DEPENDS "${PROJECT_SOURCE_DIR}/script/hsaap.py"
            "${HSA_RUNTIME_INC_PATH}/hsa.h" "${HSA_RUNTIME_INC_PATH}/hsa_ext_amd.h"
            "${HSA_RUNTIME_INC_PATH}/hsa_ext_image.h" "${HSA_RUNTIME_INC_PATH}/hsa_api_trace.h"
    COMMENT "Generating hsa_prof_str.h..."
)

add_custom_command(
    OUTPUT "${GEN_INC_DIR}/hsa_ostream_ops.h"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${GEN_INC_DIR}"
    COMMAND ${CMAKE_C_COMPILER} -E "${HSA_RUNTIME_INC_PATH}/hsa.h" -o "${GEN_INC_DIR}/hsa.h.i"
    COMMAND ${CMAKE_C_COMPILER} -E "${HSA_RUNTIME_INC_PATH}/hsa_ext_amd.h" -o "${GEN_INC_DIR}/hsa_ext_amd.h.i"
    BYPRODUCTS "${GEN_INC_DIR}/hsa.h.i" "${GEN_INC_DIR}/hsa_ext_amd.h.i"
    COMMAND ${Python3_EXECUTABLE}
        "${PROJECT_SOURCE_DIR}/script/gen_ostream_ops.py"
        -in "${GEN_INC_DIR}/hsa.h.i,${GEN_INC_DIR}/hsa_ext_amd.h.i"
        -out "${GEN_INC_DIR}/hsa_ostream_ops.h"  > /dev/null
    DEPENDS "${PROJECT_SOURCE_DIR}/script/gen_ostream_ops.py"
            "${HSA_RUNTIME_INC_PATH}/hsa.h" "${HSA_RUNTIME_INC_PATH}/hsa_ext_amd.h"
    COMMENT "Generating hsa_ostream_ops.h..."
)

get_property(HIP_INCLUDE_DIRECTORIES TARGET hip::amdhip64 PROPERTY INTERFACE_INCLUDE_DIRECTORIES)

find_file(HIP_RUNTIME_API_H hip_runtime_api.h
    PATHS ${HIP_INCLUDE_DIRECTORIES}
    PATH_SUFFIXES hip
    NO_DEFAULT_PATH
    REQUIRED
)

add_custom_command(
    OUTPUT "${GEN_INC_DIR}/hip_ostream_ops.h"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${GEN_INC_DIR}"
    COMMAND ${CMAKE_C_COMPILER}
        "$<$<BOOL:${HIP_INCLUDE_DIRECTORIES}>:-I$<JOIN:${HIP_INCLUDE_DIRECTORIES},$<SEMICOLON>-I>>"
        -E "${HIP_RUNTIME_API_H}" -D__HIP_PLATFORM_HCC__=1 -D__HIP_ROCclr__=1 -o ${GEN_INC_DIR}/hip_runtime_api.h.i
    BYPRODUCTS "${GEN_INC_DIR}/hip_runtime_api.h.i"
    COMMAND ${Python3_EXECUTABLE}
        "${PROJECT_SOURCE_DIR}/script/gen_ostream_ops.py"
        -in "${GEN_INC_DIR}/hip_runtime_api.h.i"
        -out "${GEN_INC_DIR}/hip_ostream_ops.h" > /dev/null
    DEPENDS "${PROJECT_SOURCE_DIR}/script/gen_ostream_ops.py" "${HIP_RUNTIME_API_H}"
    COMMENT "Generating hip_ostream_ops.h..."
    COMMAND_EXPAND_LISTS
)

# Build the ROCtracer library

file(GLOB ROCTRACER_SOURCES "src/core/*.cpp" "src/util/*.cpp")

add_library(roctracer ${LIBRARY_TYPE}
    ${ROCTRACER_SOURCES}
    "${GEN_INC_DIR}/hsa_ostream_ops.h"
    "${GEN_INC_DIR}/hsa_prof_str.h"
    "${GEN_INC_DIR}/hip_ostream_ops.h"
)

set_target_properties(roctracer PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    OUTPUT_NAME "roctracer64"
    LINK_DEPENDS core.exportmap
    VERSION ${ROCTRACER_VERSION}
    SOVERSION ${ROCTRACER_VERSION_MAJOR}
)

target_compile_definitions(roctracer
  PUBLIC
    AMD_INTERNAL_BUILD
  PRIVATE
    PROF_API_IMPL
    HIP_PROF_HIP_API_STRING=1
    __HIP_PLATFORM_HCC__=1
)

target_include_directories(roctracer
  PUBLIC
    ${HIP_INCLUDE_DIRECTORIES} ${HSA_RUNTIME_INCLUDE_DIRECTORIES}
  PRIVATE
    ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/inc ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_BINARY_DIR}/inc
)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/src/core/exportmap.in core.exportmap @ONLY)
target_link_options(roctracer PRIVATE -Wl,--version-script=core.exportmap -Wl,--no-undefined)

target_link_libraries(roctracer PRIVATE hsa-runtime64::hsa-runtime64 Threads::Threads dl)

install(TARGETS ${ROCTRACER_TARGET} LIBRARY DESTINATION lib)
foreach(header ${PUBLIC_HEADERS})
    get_filename_component(header_subdir ${header} DIRECTORY)
    install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/inc/${header} DESTINATION include/${ROCTRACER_NAME}/${header_subdir})
endforeach()

foreach(header ${GEN_HEADERS})
    get_filename_component(header_subdir ${header} DIRECTORY)
    install(FILES ${GEN_INC_DIR}/${header} DESTINATION include/${ROCTRACER_NAME}/${header_subdir})
endforeach()

# Build the ROCTX library

file(GLOB ROCTX_SOURCES "src/roctx/*.cpp")

add_library(roctx ${LIBRARY_TYPE} ${ROCTX_SOURCES})

set_target_properties(roctx PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    OUTPUT_NAME "roctx64"
    LINK_DEPENDS roctx.exportmap
    VERSION ${ROCTRACER_VERSION}
    SOVERSION ${ROCTRACER_VERSION_MAJOR}
)

target_include_directories(roctx PRIVATE
    ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/inc
    ${PROJECT_SOURCE_DIR}/src
)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/src/roctx/exportmap.in roctx.exportmap @ONLY)
target_link_options(roctx PRIVATE -Wl,--version-script=roctx.exportmap -Wl,--no-undefined)

install(TARGETS roctx LIBRARY DESTINATION lib)
