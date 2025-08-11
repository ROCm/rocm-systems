################################################################################
# Copyright (c) 2022 Advanced Micro Devices, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to
# deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
# sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# IN THE SOFTWARE.
################################################################################

# Plugin shared object.
add_library(
    ctf_plugin SHARED
    ctf.cpp
    plugin.cpp
    barectf.c
    "${CMAKE_CURRENT_BINARY_DIR}/barectf.h"
    ${PROJECT_SOURCE_DIR}/src/utils/helper.cpp
    hsa_begin.cpp.i
    hsa_end.cpp.i
    hip_begin.cpp.i
    hip_end.cpp.i)
set_target_properties(
    ctf_plugin
    PROPERTIES CXX_VISIBILITY_PRESET hidden
               LINK_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/../exportmap"
               LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/lib/rocprofiler"
               INSTALL_RPATH "${ROCM_APPEND_PRIVLIB_RPATH}")
set(METADATA_STREAM_FILE_DIR "${CMAKE_INSTALL_DATADIR}/${PROJECT_NAME}/plugin/ctf")
target_compile_definitions(
    ctf_plugin
    PUBLIC AMD_INTERNAL_BUILD
    PRIVATE HIP_PROF_HIP_API_STRING=1 __HIP_PLATFORM_AMD__=1
            CTF_PLUGIN_METADATA_FILE_PATH="${METADATA_STREAM_FILE_DIR}/metadata"
            CTF_PLUGIN_INSTALL_PREFIX="${CMAKE_INSTALL_PREFIX}")

# Disabling this as it causes CI to fail, we need to make sure rocm-core is installed
# before rocprofiler Add definitions for getROCmInstallPath() API
# if(USE_GET_ROCM_PATH_API) target_compile_definitions( ctf_plugin PRIVATE
# USE_GET_ROCM_PATH_API=1) endif()

target_include_directories(
    ctf_plugin PRIVATE "${PROJECT_SOURCE_DIR}" "${CMAKE_BINARY_DIR}/src/api"
                       "${CMAKE_CURRENT_BINARY_DIR}")
target_link_options(
    ctf_plugin PRIVATE "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/../exportmap"
    -Wl,--no-undefined)
target_link_libraries(ctf_plugin PRIVATE rocprofiler-v2 hsa-runtime64::hsa-runtime64
                                         stdc++fs dl)

# Disabling this as it causes CI to fail, we need to make sure rocm-core is installed
# before rocprofiler Add link dependencies for getROCmInstallPath() API
# if(USE_GET_ROCM_PATH_API) find_library(ROCM_CORE_LIB rocm-core HINTS ${ROCM_PATH}/lib)
# target_link_libraries(ctf_plugin PRIVATE ${ROCM_CORE_LIB}) endif()

install(TARGETS ctf_plugin LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}/${PROJECT_NAME}"
                                   COMPONENT plugins)

# `gen_api_files.py` and `gen_env_yaml.py` require Python 3, CppHeaderParser, PyYAML, and
# barectf.
if(NOT Python3_EXECUTABLE)
    find_package(
        Python3
        COMPONENTS Interpreter
        REQUIRED)
endif()

message("Python: ${Python3_EXECUTABLE})")

execute_process(COMMAND Python3::Interpreter -c "print('hello')")

function(check_py3_pkg pkg_name)
    execute_process(
        COMMAND "${Python3_EXECUTABLE}" -c "import ${pkg_name}"
        RESULT_VARIABLE PY3_IMPORT_RES
        OUTPUT_QUIET)

    if(NOT (${PY3_IMPORT_RES} EQUAL 0))
        message(FATAL_ERROR "Cannot find Python 3 package `${pkg_name}`")
    endif()

    message(STATUS "Found Python 3 package `${pkg_name}`")
endfunction()

check_py3_pkg(CppHeaderParser)
check_py3_pkg(yaml)
find_program(BARECTF_RES barectf REQUIRED HINTS "$ENV{HOME}/.local/bin")

# Generate barectf YAML and C++ files for HSA API.
get_property(
    HSA_RUNTIME_INCLUDE_DIRS
    TARGET hsa-runtime64::hsa-runtime64
    PROPERTY INTERFACE_INCLUDE_DIRECTORIES)
find_file(
    HSA_H hsa.h
    PATHS ${HSA_RUNTIME_INCLUDE_DIRS}
    PATH_SUFFIXES hsa
    NO_DEFAULT_PATH REQUIRED)
get_filename_component(HSA_RUNTIME_INC_PATH "${HSA_H}" DIRECTORY)
add_custom_command(
    OUTPUT hsa_erts.yaml hsa_begin.cpp.i hsa_end.cpp.i
    COMMAND ${CMAKE_C_COMPILER} -E "${HSA_RUNTIME_INC_PATH}/hsa.h" -o hsa.h.i
    COMMAND ${CMAKE_C_COMPILER} -E "${HSA_RUNTIME_INC_PATH}/hsa_ext_amd.h" -o
            hsa_ext_amd.h.i
    COMMAND ${CMAKE_COMMAND} -E cat hsa.h.i hsa_ext_amd.h.i
            "${CMAKE_BINARY_DIR}/src/api/hsa_prof_str.h" > hsa_input.h
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/gen_api_files.py" hsa
            hsa_input.h
    BYPRODUCTS hsa.h.i hsa_ext_amd.h.i hsa_input.h
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/gen_api_files.py" "${HSA_RUNTIME_INC_PATH}/hsa.h"
            "${HSA_RUNTIME_INC_PATH}/hsa_ext_amd.h"
            "${CMAKE_BINARY_DIR}/src/api/hsa_prof_str.h"
    COMMENT "Generating HSA API files for the `ctf` plugin...")

# Generate barectf YAML and C++ files for HIP API.
get_property(
    HIP_INCLUDE_DIRS
    TARGET hip::amdhip64
    PROPERTY INTERFACE_INCLUDE_DIRECTORIES)
find_file(
    HIP_RUNTIME_API_H hip_runtime_api.h
    PATHS ${HIP_INCLUDE_DIRS}
    PATH_SUFFIXES hip
    NO_DEFAULT_PATH REQUIRED)
find_file(
    HIP_PROF_STR_H hip_prof_str.h
    PATHS ${HIP_INCLUDE_DIRS}
    PATH_SUFFIXES hip hip/amd_detail
    NO_DEFAULT_PATH REQUIRED)
list(TRANSFORM HIP_INCLUDE_DIRS PREPEND -I)
add_custom_command(
    OUTPUT hip_erts.yaml hip_begin.cpp.i hip_end.cpp.i
    COMMAND
        ${CMAKE_C_COMPILER} ${HIP_INCLUDE_DIRS} -E
        "${PROJECT_SOURCE_DIR}/src/core/session/tracer/src/hip_full_api.h"
        -D__HIP_PLATFORM_AMD__=1 -D__HIP_ROCclr__=1 -o hip_runtime_api.h.i
    COMMAND cat hip_runtime_api.h.i "${HIP_PROF_STR_H}" > hip_input.h
    BYPRODUCTS hip_runtime_api.h.i hip_input.h
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/gen_api_files.py" hip
            hip_input.h
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/gen_api_files.py"
            "${PROJECT_SOURCE_DIR}/src/core/session/tracer/src/hip_full_api.h"
            "${HIP_PROF_STR_H}"
    COMMENT "Generating HIP API files for the `ctf` plugin...")

# Generate `env.yaml` (trace environment for barectf).
add_custom_command(
    OUTPUT env.yaml
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/gen_env_yaml.py"
            ${PROJECT_VERSION}
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/gen_env_yaml.py"
    COMMENT "Generating `env.yaml`...")

# Generate raw CTF tracer with barectf.
add_custom_command(
    OUTPUT barectf.c barectf.h barectf-bitfield.h metadata
    COMMAND "${BARECTF_RES}" gen "-I${CMAKE_CURRENT_BINARY_DIR}"
            "-I${CMAKE_CURRENT_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/config.yaml"
    DEPENDS hsa_erts.yaml hip_erts.yaml env.yaml "${CMAKE_CURRENT_SOURCE_DIR}/config.yaml"
            "${CMAKE_CURRENT_SOURCE_DIR}/dst_base.yaml"
    COMMENT "Generating raw CTF tracer with barectf...")
install(
    FILES "${CMAKE_CURRENT_BINARY_DIR}/metadata"
    DESTINATION "${METADATA_STREAM_FILE_DIR}"
    COMPONENT plugins)
