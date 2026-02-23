cmake_minimum_required(VERSION 3.20)

option(ENABLE_ESMI_LIB "Build ESMI Library" ON)
option(CMAKE_VERBOSE_MAKEFILE "Enable verbose output" ON)
option(CMAKE_EXPORT_COMPILE_COMMANDS "Export compile commands for linters and autocompleters" ON)

# Compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -m64 -msse -msse2")

if("${CMAKE_BUILD_TYPE}" STREQUAL Release)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ggdb -O0 -DDEBUG")
endif()

set(CMAKE_CXX_STANDARD 17 CACHE STRING "The C++ standard to use")
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(CMAKE_COMPILER_IS_GNUCC AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 5.4.0)
    message("Compiler version is " ${CMAKE_CXX_COMPILER_VERSION})
    message(FATAL_ERROR "Require at least gcc-5.4.0")
endif()

project(amd_smi_example)

# required variables
if(DEFINED ENV{ROCM_PATH})
    set(ROCM_DIR "$ENV{ROCM_PATH}" CACHE STRING "ROCm directory.")
else()
    set(ROCM_DIR "/opt/rocm" CACHE STRING "ROCm directory.")
endif()

include(GNUInstallDirs)

# add package search paths
# ../../../ should resolve to /opt/rocm or another rocm install path
# fall back to ROCM_DIR
list(APPEND CMAKE_PREFIX_PATH ../../../ ${ROCM_DIR})
list(APPEND CMAKE_LIBRARY_PATH ${ROCM_DIR}/${CMAKE_INSTALL_LIBDIR})

find_package(amd_smi CONFIG REQUIRED)

message("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
message("                    Finished Cmake Example                         ")
message("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")

# compile example files but do not install
# this is only useful if running from build directory
set(SMI_DRM_EXAMPLE_EXE "amd_smi_drm_ex")
add_executable(${SMI_DRM_EXAMPLE_EXE} "amd_smi_drm_example.cc")
target_link_libraries(${SMI_DRM_EXAMPLE_EXE} amd_smi)

set(SMI_NODRM_EXAMPLE_EXE "amd_smi_nodrm_ex")
add_executable(${SMI_NODRM_EXAMPLE_EXE} "amd_smi_nodrm_example.cc")
target_link_libraries(${SMI_NODRM_EXAMPLE_EXE} amd_smi)

if(ENABLE_ESMI_LIB)
    set(ESMI_SAMPLE_EXE "amd_smi_esmi_ex")
    add_executable(${ESMI_SAMPLE_EXE} "amdsmi_esmi_intg_example.cc")
    target_link_libraries(${ESMI_SAMPLE_EXE} amd_smi)
    target_compile_definitions(${ESMI_SAMPLE_EXE} PUBLIC ENABLE_ESMI_LIB)
endif()
