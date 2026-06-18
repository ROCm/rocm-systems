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
# When built as a subdirectory of the main project, the amd_smi target is already
# available so find_package is not needed.  When built standalone (outside the
# main project), search the build tree, the ROCm install prefix, or ROCM_DIR.
if(NOT TARGET amd_smi AND NOT TARGET amd_smi_static)
    list(APPEND CMAKE_PREFIX_PATH ${CMAKE_BINARY_DIR} ../../../ ${ROCM_DIR})
    list(APPEND CMAKE_LIBRARY_PATH ${ROCM_DIR}/${CMAKE_INSTALL_LIBDIR})
    find_package(amd_smi CONFIG REQUIRED)
endif()

# Set the link target based on which targets are available
# Prefer static library for examples
if(TARGET amd_smi_static)
    set(_AMDSMI_LINK_TARGET amd_smi_static)
elseif(TARGET amd_smi)
    set(_AMDSMI_LINK_TARGET amd_smi)
else()
    message(FATAL_ERROR "Neither amd_smi nor amd_smi_static target is available")
endif()

find_package(Threads REQUIRED)

message("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
message("                    Finished Cmake Example                         ")
message("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")

# compile example files but do not install
# this is only useful if running from build directory
set(SMI_DRM_EXAMPLE_EXE "amd_smi_drm_ex")
add_executable(${SMI_DRM_EXAMPLE_EXE} "amd_smi_drm_example.cc")
target_link_libraries(${SMI_DRM_EXAMPLE_EXE} ${_AMDSMI_LINK_TARGET})
target_compile_definitions(${SMI_DRM_EXAMPLE_EXE} PUBLIC ENABLE_ESMI_LIB)

set(SMI_NODRM_EXAMPLE_EXE "amd_smi_nodrm_ex")
add_executable(${SMI_NODRM_EXAMPLE_EXE} "amd_smi_nodrm_example.cc")
target_link_libraries(${SMI_NODRM_EXAMPLE_EXE} ${_AMDSMI_LINK_TARGET})
target_compile_definitions(${SMI_NODRM_EXAMPLE_EXE} PUBLIC ENABLE_ESMI_LIB)

set(SMI_PARTITION_EXAMPLE_EXE "amd_smi_partition_ex")
add_executable(${SMI_PARTITION_EXAMPLE_EXE} "amd_smi_partition_example.cc")
target_link_libraries(${SMI_PARTITION_EXAMPLE_EXE} ${_AMDSMI_LINK_TARGET})

if(ENABLE_ESMI_LIB)
    set(ESMI_SAMPLE_EXE "amd_smi_esmi_ex")
    add_executable(${ESMI_SAMPLE_EXE} "amdsmi_esmi_intg_example.cc")
    target_link_libraries(${ESMI_SAMPLE_EXE} ${_AMDSMI_LINK_TARGET})
    target_compile_definitions(${ESMI_SAMPLE_EXE} PUBLIC ENABLE_ESMI_LIB)
endif()

add_executable(ainic amd_smi_nic.cc)
target_compile_definitions(ainic PRIVATE ENABLE_ESMI_LIB)
# When amd-smi-lib is installed, NIC headers are under /opt/rocm/include.
# When building from source (no install), fall back to the source tree.
target_include_directories(
    ainic
    PRIVATE ${ROCM_DIR}/include ${ROCM_DIR}/include/amd_smi/impl/nic ${CMAKE_SOURCE_DIR}/src/nic/ai-nic
)
target_link_directories(ainic PRIVATE ${ROCM_DIR}/lib)
target_link_libraries(ainic PRIVATE ${_AMDSMI_LINK_TARGET})

set(SMI_FABRIC_EXAMPLE_EXE "amd_smi_fabric_ex")
add_executable(${SMI_FABRIC_EXAMPLE_EXE} "amd_smi_fabric_example.cc")
target_link_libraries(${SMI_FABRIC_EXAMPLE_EXE} PRIVATE ${_AMDSMI_LINK_TARGET})
