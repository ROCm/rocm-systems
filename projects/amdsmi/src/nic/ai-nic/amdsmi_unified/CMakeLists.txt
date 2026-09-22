# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.16)

# Capture the amd_smi project root before the project() call below opens a new
# project scope, which would repoint PROJECT_SOURCE_DIR at this subdirectory.
# Anchoring header paths here avoids counting "../" levels up to the root.
set(AMD_SMI_ROOT_DIR ${PROJECT_SOURCE_DIR})

project(amdsminic LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(PROJECT_ROOT ${CMAKE_CURRENT_SOURCE_DIR})
set(NIC_INTERFACE_HEADERS_DIR ${AMD_SMI_ROOT_DIR}/include/amd_smi/impl/nic/amdsmi_unified/interface)
set(NIC_INCLUDE_DIR ${PROJECT_ROOT}/inc ${NIC_INTERFACE_HEADERS_DIR})
set(NIC_SOURCE_DIR ${PROJECT_ROOT}/src)
set(NIC_BUILD_DIR ${CMAKE_BINARY_DIR}/build)
set(NIC_INTERFACE_DIR ${CMAKE_BINARY_DIR}/interface)

set(NIC_DEFAULT_CXX_FLAGS
    "-Wall -Wextra -Werror -Wno-missing-field-initializers -Wno-array-bounds -Wmissing-declarations -Werror=conversion -Wshift-negative-value -fPIC"
)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${NIC_DEFAULT_CXX_FLAGS}")

# Linker hardening flags - applied only at link time, not during compilation.
# These were previously in CMAKE_CXX_FLAGS which caused -Wunused-command-line-argument
# errors with clang-based compilers (e.g. amdclang).
set(NIC_LINKER_FLAGS "-Wl,-z,relro,-z,noexecstack,-z,noexecheap -Wl,--strip-debug -Wl,--strip-all")
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${NIC_LINKER_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${NIC_LINKER_FLAGS}")

file(GLOB CPP_SRCS "${NIC_SOURCE_DIR}/*.cpp")
add_library(amdsminic STATIC ${CPP_SRCS})
set_target_properties(amdsminic PROPERTIES ARCHIVE_OUTPUT_DIRECTORY ${NIC_BUILD_DIR})
target_include_directories(
    amdsminic
    PUBLIC
        "$<BUILD_INTERFACE:${PROJECT_ROOT}/inc>"
        "$<BUILD_INTERFACE:${NIC_INTERFACE_HEADERS_DIR}>"
        "$<BUILD_INTERFACE:${NIC_INTERFACE_DIR}>"
        "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/amd_smi/impl/nic/amdsmi_unified/inc>"
        "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/amd_smi/impl/nic/amdsmi_unified/interface>"
)
