#
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

cmake_minimum_required(VERSION 3.16)

# Capture the amd_smi project root before the project() call below opens a new
# project scope, which would repoint PROJECT_SOURCE_DIR at this subdirectory.
# Anchoring header paths here avoids counting "../" levels up to the root.
set(AMD_SMI_ROOT_DIR ${PROJECT_SOURCE_DIR})

project(amdsminic LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Check for libnl-3 (optional, for netlink transport backend)
find_package(PkgConfig QUIET)
if(PKG_CONFIG_FOUND)
    pkg_check_modules(LIBNL3 libnl-3.0 libnl-genl-3.0)
endif()

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

# Sources are enumerated explicitly (matching the top-level project's SRC_LIST
# style) so adding a file is a deliberate act, not a silent GLOB pickup.
set(CPP_SRCS
    "${NIC_SOURCE_DIR}/smi_ethtool_ioctl.cpp"
    "${NIC_SOURCE_DIR}/smi_nic.cpp"
    "${NIC_SOURCE_DIR}/smi_nic_interface.cpp"
    "${NIC_SOURCE_DIR}/smi_nic_subsystem.cpp"
    "${NIC_SOURCE_DIR}/smi_nic_system.cpp"
    "${NIC_SOURCE_DIR}/smi_nic_telemetry.cpp"
    "${NIC_SOURCE_DIR}/smi_nic_transport_ioctl.cpp"
    "${NIC_SOURCE_DIR}/smi_nic_vpd.cpp"
    "${NIC_SOURCE_DIR}/smi_sysfs.cpp"
    "${NIC_SOURCE_DIR}/vendors/vendor_registry.cpp"
    "${NIC_SOURCE_DIR}/vendors/broadcom/broadcom_subsystem.cpp"
    "${NIC_SOURCE_DIR}/vendors/pensando/pensando_subsystem.cpp"
)

# Netlink transport backend requires libnl-3; gated by explicit membership, not a
# filename regex, so a future non-libnl file with "netlink" in its name is not
# excluded by accident.
set(CPP_SRCS_NETLINK
    "${NIC_SOURCE_DIR}/netlink_ethtool.cpp"
    "${NIC_SOURCE_DIR}/netlink_generic.cpp"
    "${NIC_SOURCE_DIR}/netlink_wrapper.cpp"
    "${NIC_SOURCE_DIR}/smi_devlink_netlink.cpp"
    "${NIC_SOURCE_DIR}/smi_nic_transport_netlink.cpp"
)

if(LIBNL3_FOUND)
    list(APPEND CPP_SRCS ${CPP_SRCS_NETLINK})
    message(STATUS "libnl-3 found (${LIBNL3_VERSION}): building with netlink transport backend")
else()
    message(STATUS "libnl-3 not found: building without netlink transport backend")
endif()

add_library(amdsminic STATIC ${CPP_SRCS})
set_target_properties(amdsminic PROPERTIES ARCHIVE_OUTPUT_DIRECTORY ${NIC_BUILD_DIR})

# Vendor plugins live under src/vendors/<name>/; expose the vendors root so the
# registry and plugin translation units resolve their headers.
target_include_directories(amdsminic PRIVATE "${NIC_SOURCE_DIR}/vendors")

# Add compile definition if libnl-3 is available
if(LIBNL3_FOUND)
    target_compile_definitions(amdsminic PRIVATE HAVE_LIBNL3)
    target_include_directories(amdsminic PRIVATE ${LIBNL3_INCLUDE_DIRS})
    target_link_libraries(amdsminic PRIVATE ${LIBNL3_LIBRARIES})
endif()

target_include_directories(
    amdsminic
    PUBLIC
        "$<BUILD_INTERFACE:${PROJECT_ROOT}/inc>"
        "$<BUILD_INTERFACE:${NIC_INTERFACE_HEADERS_DIR}>"
        "$<BUILD_INTERFACE:${NIC_INTERFACE_DIR}>"
        "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/amd_smi/impl/nic/amdsmi_unified/inc>"
        "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/amd_smi/impl/nic/amdsmi_unified/interface>"
)
