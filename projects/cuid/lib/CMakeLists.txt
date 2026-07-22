# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
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

message("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
message("                   Cmake amdcuid library                           ")
message("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")

message("")
message("Build Configuration:")
message("-----------BuildType: " ${CMAKE_BUILD_TYPE})
message("------------Compiler: " ${CMAKE_CXX_COMPILER})
message("-------------Version: " ${CMAKE_CXX_COMPILER_VERSION})
message("------Install Prefix: " ${CMAKE_INSTALL_PREFIX})
message("-CMake inst. Bindir : " ${CMAKE_INSTALL_BINDIR})
message("--------Proj Src Dir: " ${PROJECT_SOURCE_DIR})
message("--------Proj Bld Dir: " ${PROJECT_BINARY_DIR})
message("--------Proj Lib Dir: " ${PROJECT_BINARY_DIR}/lib)
message("--------Proj Exe Dir: " ${PROJECT_BINARY_DIR}/bin)
message("")

# Configure source and include directories
set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
set(INC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)

# When built as part of the top-level project, OpenSSL is already resolved
# (either from the system or via FetchContent). QUIET prevents a redundant
# hard failure when targets are already available through FetchContent.
find_package(OpenSSL QUIET)

set(CUID_SOURCES
    ${SRC_DIR}/cuid.cc
    ${SRC_DIR}/cuid_util.cc
    ${SRC_DIR}/cuid_device.cc
    ${SRC_DIR}/cuid_device_manager.cc
    ${SRC_DIR}/cuid_cpu.cc
    ${SRC_DIR}/cuid_gpu.cc
    ${SRC_DIR}/cuid_nic.cc
    ${SRC_DIR}/cuid_npu.cc
    ${SRC_DIR}/cuid_platform.cc
    ${SRC_DIR}/cuid_file.cc
    ${SRC_DIR}/hmac.cc
    ${SRC_DIR}/pci_util.cc
    ${SRC_DIR}/acpi_parser.cc
    ${SRC_DIR}/smbios_util.cc
)

set(CUID_HEADERS
    ${INC_DIR}/amd_cuid.h
    ${SRC_DIR}/cuid_util.h
    ${SRC_DIR}/cuid_device.h
    ${SRC_DIR}/cuid_device_manager.h
    ${SRC_DIR}/cuid_cpu.h
    ${SRC_DIR}/cuid_gpu.h
    ${SRC_DIR}/cuid_nic.h
    ${SRC_DIR}/cuid_npu.h
    ${SRC_DIR}/cuid_platform.h
    ${SRC_DIR}/cuid_file.h
    ${SRC_DIR}/hmac.h
    ${SRC_DIR}/ipc_protocol.h
    ${SRC_DIR}/pci_util.h
    ${SRC_DIR}/acpi_parser.h
    ${SRC_DIR}/smbios_util.h
    ${SRC_DIR}/cuid_internal.h
)

add_library(amdcuid_static STATIC ${CUID_SOURCES} ${CUID_HEADERS})
add_library(amdcuid_shared SHARED ${CUID_SOURCES} ${CUID_HEADERS})

target_include_directories(
    amdcuid_static
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_include_directories(
    amdcuid_shared
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# Pass the version-independent shared config directory to the library at compile
# time. All installed versions of the library must agree on this path so that
# the HMAC key is shared and CUIDs remain stable across version upgrades.
target_compile_definitions(
    amdcuid_static
    PRIVATE AMDCUID_CONFIG_DIR="${AMDCUID_SHARED_CONFIG_DIR}"
)
target_compile_definitions(
    amdcuid_shared
    PRIVATE AMDCUID_CONFIG_DIR="${AMDCUID_SHARED_CONFIG_DIR}"
)

if(WIN32)
    target_link_libraries(amdcuid_shared PRIVATE bcrypt)
    target_link_libraries(amdcuid_static PUBLIC bcrypt)
else()
    target_link_libraries(amdcuid_shared PRIVATE OpenSSL::Crypto)
    target_link_libraries(amdcuid_static PUBLIC OpenSSL::Crypto)
endif()

# Shared library
install(
    TARGETS amdcuid_shared
    RUNTIME
        DESTINATION
            ${CMAKE_INSTALL_BINDIR} # Windows DLLs go here
    LIBRARY
        DESTINATION
            ${CMAKE_INSTALL_LIBDIR} # Linux .so
    ARCHIVE
        DESTINATION
            ${CMAKE_INSTALL_LIBDIR} # Static libs & Windows import libs
        COMPONENT ${LIB_COMPONENT}
)

# Static library
install(
    TARGETS amdcuid_static
    RUNTIME
        DESTINATION
            ${CMAKE_INSTALL_BINDIR} # Windows DLLs go here
    ARCHIVE
        DESTINATION
            ${CMAKE_INSTALL_LIBDIR} # Static libs & Windows import libs
        COMPONENT ${LIB_COMPONENT}
)

# Install public headers
set(PUBLIC_HEADERS ${INC_DIR}/amd_cuid.h)
install(
    FILES ${PUBLIC_HEADERS}
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${AMDCUID}
    COMPONENT ${LIB_COMPONENT}
)

if(UNIX AND NOT APPLE)
    configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/amdcuid_setup_hmac.sh.in
        ${CMAKE_CURRENT_BINARY_DIR}/amdcuid_setup_hmac.sh
        @ONLY
    )
    install(
        PROGRAMS ${CMAKE_CURRENT_BINARY_DIR}/amdcuid_setup_hmac.sh
        DESTINATION ${CMAKE_INSTALL_DATADIR}/${AMDCUID}
        COMPONENT ${LIB_COMPONENT}
    )
endif()

message("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
message("                  Finished Cmake amdcuid library                   ")
message("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
