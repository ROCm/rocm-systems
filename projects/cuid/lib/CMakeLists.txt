# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

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
    ${SRC_DIR}/sha256.cc
    ${SRC_DIR}/pci_util.cc
    ${SRC_DIR}/acpi_parser.cc
    ${SRC_DIR}/smbios_util.cc
    ${SRC_DIR}/gim_util.cc
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
    ${SRC_DIR}/sha256.h
    ${SRC_DIR}/ipc_protocol.h
    ${SRC_DIR}/pci_util.h
    ${SRC_DIR}/acpi_parser.h
    ${SRC_DIR}/smbios_util.h
    ${SRC_DIR}/cuid_internal.h
    ${SRC_DIR}/gim_util.h
)

add_library(amdcuid_static STATIC ${CUID_SOURCES} ${CUID_HEADERS})

set_property(TARGET amdcuid_static PROPERTY POSITION_INDEPENDENT_CODE ON)

target_include_directories(amdcuid_static PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

# Pass the version-independent shared config directory to the library at compile
# time. All installed versions of the library must agree on this path so that
# the HMAC key is shared and CUIDs remain stable across version upgrades.
target_compile_definitions(amdcuid_static PRIVATE AMDCUID_CONFIG_DIR="${AMDCUID_SHARED_CONFIG_DIR}")

# SHA-256/HMAC comes from src/sha256.cc, so the only external dependency left
# is the platform CSPRNG: bcrypt on Windows, getrandom(2)//dev/urandom on POSIX.
if(WIN32)
    target_link_libraries(amdcuid_static PUBLIC bcrypt)
endif()

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
install(FILES ${PUBLIC_HEADERS} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${AMDCUID} COMPONENT ${LIB_COMPONENT})

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
