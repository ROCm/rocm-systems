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
    ${SRC_DIR}/ipc_protocol.h
    ${SRC_DIR}/pci_util.h
    ${SRC_DIR}/acpi_parser.h
    ${SRC_DIR}/smbios_util.h
    ${SRC_DIR}/cuid_internal.h
    ${SRC_DIR}/gim_util.h
)

add_library(amdcuid_static STATIC ${CUID_SOURCES} ${CUID_HEADERS})

# Namespaced alias, so an in-tree consumer links the same name an installed
# consumer does. A "::" name also makes CMake diagnose a typo as a missing
# package rather than treating it as a bare -l flag.
add_library(amdcuid::amdcuid ALIAS amdcuid_static)

# Must match the alias, or an installed consumer links amdcuid::amdcuid_static
# while an in-tree one links amdcuid::amdcuid.
set_property(TARGET amdcuid_static PROPERTY EXPORT_NAME amdcuid)

set_property(TARGET amdcuid_static PROPERTY POSITION_INDEPENDENT_CODE ON)

# The public surface is include/ and nothing else, in the build tree exactly as
# in the install tree. With this directory itself PUBLIC, an in-tree consumer
# could write #include "src/hmac.h" and compile, then fail once installed
# because src/ is not shipped.
#
# The library's own sources say #include "include/amd_cuid.h" and #include
# "src/hmac.h", both relative to this directory, so it stays on the private
# include path. The install path is relative on purpose: an absolute one baked
# into the exported target makes the package non-relocatable.
target_include_directories(
    amdcuid_static
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/${AMDCUID}>
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# Pass the version-independent shared config directory to the library at compile
# time. All installed versions of the library must agree on this path so that
# the HMAC key is shared and CUIDs remain stable across version upgrades.
target_compile_definitions(amdcuid_static PRIVATE AMDCUID_CONFIG_DIR="${AMDCUID_SHARED_CONFIG_DIR}")

# SHA-256/HMAC comes from shared/sha256, built from source. The cache variable
# is project-prefixed so that a build which configures cuid and another
# rocm-systems project together does not have them fight over one entry.
set(CUID_ROCM_SHA256_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/../../../shared/sha256"
    CACHE PATH
    "Path to the shared SHA-256 component"
)
if(NOT TARGET rocm-sha256)
    if(NOT EXISTS "${CUID_ROCM_SHA256_DIR}/CMakeLists.txt")
        message(
            FATAL_ERROR
            "Shared SHA-256 not found at ${CUID_ROCM_SHA256_DIR}. Set "
            "CUID_ROCM_SHA256_DIR when building cuid outside the rocm-systems tree."
        )
    endif()
    add_subdirectory(${CUID_ROCM_SHA256_DIR} ${CMAKE_CURRENT_BINARY_DIR}/rocm-sha256)
endif()

# Absorb the shared SHA-256 objects rather than linking the target. amdcuid_static
# is in an export set, and a link dependency travels with it into that set even
# when PRIVATE, so exporting a link to rocm-sha256, a build-tree helper that is
# never installed, names a target the consumer does not have.
# The include dir is PUBLIC and BUILD_INTERFACE so in-tree targets that link
# amdcuid_static and include rocm/sha2/sha256.h directly can find it. Those
# headers are never installed and sha256.h is not part of the installed API.
target_sources(amdcuid_static PRIVATE $<TARGET_OBJECTS:rocm-sha256>)
target_include_directories(amdcuid_static PUBLIC $<BUILD_INTERFACE:${CUID_ROCM_SHA256_DIR}/include>)

# The only external dependency left is the platform CSPRNG.
if(WIN32)
    target_link_libraries(amdcuid_static PUBLIC bcrypt)
endif()

# Static library, as part of an export set so that consumers get an imported
# target with its usage requirements attached rather than a bare path they have
# to pair with a hand-written include directory.
install(
    TARGETS amdcuid_static
    EXPORT amdcuidTargets
    RUNTIME
        DESTINATION
            ${CMAKE_INSTALL_BINDIR} # Windows DLLs go here
        COMPONENT ${LIB_COMPONENT}
    ARCHIVE
        DESTINATION
            ${CMAKE_INSTALL_LIBDIR} # Static libs & Windows import libs
        COMPONENT ${LIB_COMPONENT}
)

# Install public headers
set(PUBLIC_HEADERS ${INC_DIR}/amd_cuid.h)
install(FILES ${PUBLIC_HEADERS} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${AMDCUID} COMPONENT ${LIB_COMPONENT})

# CMake package: amdcuid-config.cmake, its version file and the target export.
# This is what lets amd-smi (and anything else) say
#
#     find_package(amdcuid CONFIG)
#     target_link_libraries(foo PRIVATE amdcuid::amdcuid)
#
# instead of find_library() plus a guessed include path: the imported target
# carries the include directory and the link interface together, so a consumer
# cannot get one right and the other wrong.
include(CMakePackageConfigHelpers)

set(AMDCUID_INSTALL_CMAKEDIR ${CMAKE_INSTALL_LIBDIR}/cmake/${AMDCUID})

install(
    EXPORT amdcuidTargets
    FILE amdcuid-targets.cmake
    NAMESPACE amdcuid::
    DESTINATION ${AMDCUID_INSTALL_CMAKEDIR}
    COMPONENT ${LIB_COMPONENT}
)

configure_package_config_file(
    ${PROJECT_SOURCE_DIR}/cmake/amdcuid-config.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/amdcuid-config.cmake
    INSTALL_DESTINATION ${AMDCUID_INSTALL_CMAKEDIR}
)

# SameMajorVersion: the ABI is the public header, and the format it describes is
# frozen within a major version. A consumer asking for 1.2 must not be handed a
# 2.x that renumbered a field.
write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/amdcuid-config-version.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(
    FILES ${CMAKE_CURRENT_BINARY_DIR}/amdcuid-config.cmake ${CMAKE_CURRENT_BINARY_DIR}/amdcuid-config-version.cmake
    DESTINATION ${AMDCUID_INSTALL_CMAKEDIR}
    COMPONENT ${LIB_COMPONENT}
)

# The same export from the build tree, so a superbuild that has not installed
# anything yet can still find_package() this by adding the build directory to
# CMAKE_PREFIX_PATH.
export(EXPORT amdcuidTargets FILE ${CMAKE_CURRENT_BINARY_DIR}/amdcuid-targets.cmake NAMESPACE amdcuid::)

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
