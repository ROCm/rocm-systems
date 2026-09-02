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

# An OBJECT library first, and the archive out of its objects, so a build that
# has this project as a subdirectory can absorb the objects into its own binary
# instead of linking the archive. That is what lets AMD SMI carry CUID with no
# amdcuid package to find, no artifact to register with a superbuild, and no
# libamdcuid.a in anyone's -dev package: the same trick this library already
# uses on rocm-sha256 below, for the same reason.
add_library(amdcuid_objs OBJECT ${CUID_SOURCES} ${CUID_HEADERS})

add_library(amdcuid_static STATIC $<TARGET_OBJECTS:amdcuid_objs>)

# Namespaced alias, so an in-tree consumer links the same name an installed
# consumer does. A "::" name also makes CMake diagnose a typo as a missing
# package rather than treating it as a bare -l flag.
add_library(amdcuid::amdcuid ALIAS amdcuid_static)

# Must match the alias, or an installed consumer links amdcuid::amdcuid_static
# while an in-tree one links amdcuid::amdcuid.
set_property(TARGET amdcuid_static PROPERTY EXPORT_NAME amdcuid)

# Both, and non-negotiable: the objects end up inside libamd_smi.so, and a
# non-PIC object in a shared library fails to link on x86-64.
set_property(TARGET amdcuid_objs PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET amdcuid_static PROPERTY POSITION_INDEPENDENT_CODE ON)

# Hidden by default. These objects are absorbed into a consumer's shared
# library, whose exported set is its own business; nothing here should widen it.
# amd_smi applies a version script as well, so this is the belt to that braces,
# and it is the half that also works for a consumer that has no version script.
set_target_properties(
    amdcuid_objs
    PROPERTIES C_VISIBILITY_PRESET hidden CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN ON
)

# The public surface is include/ and nothing else, in the build tree exactly as
# in the install tree. With this directory itself PUBLIC, an in-tree consumer
# could write #include "src/hmac.h" and compile, then fail once installed
# because src/ is not shipped.
#
# The library's own sources say #include "include/amd_cuid.h" and #include
# "src/hmac.h", both relative to this directory, so it stays on the private
# include path. The install path is relative on purpose: an absolute one baked
# into the exported target makes the package non-relocatable.
foreach(_amdcuid_target IN ITEMS amdcuid_objs amdcuid_static)
    target_include_directories(
        ${_amdcuid_target}
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/${AMDCUID}>
        PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/src
    )

    # Pass the version-independent shared config directory to the library at
    # compile time. All installed versions of the library must agree on this
    # path so that the HMAC key is shared and CUIDs remain stable across
    # version upgrades.
    target_compile_definitions(
        ${_amdcuid_target}
        PRIVATE AMDCUID_CONFIG_DIR="${AMDCUID_SHARED_CONFIG_DIR}"
    )
endforeach()

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
foreach(_amdcuid_target IN ITEMS amdcuid_objs amdcuid_static)
    target_include_directories(
        ${_amdcuid_target}
        PUBLIC $<BUILD_INTERFACE:${CUID_ROCM_SHA256_DIR}/include>
    )
endforeach()

# The objects a subdirectory consumer absorbs are amdcuid's own; the SHA-256
# ones are separate and it has to take them too, so name the pair here rather
# than leaving each consumer to rediscover the split.
set(AMDCUID_IN_TREE_OBJECTS
    $<TARGET_OBJECTS:amdcuid_objs> $<TARGET_OBJECTS:rocm-sha256>
    CACHE INTERNAL
    "Objects an in-tree consumer absorbs to carry libamdcuid without linking it"
)

# The library uses std::dynamic_pointer_cast over its device hierarchy, so it
# needs RTTI. A parent project may compile with -fno-rtti globally --
# projects/amdsmi does -- and that arrives through CMAKE_CXX_FLAGS, ahead of
# anything a target sets, so it has to be countermanded where these sources are
# actually compiled. GCC and Clang take the last -f flag on the line.
if(NOT MSVC)
    target_compile_options(amdcuid_objs PRIVATE -frtti)
    target_compile_options(amdcuid_static PRIVATE -frtti)
endif()

# The only external dependency left is the platform CSPRNG.
if(WIN32)
    target_link_libraries(amdcuid_static PUBLIC bcrypt)
endif()

if(AMDCUID_INSTALL)
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
endif()

message("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
message("                  Finished Cmake amdcuid library                   ")
message("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
