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
message("--------Proj Src Dir: " ${PROJECT_SOURCE_DIR})
message("--------Proj Bld Dir: " ${PROJECT_BINARY_DIR})
message("--------Proj Lib Dir: " ${PROJECT_BINARY_DIR}/lib)
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
    ${SRC_DIR}/hmac.h
    ${SRC_DIR}/pci_util.h
    ${SRC_DIR}/acpi_parser.h
    ${SRC_DIR}/smbios_util.h
    ${SRC_DIR}/cuid_internal.h
    ${SRC_DIR}/gim_util.h
)

# An OBJECT library first, and the archive out of its objects, so a build that
# has this project as a subdirectory (AMD SMI) can absorb the objects into its
# own binary instead of linking the archive, as is done with rocm-sha256 below.
add_library(amdcuid_objs OBJECT ${CUID_SOURCES} ${CUID_HEADERS})

add_library(amdcuid_static STATIC $<TARGET_OBJECTS:amdcuid_objs>)

# Namespaced alias, so an in-tree consumer links the same name an installed
# consumer does. A "::" name also makes CMake diagnose a typo as a missing
# package rather than treating it as a bare -l flag.
add_library(amdcuid::amdcuid ALIAS amdcuid_static)

# Must match the alias, or an installed consumer links amdcuid::amdcuid_static
# while an in-tree one links amdcuid::amdcuid.
set_property(TARGET amdcuid_static PROPERTY EXPORT_NAME amdcuid)

# The objects end up inside libamd_smi.so, and a non-PIC object in a shared
# library fails to link on x86-64.
set_property(TARGET amdcuid_objs PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET amdcuid_static PROPERTY POSITION_INDEPENDENT_CODE ON)

# Hidden by default: these objects are absorbed into a consumer's shared
# library and must not widen its exported set, version script or not.
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
target_sources(amdcuid_static PRIVATE $<BUILD_INTERFACE:$<TARGET_OBJECTS:rocm-sha256-objects>>)
foreach(_amdcuid_target IN ITEMS amdcuid_objs amdcuid_static)
    target_include_directories(${_amdcuid_target} PUBLIC $<BUILD_INTERFACE:${CUID_ROCM_SHA256_DIR}/include>)
endforeach()

# A subdirectory consumer must absorb the SHA-256 objects along with amdcuid's.
set(AMDCUID_IN_TREE_OBJECTS
    $<TARGET_OBJECTS:amdcuid_objs>
    $<TARGET_OBJECTS:rocm-sha256-objects>
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

# The only external dependency is the platform CSPRNG.
if(WIN32)
    target_link_libraries(amdcuid_static PUBLIC bcrypt)
endif()

# std::mutex and std::call_once need the thread library on glibc before 2.34.
find_package(Threads REQUIRED)
target_link_libraries(amdcuid_static PUBLIC Threads::Threads)

if(AMDCUID_INSTALL)
    # Static library, as part of an export set so that consumers get an imported
    # target with its usage requirements attached rather than a bare path they have
    # to pair with a hand-written include directory.
    install(
        TARGETS amdcuid_static
        EXPORT amdcuidTargets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT ${LIB_COMPONENT}
    )
    install(TARGETS rocm-sha256-objects EXPORT amdcuidTargets COMPONENT ${LIB_COMPONENT})

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

    # SameMajorVersion: a major version bump marks an ABI break in the public
    # header. A consumer asking for 1.2 must not be handed a 2.x that renumbered a
    # field. Derived CUID values are not covered and can change in a minor release.
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
endif()

message("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
message("                  Finished Cmake amdcuid library                   ")
message("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
