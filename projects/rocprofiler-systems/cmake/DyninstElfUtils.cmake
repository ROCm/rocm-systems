# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# ======================================================================================
# DyninstElfUtils.cmake
#
# Configure the elfutils dependency (libdw, libelf) for Dyninst and the rest of
# rocprofiler-systems.
#
# Discovery uses a single path in all three supported build modes:
#
#   1. ROCPROFSYS_BUILD_ELFUTILS=ON (from source). Build elfutils via
#      ExternalProject, then pre-create three imported targets:
#        - LibDW::LibDW / LibElf::LibElf for rocprofiler-systems' own
#          consumers (rocprofiler-systems-elfutils interface lib).
#        - Dyninst::ElfUtils (interface wrapping the two above) so that
#          Dyninst's tpls/DyninstElfUtils.cmake short-circuits its own
#          find_package(Elfutils) via its `if(TARGET Dyninst::ElfUtils)`
#          check. This is the only reliable mode-1 short-circuit for
#          Dyninst — its CMakeLists.txt prepends its own cmake/Modules/
#          at position 0 of CMAKE_MODULE_PATH, so our FindLibDW / FindLibElf
#          shims never run for Dyninst's own find_package(LibDW|LibElf)
#          calls.
#
#   2. ROCPROFSYS_BUILD_ELFUTILS=OFF, system install. Shim's pkg-config
#      fallback discovers /usr/lib libdw.so / libelf.so via libdw.pc /
#      libelf.pc.
#
#   3. ROCPROFSYS_BUILD_ELFUTILS=OFF, vendored sysdep (TheRock). TheRock
#      ships libdw.pc / libelf.pc under lib/rocm_sysdeps/lib/pkgconfig/
#      and propagates PKG_CONFIG_PATH to this subproject's configure
#      environment, so the shim's pkg-config fallback resolves them the
#      same way as mode 2.
#
# Exported to consumers
# ---------------------
#   LibDW::LibDW, LibElf::LibElf  — imported targets (preferred)
#   LibDW_*, LibElf_*             — legacy variables (Dyninst's FindElfutils.cmake)
#   rocprofiler-systems-elfutils  — INTERFACE target wrapping both
#
# Optional: LibDebuginfod::LibDebuginfod via ENABLE_DEBUGINFOD.
# ======================================================================================

include_guard(GLOBAL)

if(NOT UNIX)
    return()
endif()

# libdw is not thread-safe before 0.178.
set(_min_version 0.178)
set(ElfUtils_MIN_VERSION
    ${_min_version}
    CACHE STRING
    "Minimum acceptable elfutils version"
)
if(${ElfUtils_MIN_VERSION} VERSION_LESS ${_min_version})
    rocprofiler_systems_message(
        FATAL_ERROR
        "Requested version ${ElfUtils_MIN_VERSION} is less than minimum supported version (${_min_version})"
    )
endif()

# --------------------------------------------------------------------------------------
# Mode 1: build elfutils from source
# --------------------------------------------------------------------------------------
if(ROCPROFSYS_BUILD_ELFUTILS)
    if(
        NOT (${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU")
        OR NOT (${CMAKE_C_COMPILER_ID} STREQUAL "GNU")
    )
        rocprofiler_systems_message(FATAL_ERROR
            "ElfUtils will only build with the GNU compiler"
        )
    endif()

    rocprofiler_systems_add_cache_option(
        ELFUTILS_DOWNLOAD_VERSION "Version of elfutils to download and install" STRING
        "0.195"
    )
    # Honor legacy user override (-DElfUtils_DOWNLOAD_VERSION=...) if provided.
    if(
        DEFINED ElfUtils_DOWNLOAD_VERSION
        AND NOT "${ElfUtils_DOWNLOAD_VERSION}" STREQUAL ""
    )
        set(ELFUTILS_DOWNLOAD_VERSION "${ElfUtils_DOWNLOAD_VERSION}")
    endif()

    if(${ELFUTILS_DOWNLOAD_VERSION} VERSION_LESS ${ElfUtils_MIN_VERSION})
        rocprofiler_systems_message(
            FATAL_ERROR
            "elfutils download version is set to ${ELFUTILS_DOWNLOAD_VERSION} but elfutils minimum version is set to ${ElfUtils_MIN_VERSION}"
        )
    endif()

    rocprofiler_systems_message(
        STATUS
        "Building elfutils(${ELFUTILS_DOWNLOAD_VERSION}) from source"
    )

    set(_eu_root ${TPL_STAGING_PREFIX}/elfutils)
    set(_eu_libdw "${_eu_root}/lib/libdw${CMAKE_SHARED_LIBRARY_SUFFIX}")
    set(_eu_libelf "${_eu_root}/lib/libelf${CMAKE_SHARED_LIBRARY_SUFFIX}")

    file(MAKE_DIRECTORY "${_eu_root}/lib")
    file(MAKE_DIRECTORY "${_eu_root}/include")

    # Backport elfutils commit 7508696d (released in 0.192) to fix GCC 15
    # -Werror=unterminated-string-initialization in the i386/x86_64 register
    # tables. The patch invocation is wrapped in apply_patch_idempotent.cmake
    # because CMake regenerates *-patch-info.txt on every reconfigure,
    # retriggering the patch step against already-patched source - vanilla
    # `patch` aborts in that case.
    set(_eu_patch_args)
    if(ELFUTILS_DOWNLOAD_VERSION VERSION_LESS 0.192)
        find_program(PATCH_EXECUTABLE NAMES patch REQUIRED)
        set(_eu_patch_args
            PATCH_COMMAND
            ${CMAKE_COMMAND}
            -DSRC=<SOURCE_DIR>
            -DPATCH=${CMAKE_CURRENT_LIST_DIR}/elfutils-0.188-gcc15-regs.patch
            -DPATCH_EXE=${PATCH_EXECUTABLE}
            -P
            ${CMAKE_CURRENT_LIST_DIR}/apply_patch_idempotent.cmake
        )
    endif()

    include(ExternalProject)
    ExternalProject_Add(
        rocprofiler-systems-elfutils-build
        PREFIX ${_eu_root}
        URL
            ${ElfUtils_DOWNLOAD_URL}
            "https://sourceware.org/elfutils/ftp/${ELFUTILS_DOWNLOAD_VERSION}/elfutils-${ELFUTILS_DOWNLOAD_VERSION}.tar.bz2"
            "https://mirrors.kernel.org/sourceware/elfutils/${ELFUTILS_DOWNLOAD_VERSION}/elfutils-${ELFUTILS_DOWNLOAD_VERSION}.tar.bz2"
        BUILD_IN_SOURCE 1
        ${_eu_patch_args}
        CONFIGURE_COMMAND
            ${CMAKE_COMMAND} -E env CC=${CMAKE_C_COMPILER}
            CFLAGS=-fPIC\ -O3\ -Wno-error=maybe-uninitialized CXX=${CMAKE_CXX_COMPILER}
            CXXFLAGS=-fPIC\ -O3\ -Wno-error=maybe-uninitialized
            [=[LDFLAGS=-Wl,-rpath='$$ORIGIN' -pthread]=] <SOURCE_DIR>/configure
            --enable-install-elfh --prefix=${_eu_root} --disable-libdebuginfod
            --disable-debuginfod --enable-thread-safety --disable-nls
            ${ElfUtils_CONFIG_OPTIONS} --libdir=${_eu_root}/lib
        BUILD_COMMAND make install
        BUILD_BYPRODUCTS ${_eu_libdw} ${_eu_libelf}
        INSTALL_COMMAND ""
    )

    install(
        DIRECTORY ${_eu_root}/lib/
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/${PROJECT_NAME}
        FILES_MATCHING
        PATTERN "*${CMAKE_SHARED_LIBRARY_SUFFIX}*"
    )

    # Pre-create LibDW::LibDW / LibElf::LibElf for rocprofiler-systems' own
    # consumers (the rocprofiler-systems-elfutils interface lib). Each target
    # carries an add_dependencies edge to the ExternalProject build step so
    # anything that links them waits for the .so to exist.
    #
    # Dyninst does NOT consume these — see the Dyninst::ElfUtils block below
    # for how Dyninst's subtree is short-circuited.
    set(LibDW_FOUND TRUE)
    set(LibDW_INCLUDE_DIRS "${_eu_root}/include")
    set(LibDW_LIBRARIES "${_eu_libdw}")
    set(LibDW_VERSION "${ELFUTILS_DOWNLOAD_VERSION}")
    if(NOT TARGET LibDW::LibDW)
        add_library(LibDW::LibDW UNKNOWN IMPORTED GLOBAL)
        set_target_properties(
            LibDW::LibDW
            PROPERTIES
                IMPORTED_LINK_INTERFACE_LANGUAGES "C"
                IMPORTED_LOCATION "${_eu_libdw}"
                INTERFACE_INCLUDE_DIRECTORIES "${_eu_root}/include"
        )
        add_dependencies(LibDW::LibDW rocprofiler-systems-elfutils-build)
    endif()

    set(LibElf_FOUND TRUE)
    set(LibElf_INCLUDE_DIRS "${_eu_root}/include")
    set(LibElf_LIBRARIES "${_eu_libelf}")
    set(LibElf_VERSION "${ELFUTILS_DOWNLOAD_VERSION}")
    if(NOT TARGET LibElf::LibElf)
        add_library(LibElf::LibElf UNKNOWN IMPORTED GLOBAL)
        set_target_properties(
            LibElf::LibElf
            PROPERTIES
                IMPORTED_LINK_INTERFACE_LANGUAGES "C"
                IMPORTED_LOCATION "${_eu_libelf}"
                INTERFACE_INCLUDE_DIRECTORIES "${_eu_root}/include"
        )
        add_dependencies(LibElf::LibElf rocprofiler-systems-elfutils-build)
    endif()

    # Pre-create Dyninst::ElfUtils so Dyninst's tpls/DyninstElfUtils.cmake
    # short-circuits its own find_package(Elfutils). Without this, Dyninst's
    # bundled FindLibDW/FindLibElf (prepended at position 0 of
    # CMAKE_MODULE_PATH by external/dyninst/CMakeLists.txt) run against the
    # host elfutils and clash with the bundled version we just staged.
    if(NOT TARGET Dyninst::ElfUtils)
        add_library(Dyninst::ElfUtils INTERFACE IMPORTED GLOBAL)
        target_link_libraries(Dyninst::ElfUtils INTERFACE LibElf::LibElf LibDW::LibDW)
        target_include_directories(
            Dyninst::ElfUtils
            SYSTEM
            INTERFACE "${_eu_root}/include"
        )
    endif()

    unset(_eu_root)
    unset(_eu_libdw)
    unset(_eu_libelf)
    unset(_eu_patch_args)

    # --------------------------------------------------------------------------------------
    # Modes 2 and 3: discover via shim Find modules (config-package then pkg-config).
    # --------------------------------------------------------------------------------------
else()
    find_package(LibElf ${ElfUtils_MIN_VERSION})
    if(LibElf_FOUND)
        find_package(LibDW ${ElfUtils_MIN_VERSION})
        if(ENABLE_DEBUGINFOD)
            find_package(LibDebuginfod ${ElfUtils_MIN_VERSION} REQUIRED)
        endif()
    endif()

    if(NOT (LibElf_FOUND AND LibDW_FOUND))
        if(STERILE_BUILD)
            rocprofiler_systems_message(FATAL_ERROR
                "ElfUtils not found and cannot be downloaded because build is sterile."
            )
        else()
            rocprofiler_systems_message(FATAL_ERROR
                "ElfUtils was not found. Either configure cmake to find ElfUtils properly or set ROCPROFSYS_BUILD_ELFUTILS=ON to download and build."
            )
        endif()
    endif()
endif()

# --------------------------------------------------------------------------------------
# Populate the umbrella rocprofiler-systems-elfutils interface library.
# --------------------------------------------------------------------------------------
target_link_libraries(rocprofiler-systems-elfutils INTERFACE LibElf::LibElf LibDW::LibDW)
if(ENABLE_DEBUGINFOD AND TARGET LibDebuginfod::LibDebuginfod)
    target_link_libraries(
        rocprofiler-systems-elfutils
        INTERFACE LibDebuginfod::LibDebuginfod
    )
endif()

# Legacy aggregate variables, kept for any external consumer that read them.
set(ElfUtils_INCLUDE_DIRS
    ${LibElf_INCLUDE_DIRS}
    ${LibDW_INCLUDE_DIRS}
    CACHE PATH
    "elfutils include directories"
    FORCE
)
set(ElfUtils_LIBRARIES
    ${LibElf_LIBRARIES}
    ${LibDW_LIBRARIES}
    CACHE FILEPATH
    "elfutils library files"
    FORCE
)
if(ENABLE_DEBUGINFOD AND LibDebuginfod_FOUND)
    set(ElfUtils_INCLUDE_DIRS
        ${ElfUtils_INCLUDE_DIRS}
        ${LibDebuginfod_INCLUDE_DIRS}
        CACHE PATH
        "elfutils include directories"
        FORCE
    )
    set(ElfUtils_LIBRARIES
        ${ElfUtils_LIBRARIES}
        ${LibDebuginfod_LIBRARIES}
        CACHE FILEPATH
        "elfutils library files"
        FORCE
    )
endif()

rocprofiler_systems_message(STATUS "ElfUtils libdw:  ${LibDW_LIBRARIES} (v${LibDW_VERSION})")
rocprofiler_systems_message(STATUS "ElfUtils libelf: ${LibElf_LIBRARIES} (v${LibElf_VERSION})")
