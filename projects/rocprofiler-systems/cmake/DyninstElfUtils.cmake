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
#      ExternalProject, then pre-create the LibDW::LibDW / LibELF::LibELF
#      targets so the shim FindLibDW / FindLibELF modules short-circuit
#      when Dyninst later does find_package(LibDW) / find_package(LibELF).
#
#   2. ROCPROFSYS_BUILD_ELFUTILS=OFF, system install. Shim falls through to
#      pkg-config and discovers /usr/lib libdw.so / libelf.so.
#
#   3. ROCPROFSYS_BUILD_ELFUTILS=OFF, vendored sysdep (TheRock). TheRock sets
#      libdw_DIR / libelf_DIR; shim's find_package(libdw|libelf CONFIG)
#      resolves the vendored configs.
#
# Exported to consumers
# ---------------------
#   LibDW::LibDW, LibELF::LibELF  — imported targets (preferred)
#   LibDW_*, LibELF_*             — legacy variables (Dyninst's FindElfutils.cmake)
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
    set(ELFUTILS_DOWNLOAD_VERSION ${ElfUtils_DOWNLOAD_VERSION})

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

    # Re-run installation target (kept from the original module for parity).
    add_custom_target(
        rocprofiler-systems-elfutils-install
        COMMAND make install
        WORKING_DIRECTORY ${${_eu_root}}/src/ElfUtils-External
        COMMENT "Installing ElfUtils..."
    )

    install(
        DIRECTORY ${_eu_root}/lib/
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/${PROJECT_NAME}
        FILES_MATCHING
        PATTERN "*${CMAKE_SHARED_LIBRARY_SUFFIX}*"
    )

    # Pre-create the imported targets so the FindLibDW / FindLibELF shims
    # short-circuit when Dyninst's add_subdirectory later does find_package.
    # The targets carry an add_dependencies edge to the ExternalProject build
    # step so anything that links them waits for the .so to exist.
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

    set(LibELF_FOUND TRUE)
    set(LibELF_INCLUDE_DIRS "${_eu_root}/include")
    set(LibELF_LIBRARIES "${_eu_libelf}")
    set(LibELF_VERSION "${ELFUTILS_DOWNLOAD_VERSION}")
    if(NOT TARGET LibELF::LibELF)
        add_library(LibELF::LibELF UNKNOWN IMPORTED GLOBAL)
        set_target_properties(
            LibELF::LibELF
            PROPERTIES
                IMPORTED_LINK_INTERFACE_LANGUAGES "C"
                IMPORTED_LOCATION "${_eu_libelf}"
                INTERFACE_INCLUDE_DIRECTORIES "${_eu_root}/include"
        )
        add_dependencies(LibELF::LibELF rocprofiler-systems-elfutils-build)
    endif()

    # Pre-create Dyninst::ElfUtils so Dyninst's tpls/DyninstElfUtils.cmake
    # short-circuits its own find_package(Elfutils). Without this, Dyninst's
    # bundled FindLibDW/FindLibELF (prepended at position 0 of
    # CMAKE_MODULE_PATH by external/dyninst/CMakeLists.txt) run against the
    # host elfutils and clash with the bundled version we just staged.
    if(NOT TARGET Dyninst::ElfUtils)
        add_library(Dyninst::ElfUtils INTERFACE IMPORTED GLOBAL)
        target_link_libraries(Dyninst::ElfUtils INTERFACE LibELF::LibELF LibDW::LibDW)
        target_include_directories(
            Dyninst::ElfUtils
            SYSTEM
            INTERFACE "${_eu_root}/include"
        )
    endif()

    # --------------------------------------------------------------------------------------
    # Modes 2 and 3: discover via shim Find modules (config-package then pkg-config).
    # --------------------------------------------------------------------------------------
else()
    find_package(LibELF ${ElfUtils_MIN_VERSION})
    if(LibELF_FOUND)
        find_package(LibDW ${ElfUtils_MIN_VERSION})
        if(ENABLE_DEBUGINFOD)
            find_package(LibDebuginfod ${ElfUtils_MIN_VERSION} REQUIRED)
        endif()
    endif()

    if(NOT (LibELF_FOUND AND LibDW_FOUND))
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
target_link_libraries(rocprofiler-systems-elfutils INTERFACE LibELF::LibELF LibDW::LibDW)
if(ENABLE_DEBUGINFOD AND TARGET LibDebuginfod::LibDebuginfod)
    target_link_libraries(
        rocprofiler-systems-elfutils
        INTERFACE LibDebuginfod::LibDebuginfod
    )
endif()

# Legacy aggregate variables, kept for any external consumer that read them.
set(ElfUtils_INCLUDE_DIRS
    ${LibELF_INCLUDE_DIRS}
    ${LibDW_INCLUDE_DIRS}
    CACHE PATH
    "elfutils include directories"
    FORCE
)
set(ElfUtils_LIBRARIES
    ${LibELF_LIBRARIES}
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
rocprofiler_systems_message(STATUS "ElfUtils libelf: ${LibELF_LIBRARIES} (v${LibELF_VERSION})")
