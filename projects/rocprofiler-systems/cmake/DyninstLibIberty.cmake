# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# ======================================================================================
# LibIberty.cmake
#
# Configure LibIberty for Dyninst
#
# ----------------------------------------
#
# Directly exports the following CMake variables
#
# LibIberty_ROOT_DIR       - Computed base directory of the LibIberty installation
# LibIberty_LIBRARY_DIRS   - Link directories for LibIberty libraries LibIberty_LIBRARIES
# - LibIberty library files LibIberty_INCLUDE - LibIberty include files
#
# NOTE: The exported LibIberty_ROOT_DIR can be different from the value provided by the
# user in the case that it is determined to build LibIberty from source. In such a case,
# LibIberty_ROOT_DIR will contain the directory of the from-source installation.
#
# See Modules/FindLibIberty.cmake for details
#
# ======================================================================================

include_guard(GLOBAL)

if(NOT UNIX)
    return()
endif()

# -------------- PATHS --------------------------------------------------------

# Base directory of the LibIberty installation
set(LibIberty_ROOT_DIR "/usr" CACHE PATH "Base directory of the LibIberty installation")

# Hint directory that contains the LibIberty library files
set(LibIberty_LIBRARYDIR
    "${LibIberty_ROOT_DIR}/lib"
    CACHE PATH
    "Hint directory that contains the LibIberty library files"
)

# -------------- PACKAGES -----------------------------------------------------

if(NOT ROCPROFSYS_BUILD_LIBIBERTY)
    find_package(LibIberty)
endif()

# -------------- SOURCE BUILD -------------------------------------------------

if(LibIberty_FOUND)
    set(_li_root ${LibIberty_ROOT_DIR})
    set(_li_inc_dirs ${LibIberty_INCLUDE_DIRS})
    set(_li_lib_dirs ${LibIberty_LIBRARY_DIRS})
    set(_li_libs ${LibIberty_LIBRARIES})
elseif(STERILE_BUILD)
    rocprofiler_systems_message(
        FATAL_ERROR
        "LibIberty not found and cannot be downloaded because build is sterile."
    )
elseif(NOT ROCPROFSYS_BUILD_LIBIBERTY)
    rocprofiler_systems_message(
        FATAL_ERROR
        "LibIberty was not found. Either configure cmake to find LibIberty properly or set ROCPROFSYS_BUILD_LIBIBERTY=ON to download and build"
    )
else()
    rocprofiler_systems_message(STATUS "${LibIberty_ERROR_REASON}")
    rocprofiler_systems_message(STATUS
                                "Attempting to build LibIberty as external project"
    )

    set(_li_root ${TPL_STAGING_PREFIX}/binutils)
    set(_li_project_name rocprofiler-systems-libiberty-build)
    set(_li_working_dir ${_li_root}/src/${_li_project_name})
    set(_li_inc_dirs $<BUILD_INTERFACE:${_li_root}/include>)
    set(_li_lib_dirs $<BUILD_INTERFACE:${_li_root}/lib>)
    set(_li_libs
        $<BUILD_INTERFACE:${_li_root}/lib/libiberty${CMAKE_STATIC_LIBRARY_SUFFIX}>
    )
    set(_li_build_byproducts "${_li_root}/lib/libiberty${CMAKE_STATIC_LIBRARY_SUFFIX}")

    file(MAKE_DIRECTORY "${_li_root}/lib")
    file(MAKE_DIRECTORY "${_li_root}/include")

    # Build only libiberty (not top-level "all"): full binutils needs bison, etc.
    # bfd doc rules can invoke makeinfo; use a no-op so Texinfo is not required.
    find_program(_ROCPROFSYS_LIBIBERTY_MAKEINFO_NOOP NAMES true)
    if(NOT _ROCPROFSYS_LIBIBERTY_MAKEINFO_NOOP)
        set(_ROCPROFSYS_LIBIBERTY_MAKEINFO_NOOP /usr/bin/true)
    endif()

    # Single source of truth for the binutils release used to build libiberty,
    # shared with docker/Dockerfile.*.ci (which COPY this same file to stage a
    # matching tarball at image-build time). Keeping this in one place avoids
    # the version drift already seen with the Dockerfiles' unused, stale
    # ELFUTILS_DOWNLOAD_VERSION ARG.
    file(
        STRINGS "${CMAKE_CURRENT_LIST_DIR}/binutils_version.txt"
        _li_binutils_version
        LIMIT_COUNT 1
    )
    string(STRIP "${_li_binutils_version}" _li_binutils_version)
    rocprofiler_systems_add_cache_option(
        BINUTILS_DOWNLOAD_VERSION "Version of binutils to download libiberty from" STRING
        "${_li_binutils_version}"
    )

    # ExternalProject_Add's URL list rejects any entry without a network scheme
    # (e.g. file://) once more than one URL is given, so a local override can't
    # simply be prepended to the GNU mirrors below — it must replace them.
    #
    # ftp.gnu.org is GNU's own canonical server (fastest and most reliable in
    # testing); ftpmirror.gnu.org is a third-party mirror redirector that has
    # been observed to intermittently 502, so it's kept only as a fallback.
    if(DYNINST_BINUTILS_DOWNLOAD_URL)
        set(_li_binutils_urls ${DYNINST_BINUTILS_DOWNLOAD_URL})
    else()
        set(_li_binutils_urls
            https://ftp.gnu.org/gnu/binutils/binutils-${BINUTILS_DOWNLOAD_VERSION}.tar.gz
            https://ftpmirror.gnu.org/gnu/binutils/binutils-${BINUTILS_DOWNLOAD_VERSION}.tar.gz
            https://mirrors.kernel.org/sourceware/binutils/releases/binutils-${BINUTILS_DOWNLOAD_VERSION}.tar.gz
        )
    endif()

    include(ExternalProject)
    ExternalProject_Add(
        ${_li_project_name}
        PREFIX ${_li_root}
        URL ${_li_binutils_urls}
        BUILD_IN_SOURCE 1
        CONFIGURE_COMMAND
            ${CMAKE_COMMAND} -E env CC=${CMAKE_C_COMPILER} CFLAGS=-fPIC\ -O3\ -Wno-error
            CXX=${CMAKE_CXX_COMPILER} CXXFLAGS=-fPIC\ -O3\ -Wno-error
            MAKEINFO=${_ROCPROFSYS_LIBIBERTY_MAKEINFO_NOOP} <SOURCE_DIR>/configure
            --prefix=${_li_root}
        BUILD_COMMAND make MAKEINFO=${_ROCPROFSYS_LIBIBERTY_MAKEINFO_NOOP} all-libiberty
        INSTALL_COMMAND ""
    )

    add_custom_command(
        OUTPUT ${_li_build_byproducts}
        COMMAND install
        ARGS -C ${_li_working_dir}/libiberty/libiberty.a ${_li_root}/lib
        COMMAND install
        ARGS -C ${_li_working_dir}/include/*.h ${_li_root}/include
        DEPENDS ${_li_project_name}
        COMMENT "Installing LibIberty..."
    )

    add_custom_target(
        rocprofiler-systems-libiberty-install
        ALL
        DEPENDS ${_li_build_byproducts}
    )

    # For backward compatibility
    set(IBERTY_FOUND TRUE)
    set(IBERTY_BUILD TRUE)
endif()

# -------------- EXPORT VARIABLES ---------------------------------------------

foreach(_DIR_TYPE inc lib)
    if(_li_${_DIR_TYPE}_dirs)
        list(REMOVE_DUPLICATES _li_${_DIR_TYPE}_dirs)
    endif()
endforeach()

target_include_directories(rocprofiler-systems-libiberty INTERFACE ${_li_inc_dirs})
target_link_directories(rocprofiler-systems-libiberty INTERFACE ${_lib_lib_dirs})
target_link_libraries(rocprofiler-systems-libiberty INTERFACE ${_li_libs})

set(LibIberty_ROOT_DIR
    ${_li_root}
    CACHE PATH
    "Base directory of the LibIberty installation"
    FORCE
)
set(LibIberty_INCLUDE_DIRS
    ${_li_inc_dirs}
    CACHE PATH
    "LibIberty include directories"
    FORCE
)
set(LibIberty_LIBRARY_DIRS ${_li_lib_dirs} CACHE PATH "LibIberty library directory" FORCE)
set(LibIberty_LIBRARIES ${_li_libs} CACHE FILEPATH "LibIberty library files" FORCE)

# For backward compatibility only
set(IBERTY_LIBRARIES ${LibIberty_LIBRARIES})

rocprofiler_systems_message(STATUS "LibIberty include dirs: ${LibIberty_INCLUDE_DIRS}")
rocprofiler_systems_message(STATUS "LibIberty library dirs: ${LibIberty_LIBRARY_DIRS}")
rocprofiler_systems_message(STATUS "LibIberty libraries: ${LibIberty_LIBRARIES}")

# --------------------------------------------------------------------------------------#
# Create Dyninst::LibIberty target if building from source
# --------------------------------------------------------------------------------------#
# When LibIberty is built from source, Dyninst's find_package(LibIberty) would fail
# because the bundled LibIberty isn't installed in standard locations. Creating this
# target causes Dyninst to skip find_package(LibIberty) and use the bundled dependency.
if(ROCPROFSYS_BUILD_LIBIBERTY AND NOT TARGET Dyninst::LibIberty)
    add_library(Dyninst::LibIberty INTERFACE IMPORTED)
    target_link_libraries(Dyninst::LibIberty INTERFACE ${LibIberty_LIBRARIES})
    target_include_directories(
        Dyninst::LibIberty
        SYSTEM
        INTERFACE ${LibIberty_INCLUDE_DIRS}
    )
    if(LibIberty_LIBRARY_DIRS)
        target_link_directories(Dyninst::LibIberty INTERFACE ${LibIberty_LIBRARY_DIRS})
    endif()
endif()
