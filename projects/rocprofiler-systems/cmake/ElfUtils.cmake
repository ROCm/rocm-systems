# ======================================================================================
# elfutils.cmake
#
# Configure elfutils for rocprofiler-systems
#
# ----------------------------------------
#
# Accepts the following CMake variables
#
# ElfUtils_ROOT_DIR       - Base directory the of elfutils installation
# ElfUtils_INCLUDEDIR     - Hint directory that contains the elfutils headers files
# ElfUtils_LIBRARYDIR     - Hint directory that contains the elfutils library files
# ElfUtils_MIN_VERSION    - Minimum acceptable version of elfutils
#
# Directly exports the following CMake variables
#
# ElfUtils_ROOT_DIR       - Computed base directory the of elfutils installation
# ElfUtils_INCLUDE_DIRS   - elfutils include directories ElfUtils_LIBRARY_DIRS - Link
# directories for elfutils libraries ElfUtils_LIBRARIES      - elfutils library files
#
# NOTE: The exported ElfUtils_ROOT_DIR can be different from the value provided by the
# user in the case that it is determined to build elfutils from source. In such a case,
# ElfUtils_ROOT_DIR will contain the directory of the from-source installation.
#
# See Modules/FindLibElf.cmake and Modules/FindLibDwarf.cmake for details
#
# ======================================================================================

include_guard(GLOBAL)
include(ExternalProject)

# Minimum acceptable version of elfutils NB: We need >=0.178 because libdw isn't
# thread-safe before then
set(_min_version 0.178)

set(ElfUtils_MIN_VERSION
    ${_min_version}
    CACHE STRING "Minimum acceptable elfutils version")

if(${ElfUtils_MIN_VERSION} VERSION_LESS ${_min_version})
    rocprofiler_systems_message(
        FATAL_ERROR
        "Requested version ${ElfUtils_MIN_VERSION} is less than minimum supported version (${_min_version})"
        )
endif()

# If we didn't find a suitable version on the system, then download one from the web
set(ElfUtils_DOWNLOAD_VERSION
    "0.188"
    CACHE STRING "Version of elfutils to download and install")

# make sure we are not downloading a version less than minimum
if(${ElfUtils_DOWNLOAD_VERSION} VERSION_LESS ${ElfUtils_MIN_VERSION})
    rocprofiler_systems_message(
        FATAL_ERROR
        "elfutils download version is set to ${ElfUtils_DOWNLOAD_VERSION} but elfutils minimum version is set to ${ElfUtils_MIN_VERSION}"
        )
endif()

if(CMAKE_C_COMPILER_ID MATCHES "GNU")
    set(ElfUtils_C_COMPILER
        "${CMAKE_C_COMPILER}"
        CACHE FILEPATH "C compiler used to compiler ElfUtils")
else()
    find_program(
        ElfUtils_C_COMPILER
        NAMES gcc
        PATH_SUFFIXES bin)
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    set(ElfUtils_CXX_COMPILER
        "${CMAKE_CXX_COMPILER}"
        CACHE FILEPATH "C++ compiler used to compiler ElfUtils")
else()
    find_program(
        ElfUtils_CXX_COMPILER
        NAMES g++
        PATH_SUFFIXES bin)
endif()

find_program(
    MAKE_COMMAND
    NAMES make gmake
    PATH_SUFFIXES bin)

if(NOT ElfUtils_C_COMPILER OR NOT ElfUtils_CXX_COMPILER)
    rocprofiler_systems_message(
        FATAL_ERROR
        "ElfUtils requires the GNU C and C++ compilers. ElfUtils_C_COMPILER: ${ElfUtils_C_COMPILER}, ElfUtils_CXX_COMPILER: ${ElfUtils_CXX_COMPILER}"
        )
endif()

set(_eu_root ${PROJECT_BINARY_DIR}/external/elfutils)
set(_eu_inc_dirs $<BUILD_INTERFACE:${_eu_root}/include>)
set(_eu_lib_dirs $<BUILD_INTERFACE:${_eu_root}/lib>)
set(_eu_libs $<BUILD_INTERFACE:${_eu_root}/lib/libdw${CMAKE_STATIC_LIBRARY_SUFFIX}>
             $<BUILD_INTERFACE:${_eu_root}/lib/libelf${CMAKE_STATIC_LIBRARY_SUFFIX}>)
set(_eu_build_byproducts "${_eu_root}/lib/libdw${CMAKE_STATIC_LIBRARY_SUFFIX}"
                         "${_eu_root}/lib/libelf${CMAKE_STATIC_LIBRARY_SUFFIX}")

externalproject_add(
    rocprofiler-systems-elfutils-build
    PREFIX ${PROJECT_BINARY_DIR}/external/elfutils
    URL ${ElfUtils_DOWNLOAD_URL}
        "https://sourceware.org/elfutils/ftp/${ElfUtils_DOWNLOAD_VERSION}/elfutils-${ElfUtils_DOWNLOAD_VERSION}.tar.bz2"
        "https://mirrors.kernel.org/sourceware/elfutils/${ElfUtils_DOWNLOAD_VERSION}/elfutils-${ElfUtils_DOWNLOAD_VERSION}.tar.bz2"
    BUILD_IN_SOURCE 1
    CONFIGURE_COMMAND
        ${CMAKE_COMMAND} -E env CC=${ElfUtils_C_COMPILER}
        CFLAGS=-fPIC\ -O3\ -Wno-error=null-dereference CXX=${ElfUtils_CXX_COMPILER}
        CXXFLAGS=-fPIC\ -O3\ -Wno-error=null-dereference
        [=[LDFLAGS=-Wl,-rpath='$$ORIGIN']=] <SOURCE_DIR>/configure --enable-install-elfh
        --prefix=${_eu_root} --disable-libdebuginfod --disable-debuginfod --disable-nls
        --enable-thread-safety --enable-silent-rules
    BUILD_COMMAND ${MAKE_COMMAND} install -s
    BUILD_BYPRODUCTS "${_eu_build_byproducts}"
    INSTALL_COMMAND "")

# target for re-executing the installation
add_custom_target(
    rocprofiler-systems-elfutils-install
    COMMAND ${MAKE_COMMAND} install -s
    WORKING_DIRECTORY ${PROJECT_BINARY_DIR}/external/elfutils/src/ElfUtils-External
    COMMENT "Installing ElfUtils...")

# -------------- EXPORT VARIABLES ---------------------------------------------

set(ElfUtils_ROOT_DIR
    ${_eu_root}
    CACHE PATH "Base directory the of elfutils installation" FORCE)
set(ElfUtils_INCLUDE_DIRS
    ${_eu_inc_dirs}
    CACHE PATH "elfutils include directory" FORCE)
set(ElfUtils_LIBRARY_DIRS
    ${_eu_lib_dirs}
    CACHE PATH "elfutils library directory" FORCE)
set(ElfUtils_INCLUDE_DIR
    ${ElfUtils_INCLUDE_DIRS}
    CACHE PATH "elfutils include directory" FORCE)
set(ElfUtils_LIBRARIES
    ${_eu_libs}
    CACHE FILEPATH "elfutils library files" FORCE)
