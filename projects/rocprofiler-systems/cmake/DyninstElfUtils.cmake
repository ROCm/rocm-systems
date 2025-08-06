# ======================================================================================
# elfutils.cmake
#
# Configure elfutils for Dyninst
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

if(NOT BUILD_ELFUTILS)
    find_package(Elfutils)
endif()

if(LibElf_FOUND AND LibDwarf_FOUND AND NOT ENABLE_DEBUGINFOD)
    return()
endif()

if(NOT UNIX)
    return()
endif()

# Minimum acceptable version of elfutils NB: We need >=0.178 because libdw isn't
# thread-safe before then
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

# -------------- PATHS --------------------------------------------------------

# Base directory the of elfutils installation
set(ElfUtils_ROOT_DIR "/usr" CACHE PATH "Base directory the of elfutils installation")

# Hint directory that contains the elfutils headers files
set(ElfUtils_INCLUDEDIR
    "${ElfUtils_ROOT_DIR}/include"
    CACHE PATH
    "Hint directory that contains the elfutils headers files"
)

# Hint directory that contains the elfutils library files
set(ElfUtils_LIBRARYDIR
    "${ElfUtils_ROOT_DIR}/lib"
    CACHE PATH
    "Hint directory that contains the elfutils library files"
)

# libelf/dwarf-specific directory hints
foreach(l LibElf LibDwarf LibDebuginfod)
    foreach(d ROOT_DIR INCLUDEDIR LIBRARYDIR)
        set(${l}_${d} ${ElfUtils_${d}})
    endforeach()
endforeach()

# -------------- PACKAGES------------------------------------------------------

if(NOT BUILD_ELFUTILS)
    find_package(LibElf ${ElfUtils_MIN_VERSION})

    # Don't search for libdw or libdebuginfod if we didn't find a suitable libelf
    if(LibElf_FOUND)
        find_package(LibDwarf ${ElfUtils_MIN_VERSION})
        if(ENABLE_DEBUGINFOD)
            find_package(LibDebuginfod ${ElfUtils_MIN_VERSION} REQUIRED)
        endif()
    endif()
endif()

# -------------- SOURCE BUILD -------------------------------------------------

if(LibElf_FOUND AND LibDwarf_FOUND AND (NOT ENABLE_DEBUGINFOD OR LibDebuginfod_FOUND))
    if(ENABLE_DEBUGINFOD AND LibDebuginfod_FOUND)
        set(_eu_root ${ElfUtils_ROOT_DIR})
        set(_eu_inc_dirs
            ${LibElf_INCLUDE_DIRS}
            ${LibDwarf_INCLUDE_DIRS}
            ${LibDebuginfod_INCLUDE_DIRS}
        )
        set(_eu_lib_dirs
            ${LibElf_LIBRARY_DIRS}
            ${LibDwarf_LIBRARY_DIRS}
            ${LibDebuginfod_LIBRARY_DIRS}
        )
        set(_eu_libs ${LibElf_LIBRARIES} ${LibDwarf_LIBRARIES} ${LibDebuginfod_LIBRARIES})
    else()
        set(_eu_root ${ElfUtils_ROOT_DIR})
        set(_eu_inc_dirs ${LibElf_INCLUDE_DIRS} ${LibDwarf_INCLUDE_DIRS})
        set(_eu_lib_dirs ${LibElf_LIBRARY_DIRS} ${LibDwarf_LIBRARY_DIRS})
        set(_eu_libs ${LibElf_LIBRARIES} ${LibDwarf_LIBRARIES})
    endif()
elseif(NOT (LibElf_FOUND AND LibDwarf_FOUND) AND STERILE_BUILD)
    rocprofiler_systems_message(
        FATAL_ERROR
        "ElfUtils not found and cannot be downloaded because build is sterile."
    )
elseif(NOT BUILD_ELFUTILS)
    rocprofiler_systems_message(
        FATAL_ERROR
        "ElfUtils was not found. Either configure cmake to find ElfUtils properly or set BUILD_ELFUTILS=ON to download and build"
    )
else()
    # If we didn't find a suitable version on the system, then download one from the web
    rocprofiler_systems_add_cache_option(
        ELFUTILS_DOWNLOAD_VERSION "Version of elfutils to download and install" STRING
        "0.188"
    )
    set(ELFUTILS_DOWNLOAD_VERSION ${ElfUtils_DOWNLOAD_VERSION})

    # make sure we are not downloading a version less than minimum
    if(${ELFUTILS_DOWNLOAD_VERSION} VERSION_LESS ${ElfUtils_MIN_VERSION})
        rocprofiler_systems_message(
            FATAL_ERROR
            "elfutils download version is set to ${ELFUTILS_DOWNLOAD_VERSION} but elfutils minimum version is set to ${ElfUtils_MIN_VERSION}"
        )
    endif()

    rocprofiler_systems_message(STATUS "${ElfUtils_ERROR_REASON}")
    rocprofiler_systems_message(
        STATUS
        "Attempting to build elfutils(${ELFUTILS_DOWNLOAD_VERSION}) as external project"
    )

    if(
        NOT (${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU")
        OR NOT (${CMAKE_C_COMPILER_ID} STREQUAL "GNU")
    )
        rocprofiler_systems_message(FATAL_ERROR
                                    "ElfUtils will only build with the GNU compiler"
        )
    endif()

    set(_eu_root ${TPL_STAGING_PREFIX}/elfutils)
    set(_eu_inc_dirs $<BUILD_INTERFACE:${_eu_root}/include>)
    set(_eu_lib_dirs $<BUILD_INTERFACE:${_eu_root}/lib>)
    set(_eu_libs
        $<BUILD_INTERFACE:${_eu_root}/lib/libdw${CMAKE_SHARED_LIBRARY_SUFFIX}>
        $<BUILD_INTERFACE:${_eu_root}/lib/libelf${CMAKE_SHARED_LIBRARY_SUFFIX}>
    )
    set(_eu_build_byproducts
        "${_eu_root}/lib/libdw${CMAKE_SHARED_LIBRARY_SUFFIX}"
        "${_eu_root}/lib/libelf${CMAKE_SHARED_LIBRARY_SUFFIX}"
    )

    file(MAKE_DIRECTORY "${_eu_root}/lib")
    file(MAKE_DIRECTORY "${_eu_root}/include")

    include(ExternalProject)
    ExternalProject_Add(
        rocprofiler-systems-elfutils-build
        PREFIX ${_eu_root}
        URL
            ${ElfUtils_DOWNLOAD_URL}
            "https://sourceware.org/elfutils/ftp/${ELFUTILS_DOWNLOAD_VERSION}/elfutils-${ELFUTILS_DOWNLOAD_VERSION}.tar.bz2"
            "https://mirrors.kernel.org/sourceware/elfutils/${ELFUTILS_DOWNLOAD_VERSION}/elfutils-${ELFUTILS_DOWNLOAD_VERSION}.tar.bz2"
        BUILD_IN_SOURCE 1
        CONFIGURE_COMMAND
            ${CMAKE_COMMAND} -E env CC=${CMAKE_C_COMPILER} CFLAGS=-fPIC\ -O3
            CXX=${CMAKE_CXX_COMPILER} CXXFLAGS=-fPIC\ -O3
            [=[LDFLAGS=-Wl,-rpath='$$ORIGIN']=] <SOURCE_DIR>/configure
            --enable-install-elfh --prefix=${_eu_root} --disable-libdebuginfod
            --disable-debuginfod --enable-thread-safety ${ElfUtils_CONFIG_OPTIONS}
            --libdir=${_eu_root}/lib
        BUILD_COMMAND make install
        BUILD_BYPRODUCTS ${_eu_build_byproducts}
        INSTALL_COMMAND ""
    )

    # target for re-executing the installation
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
endif()

# -------------- EXPORT VARIABLES ---------------------------------------------

set(ElfUtils_ROOT_DIR
    ${_eu_root}
    CACHE PATH
    "Base directory the of elfutils installation"
    FORCE
)
set(ElfUtils_INCLUDE_DIRS ${_eu_inc_dirs} CACHE PATH "elfutils include directory" FORCE)
set(ElfUtils_LIBRARY_DIRS ${_eu_lib_dirs} CACHE PATH "elfutils library directory" FORCE)
set(ElfUtils_INCLUDE_DIR
    ${ElfUtils_INCLUDE_DIRS}
    CACHE PATH
    "elfutils include directory"
    FORCE
)
set(ElfUtils_LIBRARIES ${_eu_libs} CACHE FILEPATH "elfutils library files" FORCE)

target_include_directories(
    rocprofiler-systems-elfutils
    SYSTEM
    INTERFACE ${ElfUtils_INCLUDE_DIRS}
)
target_compile_definitions(rocprofiler-systems-elfutils INTERFACE ${ElfUtils_DEFINITIONS})
target_link_directories(rocprofiler-systems-elfutils INTERFACE ${ElfUtils_LIBRARY_DIRS})
target_link_libraries(rocprofiler-systems-elfutils INTERFACE ${ElfUtils_LIBRARIES})

rocprofiler_systems_message(STATUS "ElfUtils includes: ${ElfUtils_INCLUDE_DIRS}")
rocprofiler_systems_message(STATUS "ElfUtils library dirs: ${ElfUtils_LIBRARY_DIRS}")
rocprofiler_systems_message(STATUS "ElfUtils libraries: ${ElfUtils_LIBRARIES}")
