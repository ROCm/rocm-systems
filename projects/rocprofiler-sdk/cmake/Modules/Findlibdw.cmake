# Try to find libdw headers and libraries.
#
# Usage of this module as follows:
#
#     find_package(libdw)
#
# Variables used by this module, they can change the default behaviour and need
# to be set before calling find_package:
#
#  libdw_ROOT         Set this variable to the root installation of
#                     libdw if the module has problems finding the
#                     proper installation path.
#
# Variables defined by this module:
#
#  libdw_FOUND              System has libdw libraries and headers
#  libdw_LIBRARIES          The libdw library
#  libdw_INCLUDE_DIRS       The location of libdw headers
#
# Interface targets defined by this module:
#
#   libdw::libdw
#

# ROCm bundles libdw in <prefix>/lib/rocm_sysdeps and rocprofiler-sdk is built against it
get_filename_component(_libdw_rocm_sysdeps
                       "${CMAKE_CURRENT_LIST_DIR}/../../../rocm_sysdeps" REALPATH)

# Function scope keeps the CMAKE_PREFIX_PATH change local; results are returned via cache
# variables.
function(_rocprofiler_sdk_libdw_pkg_config _rocm_sysdeps)
    if(IS_DIRECTORY "${_rocm_sysdeps}")
        list(PREPEND CMAKE_PREFIX_PATH "${_rocm_sysdeps}")
    endif()
    set(PKG_CONFIG_USE_CMAKE_PREFIX_PATH ON)

    pkg_check_modules(DW libdw)

    if(DW_FOUND
       AND DW_INCLUDE_DIRS
       AND DW_LINK_LIBRARIES)
        set(libdw_INCLUDE_DIR
            "${DW_INCLUDE_DIRS}"
            CACHE FILEPATH "libdw include directory")
        set(libdw_LIBRARY
            "${DW_LINK_LIBRARIES}"
            CACHE FILEPATH "libdw libraries")
        if(DW_PREFIX)
            set(libdw_ROOT_DIR
                "${DW_PREFIX}"
                CACHE FILEPATH "libdw root directory")
        endif()

        if(DW_VERSION)
            set(libdw_VERSION
                "${DW_VERSION}"
                CACHE FILEPATH "libdw version")
        endif()
    endif()
endfunction()

find_package(PkgConfig)

# An explicit libdw_ROOT (variable or environment) is authoritative: pkg-config could
# return headers from another install while find_library picks the library from
# libdw_ROOT.
if(PkgConfig_FOUND
   AND NOT libdw_ROOT
   AND NOT DEFINED ENV{libdw_ROOT})
    set(ENV{PKG_CONFIG_SYSTEM_INCLUDE_PATH} "")
    _rocprofiler_sdk_libdw_pkg_config("${_libdw_rocm_sysdeps}")
endif()

if(NOT libdw_INCLUDE_DIR OR NOT libdw_LIBRARY)
    # HINTS are searched before system paths
    find_path(
        libdw_ROOT_DIR
        NAMES include/elfutils/libdw.h
        HINTS ${libdw_ROOT} ${_libdw_rocm_sysdeps}
        PATHS ${libdw_ROOT})

    mark_as_advanced(libdw_ROOT_DIR)

    find_path(
        libdw_INCLUDE_DIR
        NAMES elfutils/libdw.h
        HINTS ${libdw_ROOT} ${_libdw_rocm_sysdeps}
        PATHS ${libdw_ROOT}
        PATH_SUFFIXES include)

    find_library(
        libdw_LIBRARY
        NAMES dw
        HINTS ${libdw_ROOT} ${_libdw_rocm_sysdeps}
        PATHS ${libdw_ROOT}
        PATH_SUFFIXES lib lib64)
endif()

unset(_libdw_rocm_sysdeps)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(libdw DEFAULT_MSG libdw_LIBRARY libdw_INCLUDE_DIR)

if(libdw_FOUND)
    if(NOT TARGET libdw::libdw)
        add_library(libdw::libdw INTERFACE IMPORTED)

        if(TARGET PkgConfig::DW AND DW_FOUND)
            target_link_libraries(libdw::libdw INTERFACE PkgConfig::DW)
        else()
            target_link_libraries(libdw::libdw INTERFACE ${libdw_LIBRARY})
            target_include_directories(libdw::libdw SYSTEM INTERFACE ${libdw_INCLUDE_DIR})
        endif()
    endif()
endif()

mark_as_advanced(libdw_INCLUDE_DIR libdw_LIBRARY)
