#
# common utilities for samples
#

find_package(PkgConfig)

if(PkgConfig_FOUND)
    set(ENV{PKG_CONFIG_SYSTEM_INCLUDE_PATH} "")
    pkg_check_modules(DW libdw)

    if(DW_FOUND
       AND DW_INCLUDE_DIRS
       AND DW_LIBRARIES)
        set(libdw_INCLUDE_DIR
            "${DW_INCLUDE_DIRS}"
            CACHE FILEPATH "libdw include directory")
        set(libdw_LIBRARY
            "${DW_LIBRARIES}"
            CACHE FILEPATH "libdw libraries")
    endif()
endif()

if(NOT libdw_INCLUDE_DIR OR NOT libdw_LIBRARY)
    find_path(
        libdw_ROOT_DIR
        NAMES include/elfutils/libdw.h
        HINTS ${libdw_ROOT}
        PATHS ${libdw_ROOT})

    mark_as_advanced(libdw_ROOT_DIR)

    find_path(
        libdw_INCLUDE_DIR
        NAMES elfutils/libdw.h
        HINTS ${libdw_ROOT}
        PATHS ${libdw_ROOT}
        PATH_SUFFIXES include)

    find_library(
        libdw_LIBRARY
        NAMES dw
        HINTS ${libdw_ROOT}
        PATHS ${libdw_ROOT}
        PATH_SUFFIXES lib lib64)
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(libdw DEFAULT_MSG libdw_LIBRARY libdw_INCLUDE_DIR)

if(libdw_FOUND AND NOT TARGET libdw::libdw)
    add_library(libdw::libdw INTERFACE IMPORTED)
    if(TARGET PkgConfig::DW AND DW_FOUND)
        target_link_libraries(libdw::libdw INTERFACE PkgConfig::DW)
    else()
        target_link_libraries(libdw::libdw INTERFACE ${libdw_LIBRARY})
        target_include_directories(libdw::libdw SYSTEM INTERFACE ${libdw_INCLUDE_DIR})
    endif()
endif()

mark_as_advanced(libdw_INCLUDE_DIR libdw_LIBRARY)

# default FAIL_REGULAR_EXPRESSION for tests
set(ROCPROFILER_DEFAULT_FAIL_REGEX
    "threw an exception|Permission denied|Could not create logging file"
    CACHE STRING "Default FAIL_REGULAR_EXPRESSION for tests")

# build flags
add_library(rocprofiler-sdk-samples-build-flags INTERFACE)
add_library(rocprofiler-sdk::samples-build-flags ALIAS
            rocprofiler-sdk-samples-build-flags)
target_compile_options(rocprofiler-sdk-samples-build-flags INTERFACE -W -Wall -Wextra
                                                                     -Wshadow)
target_compile_features(rocprofiler-sdk-samples-build-flags INTERFACE cxx_std_17)

if(ROCPROFILER_BUILD_CI OR ROCPROFILER_BUILD_WERROR)
    target_compile_options(rocprofiler-sdk-samples-build-flags INTERFACE -Werror)
endif()

# common utilities
cmake_path(GET CMAKE_CURRENT_SOURCE_DIR PARENT_PATH COMMON_LIBRARY_INCLUDE_DIR)

add_library(rocprofiler-sdk-samples-common-library INTERFACE)
add_library(rocprofiler-sdk::samples-common-library ALIAS
            rocprofiler-sdk-samples-common-library)
target_link_libraries(rocprofiler-sdk-samples-common-library
                      INTERFACE rocprofiler-sdk::samples-build-flags libdw::libdw)
target_compile_features(rocprofiler-sdk-samples-common-library INTERFACE cxx_std_17)
target_include_directories(rocprofiler-sdk-samples-common-library
                           INTERFACE ${COMMON_LIBRARY_INCLUDE_DIR})

set(EXTERNAL_SUBMODULE_DIR "${PROJECT_SOURCE_DIR}/../external")
cmake_path(ABSOLUTE_PATH EXTERNAL_SUBMODULE_DIR NORMALIZE)

if(EXISTS ${EXTERNAL_SUBMODULE_DIR}/filesystem/include/ghc/filesystem.hpp)
    target_compile_definitions(
        rocprofiler-sdk-samples-common-library
        INTERFACE $<BUILD_INTERFACE:ROCPROFILER_SAMPLES_HAS_GHC_LIB_FILESYSTEM=1>)
    target_include_directories(
        rocprofiler-sdk-samples-common-library SYSTEM
        INTERFACE $<BUILD_INTERFACE:${EXTERNAL_SUBMODULE_DIR}/filesystem/include>)
endif()

# function for getting the LD_PRELOAD environment variable
function(rocprofiler_samples_get_preload_env _VAR)
    set(_PRELOAD_ENV_LIBS ${ROCPROFILER_MEMCHECK_PRELOAD_ENV_VALUE} $ENV{LD_PRELOAD})

    foreach(_TARG ${ARGN})
        if(NOT TARGET ${_TARG})
            message(
                FATAL_ERROR
                    "rocprofiler_samples_get_preload_env: '${_TARG}' is not a valid target"
                )
        endif()
        list(APPEND _PRELOAD_ENV_LIBS $<TARGET_FILE:${_TARG}>)
    endforeach()

    if(_PRELOAD_ENV_LIBS)
        string(REPLACE ";" ":" _PRELOAD_ENV "LD_PRELOAD=${_PRELOAD_ENV_LIBS}")
    endif()

    set(${_VAR}
        "${_PRELOAD_ENV}"
        PARENT_SCOPE)
endfunction()

# function for getting the LD_LIBRARY_PATH environment variable
function(rocprofiler_samples_get_ld_library_path_env _VAR)

    set(_LDLIB_PATH "LD_LIBRARY_PATH=")
    foreach(_TARG ${ARGN})
        if(NOT TARGET ${_TARG})
            message(
                FATAL_ERROR
                    "rocprofiler_samples_get_ld_library_path_env: '${_TARG}' is not a valid target"
                )
        endif()
        string(APPEND _LDLIB_PATH "$<TARGET_FILE_DIR:${_TARG}>:")
    endforeach()
    # append the environments current LD_LIBRARY_PATH
    string(APPEND _LDLIB_PATH "$ENV{LD_LIBRARY_PATH}")

    set(${_VAR}
        "${_LDLIB_PATH}"
        PARENT_SCOPE)
endfunction()
