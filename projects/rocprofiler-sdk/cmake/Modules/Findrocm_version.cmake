#[=======================================================================[.rst:
Findrocm_version
---------------

Search the <ROCM_PATH>/.info/version* files to determine the version of ROCm

Use this module by invoking find_package with the form::

  find_package(rocm_version
    [version] [EXACT]
    [REQUIRED])

This module finds the version info for ROCm.  The cached variables are::

  rocm_version_FOUND             - Whether the ROCm versioning was found
  rocm_version_FULL_VERSION      - The exact string from `<ROCM_PATH>/.info/version` or similar
  rocm_version_MAJOR_VERSION     - Major version, e.g. 4 in 4.5.2.100-40502
  rocm_version_MINOR_VERSION     - Minor version, e.g. 5 in 4.5.2.100-40502
  rocm_version_PATCH_VERSION     - Patch version, e.g. 2 in 4.5.2.100-40502
  rocm_version_TWEAK_VERSION     - Tweak version, e.g. 100 in 4.5.2.100-40502
  rocm_version_REVISION_VERSION  - Revision version, e.g. 40502 in 4.5.2.100-40502.
  rocm_version_EPOCH_VERSION     - See deb-version for a description of epochs. Epochs are used when versioning system change
  rocm_version_CANONICAL_VERSION - `[<EPOCH>:]<MAJOR>.<MINOR>.<MINOR>[.<TWEAK>][-<REVISION>]`
  rocm_version_NUMERIC_VERSION   - e.g. `10000*<MAJOR> + 100*<MINOR> + <PATCH>`, e.g. 40502 for ROCm 4.5.2
  rocm_version_TRIPLE_VERSION    - e.g. `<MAJOR>.<MINOR>.<PATCH>`, e.g. 4.5.2 for ROCm 4.5.2

These variables are relevant for the find procedure::

  rocm_version_DEBUG             - Print info about processing
  rocm_version_VERSION_FILE      - `<FILE>` to read from in `<ROCM_PATH>/.info/<FILE>`, e.g. `version`, `version-dev`, `version-hip-libraries`, etc.
                                  It may also be a full path
  rocm_version_DIR               - Root location for <ROCM_PATH>
#]=======================================================================]

set(rocm_version_VARIABLES
    EPOCH
    MAJOR
    MINOR
    PATCH
    TWEAK
    REVISION
    TRIPLE
    NUMERIC
    CANONICAL
    FULL)

function(ROCM_VERSION_MESSAGE _TYPE)
    if(rocm_version_DEBUG)
        message(${_TYPE} "[rocm_version] ${ARGN}")
    endif()
endfunction()

# read a .info/version* file and propagate the variables to the calling scope
function(ROCM_VERSION_COMPUTE FULL_VERSION_STRING _VAR_PREFIX)

    # remove any line endings
    string(REGEX REPLACE "(\n|\r)" "" FULL_VERSION_STRING "${FULL_VERSION_STRING}")

    # store the full version so it can be set later
    set(FULL_VERSION "${FULL_VERSION_STRING}")

    # get number and remove from full version string
    string(REGEX REPLACE "([0-9]+)\:(.*)" "\\1" EPOCH_VERSION "${FULL_VERSION_STRING}")
    string(REGEX REPLACE "([0-9]+)\:(.*)" "\\2" FULL_VERSION_STRING
                         "${FULL_VERSION_STRING}")

    if(EPOCH_VERSION STREQUAL FULL_VERSION)
        set(EPOCH_VERSION)
    endif()

    # get number and remove from full version string
    string(REGEX REPLACE "([0-9]+)(.*)" "\\1" MAJOR_VERSION "${FULL_VERSION_STRING}")
    string(REGEX REPLACE "([0-9]+)(.*)" "\\2" FULL_VERSION_STRING
                         "${FULL_VERSION_STRING}")

    # get number and remove from full version string
    string(REGEX REPLACE "\.([0-9]+)(.*)" "\\1" MINOR_VERSION "${FULL_VERSION_STRING}")
    string(REGEX REPLACE "\.([0-9]+)(.*)" "\\2" FULL_VERSION_STRING
                         "${FULL_VERSION_STRING}")

    # get number and remove from full version string
    string(REGEX REPLACE "\.([0-9]+)(.*)" "\\1" PATCH_VERSION "${FULL_VERSION_STRING}")
    string(REGEX REPLACE "\.([0-9]+)(.*)" "\\2" FULL_VERSION_STRING
                         "${FULL_VERSION_STRING}")

    if(NOT PATCH_VERSION LESS 100)
        set(PATCH_VERSION 0)
    endif()

    # get number and remove from full version string
    string(REGEX REPLACE "\.([0-9]+)(.*)" "\\1" TWEAK_VERSION "${FULL_VERSION_STRING}")
    string(REGEX REPLACE "\.([0-9]+)(.*)" "\\2" FULL_VERSION_STRING
                         "${FULL_VERSION_STRING}")

    # get number
    string(REGEX REPLACE "-([0-9A-Za-z+~]+)" "\\1" REVISION_VERSION
                         "${FULL_VERSION_STRING}")

    set(CANONICAL_VERSION)
    set(_MAJOR_SEP ":")
    set(_MINOR_SEP ".")
    set(_PATCH_SEP ".")
    set(_TWEAK_SEP ".")
    set(_REVISION_SEP "-")

    foreach(_V EPOCH MAJOR MINOR PATCH TWEAK REVISION)
        if(${_V}_VERSION)
            set(CANONICAL_VERSION "${CANONICAL_VERSION}${_${_V}_SEP}${${_V}_VERSION}")
        else()
            set(CANONICAL_VERSION "${CANONICAL_VERSION}${_${_V}_SEP}0")
        endif()
    endforeach()

    set(_MAJOR_SEP "")

    foreach(_V MAJOR MINOR PATCH)
        if(${_V}_VERSION)
            set(TRIPLE_VERSION "${TRIPLE_VERSION}${_${_V}_SEP}${${_V}_VERSION}")
        else()
            set(TRIPLE_VERSION "${TRIPLE_VERSION}${_${_V}_SEP}0")
        endif()
    endforeach()

    math(
        EXPR
        NUMERIC_VERSION
        "(10000 * (${MAJOR_VERSION}+0)) + (100 * (${MINOR_VERSION}+0)) + (${PATCH_VERSION}+0)"
        )

    # propagate to parent scopes
    foreach(_V ${rocm_version_VARIABLES})
        set(${_VAR_PREFIX}_${_V}_VERSION
            ${${_V}_VERSION}
            PARENT_SCOPE)
    endforeach()
endfunction()

# this macro watches for changes in the variables and unsets the remaining cache varaible
# when they change
function(ROCM_VERSION_WATCH_FOR_CHANGE _var)
    set(_rocm_version_watch_var_name rocm_version_WATCH_VALUE_${_var})

    if(DEFINED ${_rocm_version_watch_var_name})
        if("${${_var}}" STREQUAL "${${_rocm_version_watch_var_name}}")
            if(NOT "${${_var}}" STREQUAL "")
                rocm_version_message(STATUS "${_var} :: ${${_var}}")
            endif()

            list(REMOVE_ITEM _REMAIN_VARIABLES ${_var})
            set(_REMAIN_VARIABLES
                "${_REMAIN_VARIABLES}"
                PARENT_SCOPE)
            return()
        else()
            rocm_version_message(
                STATUS
                "${_var} changed :: ${${_rocm_version_watch_var_name}} --> ${${_var}}")

            foreach(_V ${_REMAIN_VARIABLES})
                rocm_version_message(
                    STATUS "${_var} changed :: Unsetting cache variable ${_V}...")
                unset(${_V} CACHE)
            endforeach()
        endif()
    else()
        if(NOT "${${_var}}" STREQUAL "")
            rocm_version_message(STATUS "${_var} :: ${${_var}}")
        endif()
    endif()

    # store the value for the next run
    set(${_rocm_version_watch_var_name}
        "${${_var}}"
        CACHE INTERNAL "Last value of ${_var}" FORCE)
endfunction()

# scope this to a function to avoid leaking local variables
function(ROCM_VERSION_PARSE_VERSION_FILES)

    # the list of variables set by module. when one of these changes, we need to unset the
    # cache variables after it
    set(_ALL_VARIABLES)

    foreach(_V ${rocm_version_VARIABLES})
        list(APPEND _ALL_VARIABLES rocm_version_${_V}_VERSION)
    endforeach()
    set(_REMAIN_VARIABLES ${_ALL_VARIABLES})

    # read a .info/version* file and propagate the variables to the calling scope
    function(ROCM_VERSION_READ_FILE _FILE _VAR_PREFIX)
        file(READ "${_FILE}" FULL_VERSION_STRING LIMIT_COUNT 1)
        rocm_version_compute("${FULL_VERSION_STRING}" "${_VAR_PREFIX}")

        # propagate to parent scopes
        foreach(_V ${rocm_version_VARIABLES})
            set(${_VAR_PREFIX}_${_V}_VERSION
                ${${_VAR_PREFIX}_${_V}_VERSION}
                PARENT_SCOPE)
        endforeach()
    endfunction()

    # search for HIP to set ROCM_PATH if(NOT hip_FOUND) find_package(hip) endif()

    function(COMPUTE_ROCM_VERSION_DIR)
        if(EXISTS "${rocm_version_VERSION_FILE}" AND IS_ABSOLUTE
                                                     "${rocm_version_VERSION_FILE}")
            get_filename_component(_VERSION_DIR "${rocm_version_VERSION_FILE}" PATH)
            get_filename_component(_VERSION_DIR "${_VERSION_DIR}/.." REALPATH)
            set(rocm_version_DIR
                "${_VERSION_DIR}"
                CACHE PATH "Root path to ROCm's .info/${rocm_version_VERSION_FILE}"
                      ${ARGN})
            rocm_version_watch_for_change(rocm_version_DIR)
        endif()
    endfunction()

    if(rocm_version_VERSION_FILE)
        get_filename_component(_VERSION_FILE "${rocm_version_VERSION_FILE}" NAME)
        set(_VERSION_FILES ${_VERSION_FILE})
        compute_rocm_version_dir(FORCE)
    else()
        set(_VERSION_FILES version version-dev version-hip-libraries version-hiprt
                           version-hiprt-devel version-hip-sdk version-libs version-utils)
        rocm_version_message(STATUS "rocm_version version files: ${_VERSION_FILES}")
    endif()

    # convert env to cache if not defined
    foreach(_PATH rocm_version_DIR rocm_version_ROOT rocm_version_ROOT_DIR
                  ROCPROFILER_DEFAULT_ROCM_PATH ROCM_PATH)
        if(NOT DEFINED ${_PATH} AND DEFINED ENV{${_PATH}})
            set(_VAL "$ENV{${_PATH}}")
            get_filename_component(_VAL "${_VAL}" REALPATH)
            set(${_PATH}
                "${_VAL}"
                CACHE PATH "Search path for ROCm version for rocm_version")
        endif()
    endforeach()

    if(rocm_version_DIR)
        set(_PATHS ${rocm_version_DIR})
    else()
        set(_PATHS)
        foreach(
            _DIR
            ${rocm_version_DIR} ${rocm_version_ROOT} ${rocm_version_ROOT_DIR}
            $ENV{CMAKE_PREFIX_PATH} ${CMAKE_PREFIX_PATH} ${ROCPROFILER_DEFAULT_ROCM_PATH}
            ${ROCM_PATH} /opt/rocm)
            if(EXISTS ${_DIR})
                get_filename_component(_ABS_DIR "${_DIR}" REALPATH)
                list(APPEND _PATHS ${_ABS_DIR})
            endif()
        endforeach()
        rocm_version_message(STATUS "rocm_version search paths: ${_PATHS}")
    endif()

    string(REPLACE ":" ";" _PATHS "${_PATHS}")

    foreach(_PATH ${_PATHS})
        foreach(_FILE ${_VERSION_FILES})
            set(_F ${_PATH}/.info/${_FILE})
            if(EXISTS ${_F})
                set(rocm_version_VERSION_FILE
                    "${_F}"
                    CACHE FILEPATH "File with versioning info")
                rocm_version_watch_for_change(rocm_version_VERSION_FILE)
                compute_rocm_version_dir()
            else()
                rocm_version_message(AUTHOR_WARNING "File does not exist: ${_F}")
            endif()
        endforeach()
    endforeach()

    if(EXISTS "${rocm_version_VERSION_FILE}")
        set(_F "${rocm_version_VERSION_FILE}")
        rocm_version_message(STATUS "Reading ${_F}...")
        get_filename_component(_B "${_F}" NAME)
        string(REPLACE "." "_" _B "${_B}")
        string(REPLACE "-" "_" _B "${_B}")
        rocm_version_read_file(${_F} ${_B})

        foreach(_V ${rocm_version_VARIABLES})
            set(_CACHE_VAR rocm_version_${_V}_VERSION)
            set(_LOCAL_VAR ${_B}_${_V}_VERSION)
            set(rocm_version_${_V}_VERSION
                "${${_LOCAL_VAR}}"
                CACHE STRING "ROCm ${_V} version")
            rocm_version_watch_for_change(${_CACHE_VAR})
        endforeach()
    endif()
endfunction()

# execute
rocm_version_parse_version_files()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
    rocm_version
    VERSION_VAR rocm_version_FULL_VERSION
    REQUIRED_VARS rocm_version_FULL_VERSION rocm_version_TRIPLE_VERSION rocm_version_DIR
                  rocm_version_VERSION_FILE)
# don't add major/minor/patch/etc. version variables to required vars because they might
# be zero, which will cause CMake to evaluate it as not set
