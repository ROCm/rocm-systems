##########################################################################################
#
#  Handles the versioning settings
#
##########################################################################################

include_guard(DIRECTORY)

##########################################################################################
#
#  Standardizes the reading of version files
#
##########################################################################################

function(rocm_versioning_read_version_file PROJECT_NAME PROJECT_VERSION_PREFIX
         VERSION_FILE)
    if(NOT EXISTS "${VERSION_FILE}" AND NOT EXISTS
                                        "${CMAKE_CURRENT_LIST_DIR}/${VERSION_FILE}")
        message(FATAL_ERROR "[${PROJECT_NAME}] Version file '${VERSION_FILE}' not found.")
    elseif(EXISTS "${CMAKE_CURRENT_LIST_DIR}/${VERSION_FILE}")
        set(VERSION_FILE "${CMAKE_CURRENT_LIST_DIR}/${VERSION_FILE}")
    endif()

    file(READ "${VERSION_FILE}" FULL_VERSION_STRING)
    string(STRIP "${FULL_VERSION_STRING}" FULL_VERSION_STRING)
    string(REGEX REPLACE "([0-9]+)\.([0-9]+)\.([0-9]+)(.*)" "\\1.\\2.\\3"
                         PARSED_FULL_VERSION_STRING "${FULL_VERSION_STRING}")
    string(REGEX REPLACE "([0-9]+)\.([0-9]+)\.([0-9]+)(.*)" "\\1"
                         PARSED_MAJOR_VERSION_STRING "${FULL_VERSION_STRING}")
    string(REGEX REPLACE "([0-9]+)\.([0-9]+)\.([0-9]+)(.*)" "\\2"
                         PARSED_MINOR_VERSION_STRING "${FULL_VERSION_STRING}")
    string(REGEX REPLACE "([0-9]+)\.([0-9]+)\.([0-9]+)(.*)" "\\3"
                         PARSED_PATCH_VERSION_STRING "${FULL_VERSION_STRING}")
    set(${PROJECT_VERSION_PREFIX}_VERSION_MAJOR
        "${PARSED_MAJOR_VERSION_STRING}"
        PARENT_SCOPE)
    set(${PROJECT_VERSION_PREFIX}_VERSION_MINOR
        "${PARSED_MINOR_VERSION_STRING}"
        PARENT_SCOPE)
    set(${PROJECT_VERSION_PREFIX}_VERSION_PATCH
        "${PARSED_PATCH_VERSION_STRING}"
        PARENT_SCOPE)
    set(${PROJECT_VERSION_PREFIX}_VERSION
        "${PARSED_MAJOR_VERSION_STRING}.${PARSED_MINOR_VERSION_STRING}.${PARSED_PATCH_VERSION_STRING}"
        PARENT_SCOPE)

    if(FULL_VERSION_STRING MATCHES
       "([0-9]+)\.([0-9]+)\.([0-9]+)([a-zA-Z0-9\\.\\-]+)(\n|\r)(.*)")
        string(REGEX REPLACE "([0-9]+)\.([0-9]+)\.([0-9]+)([a-zA-Z0-9\\.\\-]+)(\n|\r)(.*)"
                             "\\4" PARSED_BUILD_VERSION_STRING "${FULL_VERSION_STRING}")

        if(NOT PARSED_BUILD_VERSION_STRING STREQUAL "")
            string(REGEX REPLACE "^(\\.|\\-)(.*)" "\\2" PARSED_BUILD_VERSION_STRING
                                 "${PARSED_BUILD_VERSION_STRING}")
        endif()
    else()
        set(PARSED_BUILD_VERSION_STRING "")
    endif()

    if(FULL_VERSION_STRING MATCHES "# hash: ([a-fA-F0-9]+)")
        string(REGEX REPLACE "(.*)# hash: ([a-fA-F0-9]+)(.*)" "\\2"
                             PARSED_HASH_VERSION_STRING "${FULL_VERSION_STRING}")
    else()
        set(PARSED_HASH_VERSION_STRING "")
    endif()

    set(${PROJECT_VERSION_PREFIX}_VERSION_BUILD
        "${PARSED_BUILD_VERSION_STRING}"
        PARENT_SCOPE)

    set(${PROJECT_VERSION_PREFIX}_FULL_VERSION_STRING
        "${PARSED_MAJOR_VERSION_STRING}.${PARSED_MINOR_VERSION_STRING}.${PARSED_PATCH_VERSION_STRING}${PARSED_BUILD_VERSION_STRING}"
        PARENT_SCOPE)

    set(${PROJECT_VERSION_PREFIX}_VERSION_HASH
        "${PARSED_HASH_VERSION_STRING}"
        PARENT_SCOPE)
endfunction()

##########################################################################################
#
#  Ensures the installation of version files
#
##########################################################################################

function(rocm_versioning_install_version_files PROJECT_NAME)
    if(NOT CMAKE_INSTALL_DATAROOTDIR)
        include(GNUInstallDirs)
    endif()

    cmake_parse_arguments(_ARG "" "COMPONENT" "FILES" ${ARGN})

    install(
        FILES ${_ARG_FILES}
        DESTINATION "${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_NAME}"
        COMPONENT ${_ARG_COMPONENT})
endfunction()

##########################################################################################
#
#  Below produces a developer warning because project(...) is called indirectly through a macro
#
##########################################################################################

# macro(rocm_project PROJECT_NAME) cmake_parse_arguments( _ARG ""
# "VERSION_FILE;VERSION_VAR;COMPAT_VERSION;SPDX_LICENSE;DESCRIPTION;HOMEPAGE_URL;LANGUAGES"
# "FILES" ${ARGN})

# if(NOT _ARG_VERSION_VAR) string(TOUPPER "${PROJECT_NAME}" _ARG_VERSION_VAR)
# string(REPLACE "-" "_" _ARG_VERSION_VAR "${_ARG_VERSION_VAR}") endif()

# rocm_versioning_read_version_file(${PROJECT_NAME} ${_ARG_VERSION_VAR}
# "${_ARG_VERSION_FILE}")

# set(_PROJECT_ARGS)

# macro(_rocprofiler_versioning_add_project_args _LABEL _VALUE) if(_VALUE) list(APPEND
# _PROJECT_ARGS ${_LABEL} "${_VALUE}" ${ARGN}) endif() endmacro()

# if(_ARG_COMPAT_VERSION) _rocprofiler_versioning_add_project_args(COMPAT_VERSION
# "${_ARG_COMPAT_VERSION}") endif()

# if(_ARG_SPDX_LICENSE) _rocprofiler_versioning_add_project_args(SPDX_LICENSE
# "${_ARG_SPDX_LICENSE}") endif()

# if(_ARG_DESCRIPTION) _rocprofiler_versioning_add_project_args(DESCRIPTION
# "${_ARG_DESCRIPTION}") endif()

# if(_ARG_HOMEPAGE_URL) _rocprofiler_versioning_add_project_args(HOMEPAGE_URL
# "${_ARG_HOMEPAGE_URL}") endif()

# if(_ARG_LANGUAGES) _rocprofiler_versioning_add_project_args(LANGUAGES
# "${_ARG_LANGUAGES}") else() _rocprofiler_versioning_add_project_args(LANGUAGES "C"
# "CXX") endif()

# message(STATUS "Calling project...") project(${PROJECT_NAME} VERSION
# ${${_ARG_VERSION_VAR}_VERSION} ${_PROJECT_ARGS})

# unset(_PROJECT_ARGS) endmacro()
