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
         VERSION_FILE VERSIONING_SPEC_FILE)
    if(NOT EXISTS "${VERSION_FILE}" AND NOT EXISTS
                                        "${CMAKE_CURRENT_LIST_DIR}/${VERSION_FILE}")
        message(FATAL_ERROR "[${PROJECT_NAME}] Version file '${VERSION_FILE}' not found.")
    elseif(EXISTS "${CMAKE_CURRENT_LIST_DIR}/${VERSION_FILE}")
        set(VERSION_FILE "${CMAKE_CURRENT_LIST_DIR}/${VERSION_FILE}")
    endif()

    if(NOT EXISTS "${VERSIONING_SPEC_FILE}"
       AND NOT EXISTS "${CMAKE_CURRENT_LIST_DIR}/${VERSIONING_SPEC_FILE}")
        message(
            FATAL_ERROR
                "[${PROJECT_NAME}] Versioning YAML spec file '${VERSIONING_SPEC_FILE}' not found."
            )
    elseif(EXISTS "${CMAKE_CURRENT_LIST_DIR}/${VERSIONING_SPEC_FILE}")
        set(VERSIONING_SPEC_FILE "${CMAKE_CURRENT_LIST_DIR}/${VERSIONING_SPEC_FILE}")
    endif()

    file(READ "${VERSIONING_SPEC_FILE}" FULL_VERSIONING_SPEC_STRING)

    string(STRIP "${FULL_VERSIONING_SPEC_STRING}" FULL_VERSIONING_SPEC_STRING)
    string(REGEX REPLACE ".*(name:)([ \t\"]*)([a-zA-Z0-9_-]+)([ \t\"]*)(.*)" "\\3"
                         PARSED_PROJECT_NAME "${FULL_VERSIONING_SPEC_STRING}")

    if(NOT PARSED_PROJECT_NAME STREQUAL PROJECT_NAME)
        message(
            FATAL_ERROR
                "[${PROJECT_NAME}] Mismatched project name in versioning spec file '${VERSIONING_SPEC_FILE}': "
                "expected '${PROJECT_NAME}', got '${PARSED_PROJECT_NAME}'")
    endif()

    macro(_rocm_versioning_set _VAR _VALUE)
        if("GLOBAL" IN_LIST ARGN)
            set(${_VAR}
                "${_VALUE}"
                CACHE STRING "version variable for project ${PROJECT_NAME}" FORCE)
        else()
            set(${_VAR}
                "${_VALUE}"
                PARENT_SCOPE)
        endif()
    endmacro()

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

    _rocm_versioning_set(${PROJECT_VERSION_PREFIX}_VERSION_MAJOR
                         "${PARSED_MAJOR_VERSION_STRING}")
    _rocm_versioning_set(${PROJECT_VERSION_PREFIX}_VERSION_MINOR
                         "${PARSED_MINOR_VERSION_STRING}")
    _rocm_versioning_set(${PROJECT_VERSION_PREFIX}_VERSION_PATCH
                         "${PARSED_PATCH_VERSION_STRING}")
    _rocm_versioning_set(
        ${PROJECT_VERSION_PREFIX}_VERSION
        "${PARSED_MAJOR_VERSION_STRING}.${PARSED_MINOR_VERSION_STRING}.${PARSED_PATCH_VERSION_STRING}"
        )

    if(FULL_VERSION_STRING MATCHES
       "([0-9]+)\.([0-9]+)\.([0-9]+)([a-zA-Z0-9\\.\\-]+)(\n|\r)(.*)")
        string(REGEX REPLACE "([0-9]+)\.([0-9]+)\.([0-9]+)([a-zA-Z0-9\\.\\-]+)(\n|\r)(.*)"
                             "\\4" PARSED_BUILD_VERSION_STRING "${FULL_VERSION_STRING}")

        if(NOT PARSED_BUILD_VERSION_STRING STREQUAL "")
            string(REGEX REPLACE "^(\\.|\\-)(.*)" "\\2" PARSED_BUILD_VERSION_STRING
                                 "${PARSED_BUILD_VERSION_STRING}")
        endif()
    else()
        _rocm_versioning_set(PARSED_BUILD_VERSION_STRING "")
    endif()

    if(FULL_VERSION_STRING MATCHES "# hash: ([a-fA-F0-9]+)")
        string(REGEX REPLACE "(.*)# hash: ([a-fA-F0-9]+)(.*)" "\\2"
                             PARSED_HASH_VERSION_STRING "${FULL_VERSION_STRING}")
    else()
        _rocm_versioning_set(PARSED_HASH_VERSION_STRING "")
    endif()

    _rocm_versioning_set(${PROJECT_VERSION_PREFIX}_VERSION_BUILD
                         "${PARSED_BUILD_VERSION_STRING}")

    _rocm_versioning_set(
        ${PROJECT_VERSION_PREFIX}_FULL_VERSION_STRING
        "${PARSED_MAJOR_VERSION_STRING}.${PARSED_MINOR_VERSION_STRING}.${PARSED_PATCH_VERSION_STRING}${PARSED_BUILD_VERSION_STRING}"
        )

    _rocm_versioning_set(${PROJECT_VERSION_PREFIX}_VERSION_HASH
                         "${PARSED_HASH_VERSION_STRING}")
endfunction()

##########################################################################################
#
#  Copies the version files to the build directory
#
##########################################################################################

function(rocm_versioning_configure_version_files)
    cmake_parse_arguments(_ARG "" "DESTINATION" "FILES" ${ARGN})

    if(NOT _ARG_DESTINATION)
        set(_ARG_DESTINATION "${PROJECT_BINARY_DIR}")
    endif()

    foreach(_FILE ${_ARG_FILES})
        configure_file("${_FILE}" "${_ARG_DESTINATION}/${_FILE}" COPYONLY)
    endforeach()
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
