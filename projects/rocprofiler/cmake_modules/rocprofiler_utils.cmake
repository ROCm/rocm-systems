################################################################################
# Copyright (c) 2018 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
################################################################################

# Parses the VERSION_STRING variable and places the first, second and third number values
# in the major, minor and patch variables.
function(rocprofiler_parse_version VERSION_STRING)

    string(FIND ${VERSION_STRING} "-" STRING_INDEX)

    if(${STRING_INDEX} GREATER -1)
        math(EXPR STRING_INDEX "${STRING_INDEX} + 1")
        string(SUBSTRING ${VERSION_STRING} ${STRING_INDEX} -1 VERSION_BUILD)
    endif()

    string(REGEX MATCHALL "[0123456789]+" VERSIONS ${VERSION_STRING})
    list(LENGTH VERSIONS VERSION_COUNT)

    if(${VERSION_COUNT} GREATER 0)
        list(GET VERSIONS 0 MAJOR)
        set(VERSION_MAJOR
            ${MAJOR}
            PARENT_SCOPE)
        set(TEMP_VERSION_STRING "${MAJOR}")
    endif()

    if(${VERSION_COUNT} GREATER 1)
        list(GET VERSIONS 1 MINOR)
        set(VERSION_MINOR
            ${MINOR}
            PARENT_SCOPE)
        set(TEMP_VERSION_STRING "${TEMP_VERSION_STRING}.${MINOR}")
    endif()

    if(${VERSION_COUNT} GREATER 2)
        list(GET VERSIONS 2 PATCH)
        set(VERSION_PATCH
            ${PATCH}
            PARENT_SCOPE)
        set(TEMP_VERSION_STRING "${TEMP_VERSION_STRING}.${PATCH}")
    endif()

    if(DEFINED VERSION_BUILD)
        set(VERSION_BUILD
            "${VERSION_BUILD}"
            PARENT_SCOPE)
    endif()

    set(VERSION_STRING
        "${TEMP_VERSION_STRING}"
        PARENT_SCOPE)

endfunction()

# Gets the current version of the repository using versioning tags and git describe.
# Passes back a packaging version string and a library version string.
function(rocprofiler_get_version DEFAULT_VERSION_STRING)

    rocprofiler_parse_version(${DEFAULT_VERSION_STRING})

    find_program(GIT NAMES git)

    if(GIT)

        execute_process(
            COMMAND "git describe --dirty --long --match [0-9]* 2>/dev/null"
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
            OUTPUT_VARIABLE GIT_TAG_STRING
            OUTPUT_STRIP_TRAILING_WHITESPACE
            RESULT_VARIABLE RESULT)

        if(${RESULT} EQUAL 0)

            rocprofiler_parse_version(${GIT_TAG_STRING})

        endif()

    endif()

    set(VERSION_STRING
        "${VERSION_STRING}"
        PARENT_SCOPE)
    set(VERSION_MAJOR
        "${VERSION_MAJOR}"
        PARENT_SCOPE)
    set(VERSION_MINOR
        "${VERSION_MINOR}"
        PARENT_SCOPE)
    set(VERSION_PATCH
        "${VERSION_PATCH}"
        PARENT_SCOPE)
    set(VERSION_BUILD
        "${VERSION_BUILD}"
        PARENT_SCOPE)

endfunction()

# ----------------------------------------------------------------------------------------#
# macro rocprofiler_checkout_git_submodule()
#
# Run "git submodule update" if a file in a submodule does not exist
#
# ARGS: RECURSIVE (option) -- add "--recursive" flag RELATIVE_PATH (one value) --
# typically the relative path to submodule from PROJECT_SOURCE_DIR WORKING_DIRECTORY (one
# value) -- (default: PROJECT_SOURCE_DIR) TEST_FILE (one value) -- file to check for
# (default: CMakeLists.txt) ADDITIONAL_CMDS (many value) -- any addition commands to pass
#
function(ROCPROFILER_CHECKOUT_GIT_SUBMODULE)
    # parse args
    cmake_parse_arguments(
        CHECKOUT "RECURSIVE"
        "RELATIVE_PATH;WORKING_DIRECTORY;TEST_FILE;REPO_URL;REPO_BRANCH"
        "ADDITIONAL_CMDS" ${ARGN})

    if(NOT CHECKOUT_WORKING_DIRECTORY)
        set(CHECKOUT_WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
    endif()

    if(NOT CHECKOUT_TEST_FILE)
        set(CHECKOUT_TEST_FILE "CMakeLists.txt")
    endif()

    find_package(Git)
    set(_DIR "${CHECKOUT_WORKING_DIRECTORY}/${CHECKOUT_RELATIVE_PATH}")
    # ensure the (possibly empty) directory exists
    if(NOT EXISTS "${_DIR}")
        if(NOT CHECKOUT_REPO_URL)
            message(FATAL_ERROR "submodule directory does not exist")
        endif()
    endif()

    # if this file exists --> project has been checked out if not exists --> not been
    # checked out
    set(_TEST_FILE "${_DIR}/${CHECKOUT_TEST_FILE}")
    # assuming a .gitmodules file exists
    set(_SUBMODULE "${PROJECT_SOURCE_DIR}/.gitmodules")

    set(_TEST_FILE_EXISTS OFF)
    if(EXISTS "${_TEST_FILE}" AND NOT IS_DIRECTORY "${_TEST_FILE}")
        set(_TEST_FILE_EXISTS ON)
    endif()

    if(_TEST_FILE_EXISTS)
        return()
    endif()

    find_package(Git REQUIRED)

    set(_SUBMODULE_EXISTS OFF)
    if(EXISTS "${_SUBMODULE}" AND NOT IS_DIRECTORY "${_SUBMODULE}")
        set(_SUBMODULE_EXISTS ON)
    endif()

    set(_HAS_REPO_URL OFF)
    if(NOT "${CHECKOUT_REPO_URL}" STREQUAL "")
        set(_HAS_REPO_URL ON)
    endif()

    # if the module has not been checked out
    if(NOT _TEST_FILE_EXISTS AND _SUBMODULE_EXISTS)
        # perform the checkout
        execute_process(
            COMMAND ${GIT_EXECUTABLE} submodule update --init ${_RECURSE}
                    ${CHECKOUT_ADDITIONAL_CMDS} ${CHECKOUT_RELATIVE_PATH}
            WORKING_DIRECTORY ${CHECKOUT_WORKING_DIRECTORY}
            RESULT_VARIABLE RET)

        # check the return code
        if(RET GREATER 0)
            set(_CMD "${GIT_EXECUTABLE} submodule update --init ${_RECURSE}
                ${CHECKOUT_ADDITIONAL_CMDS} ${CHECKOUT_RELATIVE_PATH}")
            message(STATUS "function(rocprofiler_checkout_git_submodule) failed.")
            message(FATAL_ERROR "Command: \"${_CMD}\"")
        else()
            set(_TEST_FILE_EXISTS ON)
        endif()
    endif()

    if(NOT _TEST_FILE_EXISTS AND _HAS_REPO_URL)
        message(
            STATUS
                "Cloning '${CHECKOUT_REPO_URL}' into ${CHECKOUT_WORKING_DIRECTORY}/${CHECKOUT_RELATIVE_PATH}..."
            )

        # remove the existing directory
        if(EXISTS "${_DIR}")
            execute_process(COMMAND ${CMAKE_COMMAND} -E remove_directory ${_DIR})
        endif()

        # perform the checkout
        execute_process(
            COMMAND ${GIT_EXECUTABLE} clone ${CHECKOUT_ADDITIONAL_CMDS}
                    ${CHECKOUT_REPO_URL} ${CHECKOUT_RELATIVE_PATH}
            WORKING_DIRECTORY ${CHECKOUT_WORKING_DIRECTORY}
            RESULT_VARIABLE RET_CLONE)

        if(NOT RET_CLONE EQUAL 0)
            message(
                SEND_ERROR
                    "Failed to clone ${CHECKOUT_REPO_URL} into ${CHECKOUT_WORKING_DIRECTORY}/${CHECKOUT_RELATIVE_PATH}"
                )
            return()
        endif()

        if(CHECKOUT_REPO_BRANCH)
            execute_process(
                COMMAND ${GIT_EXECUTABLE} checkout ${CHECKOUT_REPO_BRANCH}
                WORKING_DIRECTORY ${CHECKOUT_WORKING_DIRECTORY}/${CHECKOUT_RELATIVE_PATH}
                RESULT_VARIABLE RET_BRANCH)

            if(NOT RET_BRANCH EQUAL 0)
                message(
                    SEND_ERROR
                        "Failed to checkout '${CHECKOUT_REPO_BRANCH}' for ${CHECKOUT_REPO_URL} in ${CHECKOUT_WORKING_DIRECTORY}/${CHECKOUT_RELATIVE_PATH}"
                    )
                return()
            endif()
        endif()

        # perform the submodule update
        if(CHECKOUT_RECURSIVE
           AND EXISTS "${_DIR}"
           AND IS_DIRECTORY "${_DIR}")
            execute_process(
                COMMAND ${GIT_EXECUTABLE} submodule update --init ${_RECURSE}
                WORKING_DIRECTORY ${_DIR}
                RESULT_VARIABLE RET_RECURSIVE)
            if(NOT RET_RECURSIVE EQUAL 0)
                message(
                    SEND_ERROR
                        "Failed to update submodules for ${CHECKOUT_REPO_URL} in ${CHECKOUT_WORKING_DIRECTORY}/${CHECKOUT_RELATIVE_PATH}"
                    )
                return()
            endif()
        endif()

        set(_TEST_FILE_EXISTS ON)
    endif()

    if(NOT EXISTS "${_TEST_FILE}" OR NOT _TEST_FILE_EXISTS)
        message(
            FATAL_ERROR
                "Error checking out submodule: '${CHECKOUT_RELATIVE_PATH}' to '${_DIR}'")
    endif()
endfunction()

## Configure Copyright File for Debian Package
function( configure_pkg PACKAGE_NAME_T COMPONENT_NAME_T PACKAGE_VERSION_T MAINTAINER_NM_T MAINTAINER_EMAIL_T)
    # Check If Debian Platform
    find_file (DEBIAN debian_version debconf.conf PATHS /etc)
    if(DEBIAN)
        set( BUILD_DEBIAN_PKGING_FLAG ON CACHE BOOL "Internal Status Flag to indicate Debian Packaging Build" FORCE )
        set_debian_pkg_cmake_flags( ${PACKAGE_NAME_T} ${PACKAGE_VERSION_T}
                                    ${MAINTAINER_NM_T} ${MAINTAINER_EMAIL_T} )

        # Create debian directory in build tree
        file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/DEBIAN")

        # Configure the copyright file
        configure_file(
            "${CMAKE_SOURCE_DIR}/DEBIAN/copyright.in"
            "${CMAKE_BINARY_DIR}/DEBIAN/copyright"
            @ONLY
        )

        # Install copyright file
        install ( FILES "${CMAKE_BINARY_DIR}/DEBIAN/copyright"
                DESTINATION "${CMAKE_INSTALL_DOCDIR}"
                COMPONENT ${COMPONENT_NAME_T} )

        # Configure the changelog file
        configure_file(
            "${CMAKE_SOURCE_DIR}/CHANGELOG.md"
            "${CMAKE_BINARY_DIR}/DEBIAN/CHANGELOG.md"
            @ONLY
        )

        # Install Change Log
        find_program ( DEB_GZIP_EXEC gzip )
        if(EXISTS "${CMAKE_BINARY_DIR}/DEBIAN/CHANGELOG.md" )
            execute_process(
            COMMAND ${DEB_GZIP_EXEC} -f -n -9 "${CMAKE_BINARY_DIR}/DEBIAN/CHANGELOG.md"
            WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/DEBIAN"
            RESULT_VARIABLE result
            OUTPUT_VARIABLE output
            ERROR_VARIABLE error
            )
            if(NOT ${result} EQUAL 0)
                message(FATAL_ERROR "Failed to compress: ${error}")
            endif()
            install ( FILES "${CMAKE_BINARY_DIR}/DEBIAN/${DEB_CHANGELOG_INSTALL_FILENM}"
                    DESTINATION ${CMAKE_INSTALL_DOCDIR}
                    COMPONENT ${COMPONENT_NAME_T})
        endif()

    else()
        # License file
        install ( FILES ${LICENSE_FILE}
            DESTINATION ${CMAKE_INSTALL_DOCDIR} RENAME LICENSE.txt
            COMPONENT ${COMPONENT_NAME_T})
    endif()
endfunction()

# Set variables for changelog and copyright
# For Debian specific Packages
function( set_debian_pkg_cmake_flags DEB_PACKAGE_NAME_T DEB_PACKAGE_VERSION_T DEB_MAINTAINER_NM_T DEB_MAINTAINER_EMAIL_T )
    # Setting configure flags
    set( DEB_PACKAGE_NAME             "${DEB_PACKAGE_NAME_T}" CACHE STRING "Debian Package Name" )
    set( DEB_PACKAGE_VERSION          "${DEB_PACKAGE_VERSION_T}" CACHE STRING "Debian Package Version String" )
    set( DEB_MAINTAINER_NAME          "${DEB_MAINTAINER_NM_T}" CACHE STRING "Debian Package Maintainer Name" )
    set( DEB_MAINTAINER_EMAIL         "${DEB_MAINTAINER_EMAIL_T}" CACHE STRING "Debian Package Maintainer Email" )
    set( DEB_COPYRIGHT_YEAR           "2025" CACHE STRING "Debian Package Copyright Year" )
    set( DEB_LICENSE                  "MIT" CACHE STRING "Debian Package License Type" )
    set( DEB_CHANGELOG_INSTALL_FILENM "CHANGELOG.md.gz" CACHE STRING "Debian Package ChangeLog File Name" )

    # Get TimeStamp
    find_program( DEB_DATE_TIMESTAMP_EXEC date )
    set ( DEB_TIMESTAMP_FORMAT_OPTION "-R" )
    execute_process (
        COMMAND ${DEB_DATE_TIMESTAMP_EXEC} ${DEB_TIMESTAMP_FORMAT_OPTION}
        OUTPUT_VARIABLE TIMESTAMP_T
    )
    set( DEB_TIMESTAMP                "${TIMESTAMP_T}" CACHE STRING "Current Time Stamp for Copyright/Changelog" )

    message(STATUS "DEB_PACKAGE_NAME             : ${DEB_PACKAGE_NAME}" )
    message(STATUS "DEB_PACKAGE_VERSION          : ${DEB_PACKAGE_VERSION}" )
    message(STATUS "DEB_MAINTAINER_NAME          : ${DEB_MAINTAINER_NAME}" )
    message(STATUS "DEB_MAINTAINER_EMAIL         : ${DEB_MAINTAINER_EMAIL}" )
    message(STATUS "DEB_COPYRIGHT_YEAR           : ${DEB_COPYRIGHT_YEAR}" )
    message(STATUS "DEB_LICENSE                  : ${DEB_LICENSE}" )
    message(STATUS "DEB_TIMESTAMP                : ${DEB_TIMESTAMP}" )
    message(STATUS "DEB_CHANGELOG_INSTALL_FILENM : ${DEB_CHANGELOG_INSTALL_FILENM}" )
endfunction()
