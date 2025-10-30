################################################################################
##
## The University of Illinois/NCSA
## Open Source License (NCSA)
##
## Copyright (c) 2014-2017, Advanced Micro Devices, Inc. All rights reserved.
##
## Developed by:
##
##                 AMD Research and AMD HSA Software Development
##
##                 Advanced Micro Devices, Inc.
##
##                 www.amd.com
##
## Permission is hereby granted, free of charge, to any person obtaining a copy
## of this software and associated documentation files (the "Software"), to
## deal with the Software without restriction, including without limitation
## the rights to use, copy, modify, merge, publish, distribute, sublicense,
## and#or sell copies of the Software, and to permit persons to whom the
## Software is furnished to do so, subject to the following conditions:
##
##  - Redistributions of source code must retain the above copyright notice,
##    this list of conditions and the following disclaimers.
##  - Redistributions in binary form must reproduce the above copyright
##    notice, this list of conditions and the following disclaimers in
##    the documentation and#or other materials provided with the distribution.
##  - Neither the names of Advanced Micro Devices, Inc,
##    nor the names of its contributors may be used to endorse or promote
##    products derived from this Software without specific prior written
##    permission.
##
## THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
## IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
## FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
## THE CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
## OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
## ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
## DEALINGS WITH THE SOFTWARE.
##
################################################################################

## Parses the VERSION_STRING variable and places
## the first, second and third number values in
## the major, minor and patch variables.
function(parse_version VERSION_STRING)
    string(FIND ${VERSION_STRING} "-" STRING_INDEX)

    if(${STRING_INDEX} GREATER -1)
        math(EXPR STRING_INDEX "${STRING_INDEX} + 1")
        string(SUBSTRING ${VERSION_STRING} ${STRING_INDEX} -1 VERSION_BUILD)
    endif()

    string(REGEX MATCHALL "[0123456789]+" VERSIONS ${VERSION_STRING})
    list(LENGTH VERSIONS VERSION_COUNT)

    if(${VERSION_COUNT} GREATER 0)
        list(GET VERSIONS 0 MAJOR)
        set(VERSION_MAJOR ${MAJOR} PARENT_SCOPE)
        set(TEMP_VERSION_STRING "${MAJOR}")
    endif()

    if(${VERSION_COUNT} GREATER 1)
        list(GET VERSIONS 1 MINOR)
        set(VERSION_MINOR ${MINOR} PARENT_SCOPE)
        set(TEMP_VERSION_STRING "${TEMP_VERSION_STRING}.${MINOR}")
    endif()

    if(${VERSION_COUNT} GREATER 2)
        list(GET VERSIONS 2 PATCH)
        set(VERSION_PATCH ${PATCH} PARENT_SCOPE)
        set(TEMP_VERSION_STRING "${TEMP_VERSION_STRING}.${PATCH}")
    endif()

    set(VERSION_STRING "${TEMP_VERSION_STRING}" PARENT_SCOPE)
endfunction()

## Gets the current version of the repository
## using versioning tags and git describe.
## Passes back a packaging version string
## and a library version string.
function(get_version_from_tag DEFAULT_VERSION_STRING VERSION_PREFIX GIT)
    parse_version(${DEFAULT_VERSION_STRING})

    if(GIT)
        execute_process(
            COMMAND git describe --tags --dirty --long --match ${VERSION_PREFIX}-[0-9.]*
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
            OUTPUT_VARIABLE GIT_TAG_STRING
            OUTPUT_STRIP_TRAILING_WHITESPACE
            RESULT_VARIABLE RESULT
        )
        if(${RESULT} EQUAL 0)
            parse_version(${GIT_TAG_STRING})
        endif()
    endif()

    set(VERSION_STRING "${VERSION_STRING}" PARENT_SCOPE)
    set(VERSION_MAJOR "${VERSION_MAJOR}" PARENT_SCOPE)
    set(VERSION_MINOR "${VERSION_MINOR}" PARENT_SCOPE)
    set(VERSION_PATCH "${VERSION_PATCH}" PARENT_SCOPE)
endfunction()

function(num_change_since_prev_pkg VERSION_PREFIX)
    find_program(get_commits NAMES version_util.sh PATHS ${CMAKE_CURRENT_SOURCE_DIR}/cmake_modules)
    if(get_commits)
        execute_process(
            COMMAND ${get_commits} -c ${VERSION_PREFIX}
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
            OUTPUT_VARIABLE NUM_COMMITS
            OUTPUT_STRIP_TRAILING_WHITESPACE
            RESULT_VARIABLE RESULT
        )

        set(NUM_COMMITS "${NUM_COMMITS}" PARENT_SCOPE)

        if(${RESULT} EQUAL 0)
            message("${NUM_COMMITS} were found since previous release")
        else()
            message("Unable to determine number of commits since previous release")
        endif()
    else()
        message("WARNING: Didn't find version_util.sh")
        set(NUM_COMMITS "unknown" PARENT_SCOPE)
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
