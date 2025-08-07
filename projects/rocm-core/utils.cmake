################################################################################
##
## MIT License
##
## Copyright (c) 2017 - 2023 Advanced Micro Devices, Inc. All rights Reserved.
##
## Permission is hereby granted, free of charge, to any person obtaining a copy
## of this software and associated documentation files (the "Software"), to deal
## in the Software without restriction, including without limitation the rights
## to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
## copies of the Software, and to permit persons to whom the Software is
## furnished to do so, subject to the following conditions:
##
## The above copyright notice and this permission notice shall be included in all
## copies or substantial portions of the Software.
##
## THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
## IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
## FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
## AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
## LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
## OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
## SOFTWARE.
##
################################################################################

## Parses the VERSION_STRING variable and places
## the first, second and third number values in
## the major, minor and patch variables.
function( parse_rocm_version VERSION_STRING )

    string ( FIND ${VERSION_STRING} "-" STRING_INDEX )

    if ( ${STRING_INDEX} GREATER -1 )
        math ( EXPR STRING_INDEX "${STRING_INDEX} + 1" )
        string ( SUBSTRING ${VERSION_STRING} ${STRING_INDEX} -1 VERSION_BUILD )
    endif ()

    string ( REGEX MATCHALL "[0123456789]+" VERSIONS ${VERSION_STRING} )
    list ( LENGTH VERSIONS VERSION_COUNT )

    if ( ${VERSION_COUNT} GREATER 0)
        list ( GET VERSIONS 0 MAJOR )
        set ( VERSION_MAJOR ${MAJOR} PARENT_SCOPE )
        set ( TEMP_VERSION_STRING "${MAJOR}" )
    endif ()

    if ( ${VERSION_COUNT} GREATER 1 )
        list ( GET VERSIONS 1 MINOR )
        set ( VERSION_MINOR ${MINOR} PARENT_SCOPE )
        set ( TEMP_VERSION_STRING "${TEMP_VERSION_STRING}.${MINOR}" )
    endif ()

    if ( ${VERSION_COUNT} GREATER 2 )
        list ( GET VERSIONS 2 PATCH )
        set ( VERSION_PATCH ${PATCH} PARENT_SCOPE )
        set ( TEMP_VERSION_STRING "${TEMP_VERSION_STRING}.${PATCH}" )
    endif ()

    if ( ${VERSION_COUNT} GREATER 3 )
        list ( GET VERSIONS 3 POINT )
        set ( VERSION_POINT ${POINT} PARENT_SCOPE )
        set ( TEMP_VERSION_STRING "${TEMP_VERSION_STRING}.${POINT}" )
    endif ()

    if ( DEFINED VERSION_BUILD )
        set ( VERSION_BUILD "${VERSION_BUILD}" PARENT_SCOPE )
    endif ()

    set ( VERSION_STRING "${TEMP_VERSION_STRING}" PARENT_SCOPE )

endfunction ()


## Sets cmake variables which can be derived from existing
function( set_variables )
    set( VERSION_COMMIT_COUNT 0 )
    set( VERSION_HASH "unknown" )

    find_program( GIT NAMES git )

    if( GIT )
        # Get branch commit (common ancestor) of current branch and master branch.
        execute_process(COMMAND git merge-base HEAD origin/HEAD
                        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                        OUTPUT_VARIABLE GIT_MERGE_BASE
                        OUTPUT_STRIP_TRAILING_WHITESPACE
                        RESULT_VARIABLE RESULT )

        if( ${RESULT} EQUAL 0 )
            # Count commits from branch point.
            execute_process(COMMAND git rev-list --count ${GIT_MERGE_BASE}..HEAD
                            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                            OUTPUT_VARIABLE VERSION_COMMIT_COUNT
                            OUTPUT_STRIP_TRAILING_WHITESPACE
                            RESULT_VARIABLE RESULT )
            if(NOT ${RESULT} EQUAL 0 )
                set( VERSION_COMMIT_COUNT 0 )
            endif()
        endif()

        # Get current short hash.
        execute_process(COMMAND git rev-parse --short HEAD
                        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                        OUTPUT_VARIABLE VERSION_HASH
                        OUTPUT_STRIP_TRAILING_WHITESPACE
                        RESULT_VARIABLE RESULT )
        if( ${RESULT} EQUAL 0 )
            # Check for dirty workspace.
            execute_process(COMMAND git diff --quiet
                            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                            RESULT_VARIABLE RESULT )
            if(${RESULT} EQUAL 1)
                set(VERSION_HASH "${VERSION_HASH}-dirty")
            endif()
        else()
            set( VERSION_HASH "unknown" )
        endif()
    endif()

    #set libpatch version
    if(NOT DEFINED ENV{ROCM_LIBPATCH_VERSION})
      set(ROCM_LIBPATCH_VERSION "${VERSION_MAJOR}")
      string(LENGTH ${VERSION_MINOR} LENSTR)
      if(LENSTR EQUAL 1) # length of version cannot be zero hence it would be 1 or greater
         set(ROCM_LIBPATCH_VERSION "${ROCM_LIBPATCH_VERSION}0${VERSION_MINOR}")
      else() # length is greater than 1
         set(ROCM_LIBPATCH_VERSION "${ROCM_LIBPATCH_VERSION}${VERSION_MINOR}")
      endif()

      string(LENGTH ${VERSION_PATCH} LENSTR)
      if(LENSTR EQUAL 1) # length of version cannot be zero hence it would be 1 or greater
         set(ROCM_LIBPATCH_VERSION "${ROCM_LIBPATCH_VERSION}0${VERSION_PATCH}")
      else() # length is greater than 1
         set(ROCM_LIBPATCH_VERSION "${ROCM_LIBPATCH_VERSION}${VERSION_PATCH}")
      endif()

      set(ROCM_LIBPATCH_VERSION "${ROCM_LIBPATCH_VERSION}" PARENT_SCOPE )
    else()
      set (ROCM_LIBPATCH_VERSION $ENV{ROCM_LIBPATCH_VERSION} PARENT_SCOPE )
    endif()

    if ( DEFINED ENV{CPACK_RPM_PACKAGE_RELEASE} )
      set ( CPACK_RPM_PACKAGE_RELEASE ${CPACK_RPM_PACKAGE_RELEASE} PARENT_SCOPE )
    else()
      set ( CPACK_RPM_PACKAGE_RELEASE "local" PARENT_SCOPE )
    endif()

    if ( DEFINED ENV{CPACK_DEBIAN_PACKAGE_RELEASE} )
      set ( CPACK_DEBIAN_PACKAGE_RELEASE ${CPACK_DEBIAN_PACKAGE_RELEASE} PARENT_SCOPE )
    else()
      set ( CPACK_DEBIAN_PACKAGE_RELEASE "local" PARENT_SCOPE )
    endif()

    set( VERSION_COMMIT_COUNT "${VERSION_COMMIT_COUNT}" PARENT_SCOPE )
    set( VERSION_HASH "${VERSION_HASH}" PARENT_SCOPE )
    set( VERSION_BUILD  "${VERSION_BUILD}" PARENT_SCOPE )

    message(STATUS "VERSION_MAJOR : ${VERSION_MAJOR}" )
    message(STATUS "VERSION_MINOR : ${VERSION_MINOR}" )
    message(STATUS "VERSION_PATCH : ${VERSION_PATCH}" )
    message(STATUS "VERSION_POINT : ${VERSION_POINT}" )
    message(STATUS "ROCM_LIBPATCH_VERSION : ${ROCM_LIBPATCH_VERSION}" )
    message(STATUS "VERSION_COMMIT_COUNT : ${VERSION_COMMIT_COUNT}" )
    message(STATUS "VERSION_HASH : ${VERSION_HASH}" )
    message(STATUS "VERSION_BUILD : ${VERSION_BUILD}" )
    message(STATUS "CPACK_DEBIAN_PACKAGE_RELEASE : ${CPACK_DEBIAN_PACKAGE_RELEASE}" )
    message(STATUS "CPACK_RPM_PACKAGE_RELEASE : ${CPACK_RPM_PACKAGE_RELEASE}" )

endfunction()

## Configure Copyright File for Debian Package
function( configure_debian_pkg PACKAGE_NAME_T COMPONENT_NAME_T PACKAGE_VERSION_T MAINTAINER_NM_T MAINTAINER_EMAIL_T)
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
        "${CMAKE_SOURCE_DIR}/DEBIAN/changelog.in"
        "${CMAKE_BINARY_DIR}/DEBIAN/changelog.Debian"
        @ONLY
      )

      if( BUILD_ENABLE_LINTIAN_OVERRIDES )
	if(NOT BUILD_SHARED_LIBS)
	  string(FIND ${DEB_OVERRIDES_INSTALL_FILENM} "static" OUT_VAR1)
	  if(OUT_VAR1 EQUAL -1)
	    set( DEB_OVERRIDES_INSTALL_FILENM "${DEB_OVERRIDES_INSTALL_FILENM}-static" )
          endif()
	else()
          if(ENABLE_ASAN_PACKAGING)
	    string( FIND ${DEB_OVERRIDES_INSTALL_FILENM} "asan" OUT_VAR2)
	    if(OUT_VAR2 EQUAL -1)
	      set( DEB_OVERRIDES_INSTALL_FILENM "${DEB_OVERRIDES_INSTALL_FILENM}-asan" )
	    endif()
          endif()
	endif()
	set( DEB_OVERRIDES_INSTALL_FILENM
		"${DEB_OVERRIDES_INSTALL_FILENM}" CACHE STRING "Debian Package Lintian Override File Name" FORCE)
        # Configure the changelog file
        configure_file(
          "${CMAKE_SOURCE_DIR}/DEBIAN/overrides.in"
          "${CMAKE_BINARY_DIR}/DEBIAN/${DEB_OVERRIDES_INSTALL_FILENM}"
	   FILE_PERMISSIONS OWNER_READ OWNER_WRITE GROUP_READ WORLD_READ
          @ONLY
        )
      endif()

      # Install Change Log 
      find_program ( DEB_GZIP_EXEC gzip )
      if(EXISTS "${CMAKE_BINARY_DIR}/DEBIAN/changelog.Debian" )
        execute_process(
          COMMAND ${DEB_GZIP_EXEC} -n -9 "${CMAKE_BINARY_DIR}/DEBIAN/changelog.Debian"
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
    set( DEB_CHANGELOG_INSTALL_FILENM "changelog.Debian.gz" CACHE STRING "Debian Package ChangeLog File Name" ) 

    if( BUILD_ENABLE_LINTIAN_OVERRIDES )
      set( DEB_OVERRIDES_INSTALL_FILENM "${DEB_PACKAGE_NAME}" CACHE STRING "Debian Package Lintian Override File Name" )
      set( DEB_OVERRIDES_INSTALL_PATH   "/usr/share/lintian/overrides/" CACHE STRING "Deb Pkg Lintian Override Install Loc" )
    endif()

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
