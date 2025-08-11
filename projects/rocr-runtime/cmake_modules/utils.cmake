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

function( get_path LIB CACHED_PATH HELP )

    set( options "")
    set( oneValueArgs RESULT )
    set( multiValueArgs HINTS NAMES )
    cmake_parse_arguments(ARGS "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN} )

    # Search for canary file.
    if( ${LIB} )
        find_library( FULLPATH NAMES ${ARGS_NAMES} HINTS ${${CACHED_PATH}} ${ARGS_HINTS} )
    else()
        find_file( FULLPATH NAMES ${ARGS_NAMES} HINTS ${${CACHED_PATH}} ${ARGS_HINTS} )
    endif()
    set( RESULT (NOT ${FULLPATH} MATCHES NOTFOUND) )
    
    # Extract path
    get_filename_component ( DIRPATH ${FULLPATH} DIRECTORY )
    
    # Check path against cache
    if( NOT "${${CACHED_PATH}}" STREQUAL "" )
        if ( NOT "${${CACHED_PATH}}" STREQUAL "${DIRPATH}" )
            message(WARNING "${CACHED_PATH} may be incorrect." )
            set( DIRPATH ${${CACHED_PATH}} )
        endif()
    elseif(NOT ${RESULT})
        message(WARNING "${CACHED_PATH} not located during path search.")
    endif()

    # Set cache variable and help text
    set( ${CACHED_PATH} ${DIRPATH} CACHE PATH ${HELP} FORCE )
    unset( FULLPATH CACHE )

    # Return success flag
    if( NOT ${ARGS_RESULT} STREQUAL "" )
        set( ${ARGS_RESULT} ${RESULT} PARENT_SCOPE)
    endif()

endfunction()

## Searches for a file using include paths and stores the path to that file in the cache
## using the cached value if set.  Search paths are optional.  Returns success in RESULT.
## get_include_path(<VAR> NAMES name1 [name2...] [HINTS path1 [path2 ... ENV var]] [RESULT <var>]
macro( get_include_path CACHED_PATH HELP )
    get_path( 0 ${ARGV} )
endmacro()

## Searches for a file using library paths and stores the path to that file in the cache
## using the cached value if set.  Search paths are optional.  Returns success in RESULT.
## get_library_path(<VAR> NAMES name1 [name2...] [HINTS path1 [path2 ... ENV var]] [RESULT <var>]
macro( get_library_path CACHED_PATH HELP )
    get_path( 1 ${ARGV} )
endmacro()

## Parses the VERSION_STRING variable and places
## the first, second and third number values in
## the major, minor and patch variables.
function( parse_version VERSION_STRING )

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
    endif ()

    if ( ${VERSION_COUNT} GREATER 1 )
        list ( GET VERSIONS 1 MINOR )
        set ( VERSION_MINOR ${MINOR} PARENT_SCOPE )
    endif ()

    if ( ${VERSION_COUNT} GREATER 2 )
        list ( GET VERSIONS 2 PATCH )
        set ( VERSION_PATCH ${PATCH} PARENT_SCOPE )
    endif ()

endfunction ()

## Gets the current version of the repository
## using versioning tags and git describe.
## Passes back a packaging version string
## and a library version string.
function ( get_version DEFAULT_VERSION_STRING )

    set( VERSION_JOB "local-build" )
    set( VERSION_COMMIT_COUNT 0 )
    set( VERSION_HASH "unknown" )

    find_program( GIT NAMES git )

    if( GIT )

        #execute_process ( COMMAND git describe --tags --dirty --long
        #                  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        #                  OUTPUT_VARIABLE GIT_TAG_STRING
        #                  OUTPUT_STRIP_TRAILING_WHITESPACE
        #                  RESULT_VARIABLE RESULT )

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

    # Build automation IDs
    if(DEFINED ENV{ROCM_BUILD_ID})
        set( VERSION_JOB $ENV{ROCM_BUILD_ID} )
    endif()

    parse_version(${DEFAULT_VERSION_STRING})

    set( VERSION_MAJOR  "${VERSION_MAJOR}" PARENT_SCOPE )
    set( VERSION_MINOR  "${VERSION_MINOR}" PARENT_SCOPE )
    set( VERSION_PATCH  "${VERSION_PATCH}" PARENT_SCOPE )
    set( VERSION_COMMIT_COUNT "${VERSION_COMMIT_COUNT}" PARENT_SCOPE )
    set( VERSION_HASH "${VERSION_HASH}" PARENT_SCOPE )
    set( VERSION_JOB "${VERSION_JOB}" PARENT_SCOPE )

    #message("${VERSION_MAJOR}" )
    #message("${VERSION_MINOR}" )
    #message("${VERSION_PATCH}" )
    #message("${VERSION_COMMIT_COUNT}")
    #message("${VERSION_HASH}")
    #message("${VERSION_JOB}")

endfunction()

## Collects subdirectory names and returns them in a list
function ( listsubdirs DIRPATH SUBDIRECTORIES )
    file( GLOB CONTENTS RELATIVE ${DIRPATH} "${DIRPATH}/*" )
    set ( FOLDERS, "" )
    foreach( ITEM IN LISTS CONTENTS)
        if( IS_DIRECTORY "${DIRPATH}/${ITEM}" )
            list( APPEND FOLDERS ${ITEM} )
        endif()
    endforeach()
    set (${SUBDIRECTORIES} ${FOLDERS} PARENT_SCOPE)
endfunction()

## Sets el7 flag to be true
function (Checksetel7 EL7_DISTRO)
execute_process(COMMAND rpm --eval %{?dist}
                 RESULT_VARIABLE PROC_RESULT
                 OUTPUT_VARIABLE EVAL_RESULT
                 OUTPUT_STRIP_TRAILING_WHITESPACE)
message("RESULT_VARIABLE ${PROC_RESULT} OUTPUT_VARIABLE: ${EVAL_RESULT}")
if (PROC_RESULT EQUAL "0" AND NOT EVAL_RESULT STREQUAL "")
  if ("${EVAL_RESULT}" STREQUAL ".el7")
     set (${EL7_DISTRO} TRUE PARENT_SCOPE)
  endif()
endif()
endfunction()
