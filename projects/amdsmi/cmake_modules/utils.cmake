################################################################################
## Copyright (C) Advanced Micro Devices. All rights reserved.
##
## Permission is hereby granted, free of charge, to any person obtaining a copy of
## this software and associated documentation files (the "Software"), to deal in
## the Software without restriction, including without limitation the rights to
## use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
## the Software, and to permit persons to whom the Software is furnished to do so,
## subject to the following conditions:
##
## The above copyright notice and this permission notice shall be included in all
## copies or substantial portions of the Software.
##
## THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
## IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
## FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
## COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
## IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
## CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
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

    string(REGEX MATCHALL "[0-9]+" VERSIONS ${VERSION_STRING})
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

function(get_version_from_file REL_FILE_PATH ITEM)
    set(FILE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/${REL_FILE_PATH}")
    set(OUTPUT_ITEM "0")

    if(EXISTS "${FILE_PATH}")
        file(READ ${FILE_PATH} file_contents)
        string(REGEX MATCHALL "AMDSMI_LIB_VERSION_${ITEM} *[0-9]+" OUTPUT_STR "${file_contents}")
        list(LENGTH OUTPUT_STR OUTPUT_STR_LENGTH)
        if(${OUTPUT_STR_LENGTH} GREATER 0)
            string(REGEX MATCH "[0-9]+" OUTPUT_ITEM "${OUTPUT_STR}")
        endif()
    endif()

    set(${ITEM} "${OUTPUT_ITEM}" PARENT_SCOPE)
endfunction()

# Parses file for a pattern and replaces the value
# associated with that pattern with a specified value
# Replaces VERSION(MAJOR.MINOR.RELEASE) with updated values
function(update_version_in_file REL_FILE_PATH DEFAULT_VERSION PAT1 PAT2 PAT3)
    get_version_from_file(${REL_FILE_PATH} "MAJOR")
    get_version_from_file(${REL_FILE_PATH} "MINOR")
    get_version_from_file(${REL_FILE_PATH} "RELEASE")
    set(FILE_VERSION "${MAJOR}.${MINOR}.${RELEASE}")

    if(DEFAULT_VERSION VERSION_GREATER FILE_VERSION)
        set(FILE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/${REL_FILE_PATH}")
        if(EXISTS "${FILE_PATH}")
            parse_version(${DEFAULT_VERSION})
            file(READ ${FILE_PATH} file_contents_new)

            string(REGEX REPLACE "${PAT1}MAJOR${PAT2} *[0-9]*" "${PAT1}MAJOR${PAT3}${VERSION_MAJOR}" file_contents
                                 "${file_contents_new}")
            string(REGEX REPLACE "${PAT1}MINOR${PAT2} *[0-9]*" "${PAT1}MINOR${PAT3}${VERSION_MINOR}" file_contents_new
                                 "${file_contents}")
            string(REGEX REPLACE "${PAT1}RELEASE${PAT2} *[0-9]*" "${PAT1}RELEASE${PAT3}${VERSION_PATCH}" file_contents
                                 "${file_contents_new}")

            file(WRITE ${FILE_PATH} "${file_contents}")
        endif()
        set(VERSION_STRING "${DEFAULT_VERSION}" PARENT_SCOPE)
    else()
        set(VERSION_STRING "${FILE_VERSION}" PARENT_SCOPE)
    endif()
endfunction()

## Gets the current version of the repository
## using versioning tags and git describe.
## Passes back a packaging version string
## and a library version string.
function(get_version_from_tag DEFAULT_VERSION_STRING VERSION_PREFIX GIT)
    parse_version(${DEFAULT_VERSION_STRING})
    set(DEFAULT_VERSION_MAJOR "${VERSION_MAJOR}")
    set(DEFAULT_VERSION_MINOR "${VERSION_MINOR}")
    set(DEFAULT_VERSION_PATCH "${VERSION_PATCH}")

    if(GIT)
        execute_process(
            COMMAND git tag --list --sort=-version:refname "${VERSION_PREFIX}*"
            COMMAND head -n 1
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
            OUTPUT_VARIABLE GIT_TAG_STRING
            OUTPUT_STRIP_TRAILING_WHITESPACE RESULTS_VARIABLE RESULTS)
        if(GIT_TAG_STRING)
            parse_version(${GIT_TAG_STRING})
        endif()
    endif()

    if(VERSION_STRING VERSION_GREATER DEFAULT_VERSION_STRING)
        set(VERSION_STRING "${VERSION_STRING}" PARENT_SCOPE)
        set(VERSION_MAJOR "${VERSION_MAJOR}" PARENT_SCOPE)
        set(VERSION_MINOR "${VERSION_MINOR}" PARENT_SCOPE)
        set(VERSION_PATCH "${VERSION_PATCH}" PARENT_SCOPE)
    else()
        set(VERSION_STRING "${DEFAULT_VERSION_STRING}" PARENT_SCOPE)
        set(VERSION_MAJOR "${DEFAULT_VERSION_MAJOR}" PARENT_SCOPE)
        set(VERSION_MINOR "${DEFAULT_VERSION_MINOR}" PARENT_SCOPE)
        set(VERSION_PATCH "${DEFAULT_VERSION_PATCH}" PARENT_SCOPE)
    endif()
endfunction()

function(num_change_since_prev_pkg VERSION_PREFIX)
    find_program(get_commits NAMES version_util.sh PATHS ${CMAKE_CURRENT_SOURCE_DIR}/cmake_modules)
    if(get_commits)
        execute_process(
            COMMAND ${get_commits} -c ${VERSION_PREFIX}
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
            OUTPUT_VARIABLE NUM_COMMITS
            OUTPUT_STRIP_TRAILING_WHITESPACE
            RESULT_VARIABLE RESULT)

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

function(get_package_version_number DEFAULT_VERSION_STRING VERSION_PREFIX GIT)
    parse_version(${DEFAULT_VERSION_STRING})
    num_change_since_prev_pkg(${VERSION_PREFIX})
    set(PKG_VERSION_STR "${VERSION_STRING}.${NUM_COMMITS}")
    if(DEFINED ENV{ROCM_BUILD_ID})
        set(VERSION_ID $ENV{ROCM_BUILD_ID})
    else()
        set(VERSION_ID "local-build-0")
    endif()

    set(PKG_VERSION_STR "${PKG_VERSION_STR}-${VERSION_ID}")

    if(GIT)
        execute_process(
            COMMAND git rev-parse --short HEAD
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
            OUTPUT_VARIABLE VERSION_HASH
            OUTPUT_STRIP_TRAILING_WHITESPACE
            RESULT_VARIABLE RESULT)
        if(${RESULT} EQUAL 0)
            # Check for dirty workspace.
            execute_process(COMMAND git diff --quiet WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                            RESULT_VARIABLE RESULT)
            if(${RESULT} EQUAL 1)
                set(VERSION_HASH "${VERSION_HASH}-dirty")
            endif()
        else()
            set(VERSION_HASH "unknown")
        endif()
    else()
        set(VERSION_HASH "unknown")
    endif()
    set(PKG_VERSION_STR "${PKG_VERSION_STR}-${VERSION_HASH}")
    set(PKG_VERSION_STR ${PKG_VERSION_STR} PARENT_SCOPE)
    set(PKG_VERSION_HASH ${VERSION_HASH} PARENT_SCOPE)
    set(CPACK_PACKAGE_VERSION_MAJOR ${VERSION_MAJOR} PARENT_SCOPE)
    set(CPACK_PACKAGE_VERSION_MINOR ${VERSION_MINOR} PARENT_SCOPE)
    set(CPACK_PACKAGE_VERSION_PATCH ${VERSION_PATCH} PARENT_SCOPE)
endfunction()
