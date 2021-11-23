# include guard
include_guard(DIRECTORY)

# MacroUtilities - useful macros and functions for generic tasks
#

cmake_policy(PUSH)
cmake_policy(SET CMP0054 NEW)
cmake_policy(SET CMP0057 NEW)

include(CMakeDependentOption)
include(CMakeParseArguments)

# -----------------------------------------------------------------------
# message which handles HOSTTRACE_QUIET_CONFIG settings
# -----------------------------------------------------------------------
#
function(HOSTTRACE_MESSAGE TYPE)
    if(NOT HOSTTRACE_QUIET_CONFIG)
        message(${TYPE} "[hosttrace] ${ARGN}")
    endif()
endfunction()

# -----------------------------------------------------------------------
# Save a set of variables with the given prefix
# -----------------------------------------------------------------------
macro(HOSTTRACE_SAVE_VARIABLES _PREFIX)
    # parse args
    cmake_parse_arguments(
        SAVE
        "" # options
        "CONDITION" # single value args
        "VARIABLES" # multiple value args
        ${ARGN})
    if(DEFINED SAVE_CONDITION AND NOT "${SAVE_CONDITION}" STREQUAL "")
        if(${SAVE_CONDITION})
            foreach(_VAR ${SAVE_VARIABLES})
                if(DEFINED ${_VAR})
                    set(${_PREFIX}_${_VAR} "${${_VAR}}")
                else()
                    message(AUTHOR_WARNING "${_VAR} is not defined")
                endif()
            endforeach()
        endif()
    else()
        foreach(_VAR ${SAVE_VARIABLES})
            if(DEFINED ${_VAR})
                set(${_PREFIX}_${_VAR} "${${_VAR}}")
            else()
                message(AUTHOR_WARNING "${_VAR} is not defined")
            endif()
        endforeach()
    endif()
    unset(SAVE_CONDITION)
    unset(SAVE_VARIABLES)
endmacro()

# -----------------------------------------------------------------------
# Restore a set of variables with the given prefix
# -----------------------------------------------------------------------
macro(HOSTTRACE_RESTORE_VARIABLES _PREFIX)
    # parse args
    cmake_parse_arguments(
        RESTORE
        "" # options
        "CONDITION" # single value args
        "VARIABLES" # multiple value args
        ${ARGN})
    if(DEFINED RESTORE_CONDITION AND NOT "${RESTORE_CONDITION}" STREQUAL "")
        if(${RESTORE_CONDITION})
            foreach(_VAR ${RESTORE_VARIABLES})
                if(DEFINED ${_PREFIX}_${_VAR})
                    set(${_VAR} ${${_PREFIX}_${_VAR}})
                    unset(${_PREFIX}_${_VAR})
                else()
                    message(AUTHOR_WARNING "${_PREFIX}_${_VAR} is not defined")
                endif()
            endforeach()
        endif()
    else()
        foreach(_VAR ${RESTORE_VARIABLES})
            if(DEFINED ${_PREFIX}_${_VAR})
                set(${_VAR} ${${_PREFIX}_${_VAR}})
                unset(${_PREFIX}_${_VAR})
            else()
                message(AUTHOR_WARNING "${_PREFIX}_${_VAR} is not defined")
            endif()
        endforeach()
    endif()
    unset(RESTORE_CONDITION)
    unset(RESTORE_VARIABLES)
endmacro()

# -----------------------------------------------------------------------
# function - hosttrace_capitalize - make a string capitalized (first letter is capital)
# usage: capitalize("SHARED" CShared) message(STATUS "-- CShared is \"${CShared}\"") $ --
# CShared is "Shared"
function(HOSTTRACE_CAPITALIZE str var)
    # make string lower
    string(TOLOWER "${str}" str)
    string(SUBSTRING "${str}" 0 1 _first)
    string(TOUPPER "${_first}" _first)
    string(SUBSTRING "${str}" 1 -1 _remainder)
    string(CONCAT str "${_first}" "${_remainder}")
    set(${var}
        "${str}"
        PARENT_SCOPE)
endfunction()

# ------------------------------------------------------------------------------#
# function add_hosttrace_test_target()
#
# Creates a target which runs ctest but depends on all the tests being built.
#
function(ADD_HOSTTRACE_TEST_TARGET)
    if(NOT TARGET hosttrace-test)
        add_custom_target(
            hosttrace-test
            COMMAND ${CMAKE_COMMAND} --build ${PROJECT_BINARY_DIR} --target test
            WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
            COMMENT "Running tests...")
    endif()
endfunction()

# ----------------------------------------------------------------------------------------#
# macro hosttrace_checkout_git_submodule()
#
# Run "git submodule update" if a file in a submodule does not exist
#
# ARGS: RECURSIVE (option) -- add "--recursive" flag RELATIVE_PATH (one value) --
# typically the relative path to submodule from PROJECT_SOURCE_DIR WORKING_DIRECTORY (one
# value) -- (default: PROJECT_SOURCE_DIR) TEST_FILE (one value) -- file to check for
# (default: CMakeLists.txt) ADDITIONAL_CMDS (many value) -- any addition commands to pass
#
function(HOSTTRACE_CHECKOUT_GIT_SUBMODULE)
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

    # default assumption
    if(NOT CHECKOUT_REPO_BRANCH)
        set(CHECKOUT_REPO_BRANCH "master")
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
            message(STATUS "function(hosttrace_checkout_git_submodule) failed.")
            message(FATAL_ERROR "Command: \"${_CMD}\"")
        else()
            set(_TEST_FILE_EXISTS ON)
        endif()
    endif()

    if(NOT _TEST_FILE_EXISTS AND _HAS_REPO_URL)
        message(
            STATUS "Checking out '${CHECKOUT_REPO_URL}' @ '${CHECKOUT_REPO_BRANCH}'...")

        # remove the existing directory
        if(EXISTS "${_DIR}")
            execute_process(COMMAND ${CMAKE_COMMAND} -E remove_directory ${_DIR})
        endif()

        # perform the checkout
        execute_process(
            COMMAND
                ${GIT_EXECUTABLE} clone -b ${CHECKOUT_REPO_BRANCH}
                ${CHECKOUT_ADDITIONAL_CMDS} ${CHECKOUT_REPO_URL} ${CHECKOUT_RELATIVE_PATH}
            WORKING_DIRECTORY ${CHECKOUT_WORKING_DIRECTORY}
            RESULT_VARIABLE RET)

        # perform the submodule update
        if(CHECKOUT_RECURSIVE
           AND EXISTS "${_DIR}"
           AND IS_DIRECTORY "${_DIR}")
            execute_process(
                COMMAND ${GIT_EXECUTABLE} submodule update --init ${_RECURSE}
                WORKING_DIRECTORY ${_DIR}
                RESULT_VARIABLE RET)
        endif()

        # check the return code
        if(RET GREATER 0)
            set(_CMD
                "${GIT_EXECUTABLE} clone -b ${CHECKOUT_REPO_BRANCH}
                ${CHECKOUT_ADDITIONAL_CMDS} ${CHECKOUT_REPO_URL} ${CHECKOUT_RELATIVE_PATH}"
                )
            message(STATUS "function(hosttrace_checkout_git_submodule) failed.")
            message(FATAL_ERROR "Command: \"${_CMD}\"")
        else()
            set(_TEST_FILE_EXISTS ON)
        endif()
    endif()

    if(NOT EXISTS "${_TEST_FILE}" OR NOT _TEST_FILE_EXISTS)
        message(
            FATAL_ERROR
                "Error checking out submodule: '${CHECKOUT_RELATIVE_PATH}' to '${_DIR}'")
    endif()

endfunction()

# ----------------------------------------------------------------------------------------#
# try to find a package quietly
#
function(HOSTTRACE_TEST_FIND_PACKAGE PACKAGE_NAME OUTPUT_VAR)
    cmake_parse_arguments(PACKAGE "" "" "UNSET" ${ARGN})
    find_package(${PACKAGE_NAME} QUIET ${PACKAGE_UNPARSED_ARGUMENTS})
    if(NOT ${PACKAGE_NAME}_FOUND)
        set(${OUTPUT_VAR}
            OFF
            PARENT_SCOPE)
    else()
        set(${OUTPUT_VAR}
            ON
            PARENT_SCOPE)
    endif()
    foreach(_ARG ${PACKAGE_UNSET} FIND_PACKAGE_MESSAGE_DETAILS_${PACKAGE_NAME})
        unset(${_ARG} CACHE)
    endforeach()
endfunction()

# ----------------------------------------------------------------------------------------#
# macro to add an interface lib
#
macro(HOSTTRACE_ADD_INTERFACE_LIBRARY _TARGET)
    add_library(${_TARGET} INTERFACE)
    add_library(${PROJECT_NAME}::${_TARGET} ALIAS ${_TARGET})
    install(
        TARGETS ${_TARGET}
        DESTINATION ${CMAKE_INSTALL_LIBDIR}
        EXPORT ${PROJECT_NAME}-library-depends
        OPTIONAL)
    if(NOT "${ARGN}" STREQUAL "")
        set_property(GLOBAL APPEND PROPERTY ${PROJECT_NAME}_CMAKE_INTERFACE_DOC
                                            "${PROJECT_NAME}::${_TARGET}` | ${ARGN} |")
    endif()
endmacro()

function(HOSTTRACE_ADD_RPATH)
    set(_DIRS)
    foreach(_ARG ${ARGN})
        if(EXISTS "${_ARG}" AND IS_DIRECTORY "${_ARG}")
            list(APPEND _DIRS "${_ARG}")
        endif()
        get_filename_component(_DIR "${_ARG}" DIRECTORY)
        if(EXISTS "${_DIR}" AND IS_DIRECTORY "${_DIR}")
            list(APPEND _DIRS "${_DIR}")
        endif()
    endforeach()
    if(_DIRS)
        list(REMOVE_DUPLICATES _DIRS)
        string(REPLACE ";" ":" _RPATH "${_DIRS}")
        # message(STATUS "\n\tRPATH additions: ${_RPATH}\n")
        set(CMAKE_INSTALL_RPATH
            "${CMAKE_INSTALL_RPATH}:${_RPATH}"
            PARENT_SCOPE)
    endif()
endfunction()

# -----------------------------------------------------------------------
# function add_feature(<NAME> <DOCSTRING>) Add a project feature, whose activation is
# specified by the existence of the variable <NAME>, to the list of enabled/disabled
# features, plus a docstring describing the feature
#
function(HOSTTRACE_ADD_FEATURE _var _description)
    set(EXTRA_DESC "")
    foreach(currentArg ${ARGN})
        if(NOT "${currentArg}" STREQUAL "${_var}"
           AND NOT "${currentArg}" STREQUAL "${_description}"
           AND NOT "${currentArg}" STREQUAL "CMAKE_DEFINE"
           AND NOT "${currentArg}" STREQUAL "DOC")
            set(EXTRA_DESC "${EXTA_DESC}${currentArg}")
        endif()
    endforeach()

    set_property(GLOBAL APPEND PROPERTY ${PROJECT_NAME}_FEATURES ${_var})
    set_property(GLOBAL PROPERTY ${_var}_DESCRIPTION "${_description}${EXTRA_DESC}")

    if("CMAKE_DEFINE" IN_LIST ARGN)
        set_property(GLOBAL APPEND PROPERTY ${PROJECT_NAME}_CMAKE_DEFINES
                                            "${_var} @${_var}@")
        if(HOSTTRACE_BUILD_DOCS)
            set_property(
                GLOBAL APPEND PROPERTY ${PROJECT_NAME}_CMAKE_OPTIONS_DOC
                                       "${_var}` | ${_description}${EXTRA_DESC} |")
        endif()
    elseif("DOC" IN_LIST ARGN AND HOSTTRACE_BUILD_DOCS)
        set_property(GLOBAL APPEND PROPERTY ${PROJECT_NAME}_CMAKE_OPTIONS_DOC
                                            "${_var}` | ${_description}${EXTRA_DESC} |")
    endif()
endfunction()

# ----------------------------------------------------------------------------------------#
# function add_option(<OPTION_NAME> <DOCSRING> <DEFAULT_SETTING> [NO_FEATURE]) Add an
# option and add as a feature if NO_FEATURE is not provided
#
function(HOSTTRACE_ADD_OPTION _NAME _MESSAGE _DEFAULT)
    option(${_NAME} "${_MESSAGE}" ${_DEFAULT})
    if("NO_FEATURE" IN_LIST ARGN)
        mark_as_advanced(${_NAME})
    else()
        hosttrace_add_feature(${_NAME} "${_MESSAGE}")
        if(HOSTTRACE_BUILD_DOCS)
            set_property(GLOBAL APPEND PROPERTY ${PROJECT_NAME}_CMAKE_OPTIONS_DOC
                                                "${_NAME}` | ${_MESSAGE} |")
        endif()
    endif()
    if("ADVANCED" IN_LIST ARGN)
        mark_as_advanced(${_NAME})
    endif()
    if("CMAKE_DEFINE" IN_LIST ARGN)
        set_property(GLOBAL APPEND PROPERTY ${PROJECT_NAME}_CMAKE_DEFINES ${_NAME})
    endif()
endfunction()

# ----------------------------------------------------------------------------------------#
# function print_enabled_features() Print enabled  features plus their docstrings.
#
function(HOSTTRACE_PRINT_ENABLED_FEATURES)
    set(_basemsg "The following features are defined/enabled (+):")
    set(_currentFeatureText "${_basemsg}")
    get_property(_features GLOBAL PROPERTY ${PROJECT_NAME}_FEATURES)
    if(NOT "${_features}" STREQUAL "")
        list(REMOVE_DUPLICATES _features)
        list(SORT _features)
    endif()
    foreach(_feature ${_features})
        if(${_feature})
            # add feature to text
            set(_currentFeatureText "${_currentFeatureText}\n     ${_feature}")
            # get description
            get_property(_desc GLOBAL PROPERTY ${_feature}_DESCRIPTION)
            # print description, if not standard ON/OFF, print what is set to
            if(_desc)
                if(NOT "${${_feature}}" STREQUAL "ON" AND NOT "${${_feature}}" STREQUAL
                                                          "TRUE")
                    set(_currentFeatureText
                        "${_currentFeatureText}: ${_desc} -- [\"${${_feature}}\"]")
                else()
                    string(REGEX REPLACE "^${PROJECT_NAME}_USE_" "" _feature_tmp
                                         "${_feature}")
                    string(TOLOWER "${_feature_tmp}" _feature_tmp_l)
                    hosttrace_capitalize("${_feature_tmp}" _feature_tmp_c)
                    foreach(_var _feature _feature_tmp _feature_tmp_l _feature_tmp_c)
                        set(_ver "${${${_var}}_VERSION}")
                        if(NOT "${_ver}" STREQUAL "")
                            set(_desc "${_desc} -- [found version ${_ver}]")
                            break()
                        endif()
                        unset(_ver)
                    endforeach()
                    set(_currentFeatureText "${_currentFeatureText}: ${_desc}")
                endif()
                set(_desc NOTFOUND)
            endif()
        endif()
    endforeach()

    if(NOT "${_currentFeatureText}" STREQUAL "${_basemsg}")
        message(STATUS "${_currentFeatureText}\n")
    endif()
endfunction()

# ----------------------------------------------------------------------------------------#
# function print_disabled_features() Print disabled features plus their docstrings.
#
function(HOSTTRACE_PRINT_DISABLED_FEATURES)
    set(_basemsg "The following features are NOT defined/enabled (-):")
    set(_currentFeatureText "${_basemsg}")
    get_property(_features GLOBAL PROPERTY ${PROJECT_NAME}_FEATURES)
    if(NOT "${_features}" STREQUAL "")
        list(REMOVE_DUPLICATES _features)
        list(SORT _features)
    endif()
    foreach(_feature ${_features})
        if(NOT ${_feature})
            set(_currentFeatureText "${_currentFeatureText}\n     ${_feature}")

            get_property(_desc GLOBAL PROPERTY ${_feature}_DESCRIPTION)

            if(_desc)
                set(_currentFeatureText "${_currentFeatureText}: ${_desc}")
                set(_desc NOTFOUND)
            endif(_desc)
        endif()
    endforeach(_feature)

    if(NOT "${_currentFeatureText}" STREQUAL "${_basemsg}")
        message(STATUS "${_currentFeatureText}\n")
    endif()
endfunction()

# ----------------------------------------------------------------------------------------#
# function print_features() Print all features plus their docstrings.
#
function(HOSTTRACE_PRINT_FEATURES)
    hosttrace_print_enabled_features()
    hosttrace_print_disabled_features()
endfunction()

# ----------------------------------------------------------------------------------------#
# this function is provided to easily select which files use alternative compiler:
#
# GLOBAL      --> all files TARGET      --> all files in a target SOURCE      --> specific
# source files DIRECTORY   --> all files in directory PROJECT     --> all files/targets in
# a project/subproject
#
function(hosttrace_custom_compilation)
    cmake_parse_arguments(COMP "GLOBAL;PROJECT" "COMPILER" "DIRECTORY;TARGET;SOURCE"
                          ${ARGN})

    # find hosttrace_launch_compiler
    find_program(
        HOSTTRACE_COMPILE_LAUNCHER
        NAMES hosttrace_launch_compiler
        HINTS ${PROJECT_SOURCE_DIR} ${CMAKE_SOURCE_DIR}
        PATHS ${PROJECT_SOURCE_DIR} ${CMAKE_SOURCE_DIR}
        PATH_SUFFIXES scripts bin)

    if(NOT COMP_COMPILER)
        message(FATAL_ERROR "hosttrace_custom_compilation not provided COMPILER argument")
    endif()

    if(NOT HOSTTRACE_COMPILE_LAUNCHER)
        message(
            FATAL_ERROR
                "hosttrace could not find 'hosttrace_launch_compiler'. Please set '-DHOSTTRACE_COMPILE_LAUNCHER=/path/to/launcher'"
            )
    endif()

    if(COMP_GLOBAL)
        # if global, don't bother setting others
        set_property(
            GLOBAL
            PROPERTY
                RULE_LAUNCH_COMPILE
                "${HOSTTRACE_COMPILE_LAUNCHER} ${COMP_COMPILER} ${CMAKE_CXX_COMPILER}")
        set_property(
            GLOBAL
            PROPERTY
                RULE_LAUNCH_LINK
                "${HOSTTRACE_COMPILE_LAUNCHER} ${COMP_COMPILER} ${CMAKE_CXX_COMPILER}")
    else()
        foreach(_TYPE PROJECT DIRECTORY TARGET SOURCE)
            # make project/subproject scoping easy, e.g.
            # hosttrace_custom_compilation(PROJECT) after project(...)
            if("${_TYPE}" STREQUAL "PROJECT" AND COMP_${_TYPE})
                list(APPEND COMP_DIRECTORY ${PROJECT_SOURCE_DIR})
                unset(COMP_${_TYPE})
            endif()
            # set the properties if defined
            if(COMP_${_TYPE})
                foreach(_VAL ${COMP_${_TYPE}})
                    set_property(
                        ${_TYPE} ${_VAL}
                        PROPERTY
                            RULE_LAUNCH_COMPILE
                            "${HOSTTRACE_COMPILE_LAUNCHER} ${COMP_COMPILER} ${CMAKE_CXX_COMPILER}"
                        )
                    set_property(
                        ${_TYPE} ${_VAL}
                        PROPERTY
                            RULE_LAUNCH_LINK
                            "${HOSTTRACE_COMPILE_LAUNCHER} ${COMP_COMPILER} ${CMAKE_CXX_COMPILER}"
                        )
                endforeach()
            endif()
        endforeach()
    endif()
endfunction()

cmake_policy(POP)
