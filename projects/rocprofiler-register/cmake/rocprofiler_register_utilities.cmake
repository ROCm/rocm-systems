# include guard
include_guard(GLOBAL)

# MacroUtilities - useful macros and functions for generic tasks
#

cmake_policy(PUSH)
cmake_policy(SET CMP0054 NEW)
cmake_policy(SET CMP0057 NEW)

include(CMakeDependentOption)
include(CMakeParseArguments)

# -----------------------------------------------------------------------
# message which handles ROCPROFILER_REGISTER_QUIET_CONFIG settings
# -----------------------------------------------------------------------
#
function(ROCPROFILER_REGISTER_MESSAGE TYPE)
    if(NOT ROCPROFILER_REGISTER_QUIET_CONFIG)
        message(${TYPE} "[rocprofiler-register] ${ARGN}")
    endif()
endfunction()

# -----------------------------------------------------------------------
# Save a set of variables with the given prefix
# -----------------------------------------------------------------------
macro(ROCPROFILER_REGISTER_SAVE_VARIABLES _PREFIX)
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
macro(ROCPROFILER_REGISTER_RESTORE_VARIABLES _PREFIX)
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
# function - rocprofiler_register_capitalize - make a string capitalized (first letter is
# capital) usage: capitalize("SHARED" CShared) message(STATUS "-- CShared is
# \"${CShared}\"") $ -- CShared is "Shared"
function(ROCPROFILER_REGISTER_CAPITALIZE str var)
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
# function rocprofiler_register_strip_target(<TARGET> [FORCE] [EXPLICIT])
#
# Creates a post-build command which strips a binary. FORCE flag will override
#
function(ROCPROFILER_REGISTER_STRIP_TARGET)
    cmake_parse_arguments(STRIP "FORCE;EXPLICIT" "" "ARGS" ${ARGN})

    list(LENGTH STRIP_UNPARSED_ARGUMENTS NUM_UNPARSED)

    if(NUM_UNPARSED EQUAL 1)
        set(_TARGET "${STRIP_UNPARSED_ARGUMENTS}")
    else()
        rocprofiler_register_message(
            FATAL_ERROR
            "rocprofiler_register_strip_target cannot deduce target from \"${ARGN}\"")
    endif()

    if(NOT TARGET "${_TARGET}")
        rocprofiler_register_message(
            FATAL_ERROR
            "rocprofiler_register_strip_target not provided valid target: \"${_TARGET}\"")
    endif()

    if(CMAKE_STRIP AND (STRIP_FORCE OR ROCPROFILER_REGISTER_STRIP_LIBRARIES))
        if(STRIP_EXPLICIT)
            add_custom_command(
                TARGET ${_TARGET}
                POST_BUILD
                COMMAND ${CMAKE_STRIP} ${STRIP_ARGS} $<TARGET_FILE:${_TARGET}>
                WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
                COMMENT "Stripping ${_TARGET}...")
        else()
            add_custom_command(
                TARGET ${_TARGET}
                POST_BUILD
                COMMAND
                    ${CMAKE_STRIP} -w --keep-symbol="rocprofiler_register_init"
                    --keep-symbol="rocprofiler_register_finalize"
                    --keep-symbol="rocprofiler_register_push_trace"
                    --keep-symbol="rocprofiler_register_pop_trace"
                    --keep-symbol="rocprofiler_register_push_region"
                    --keep-symbol="rocprofiler_register_pop_region"
                    --keep-symbol="rocprofiler_register_set_env"
                    --keep-symbol="rocprofiler_register_set_mpi"
                    --keep-symbol="rocprofiler_register_reset_preload"
                    --keep-symbol="rocprofiler_register_set_instrumented"
                    --keep-symbol="rocprofiler_register_user_*"
                    --keep-symbol="ompt_start_tool" --keep-symbol="kokkosp_*"
                    --keep-symbol="OnLoad" --keep-symbol="OnUnload"
                    --keep-symbol="OnLoadToolProp" --keep-symbol="OnUnloadTool"
                    --keep-symbol="__libc_start_main" ${STRIP_ARGS}
                    $<TARGET_FILE:${_TARGET}>
                WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
                COMMENT "Stripping ${_TARGET}...")
        endif()
    endif()
endfunction()

# ------------------------------------------------------------------------------#
# function add_rocprofiler_register_test_target()
#
# Creates a target which runs ctest but depends on all the tests being built.
#
function(ADD_ROCPROFILER_REGISTER_TEST_TARGET)
    if(NOT TARGET rocprofiler-register-test)
        add_custom_target(
            rocprofiler-register-test
            COMMAND ${CMAKE_COMMAND} --build ${PROJECT_BINARY_DIR} --target test
            WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
            COMMENT "Running tests...")
    endif()
endfunction()

# ----------------------------------------------------------------------------------------#
# macro rocprofiler_register_checkout_git_submodule()
#
# Run "git submodule update" if a file in a submodule does not exist
#
# ARGS: RECURSIVE (option) -- add "--recursive" flag RELATIVE_PATH (one value) --
# typically the relative path to submodule from PROJECT_SOURCE_DIR WORKING_DIRECTORY (one
# value) -- (default: PROJECT_SOURCE_DIR) TEST_FILE (one value) -- file to check for
# (default: CMakeLists.txt) ADDITIONAL_CMDS (many value) -- any addition commands to pass
#
function(ROCPROFILER_REGISTER_CHECKOUT_GIT_SUBMODULE)
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
            message(
                STATUS "function(rocprofiler_register_checkout_git_submodule) failed.")
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
            message(
                STATUS "function(rocprofiler_register_checkout_git_submodule) failed.")
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
function(ROCPROFILER_REGISTER_TEST_FIND_PACKAGE PACKAGE_NAME OUTPUT_VAR)
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
function(ROCPROFILER_REGISTER_ADD_INTERFACE_LIBRARY _TARGET _DESCRIPT)
    add_library(${_TARGET} INTERFACE)
    string(REPLACE "${PROJECT_NAME}-" "" _ALIAS_TARGET "${_TARGET}")
    add_library(${PROJECT_NAME}::${_ALIAS_TARGET} ALIAS ${_TARGET})
    add_library(${PROJECT_NAME}::${_TARGET} ALIAS ${_TARGET})
    set(_ARGS "${ARGN}")
    if(NOT "INTERNAL" IN_LIST _ARGS)
        install(
            TARGETS ${_TARGET}
            DESTINATION ${CMAKE_INSTALL_LIBDIR}
            COMPONENT core
            EXPORT ${PROJECT_NAME}-library-targets
            OPTIONAL)
    endif()
endfunction()

# -----------------------------------------------------------------------
# function add_feature(<NAME> <DOCSTRING>) Add a project feature, whose activation is
# specified by the existence of the variable <NAME>, to the list of enabled/disabled
# features, plus a docstring describing the feature
#
function(ROCPROFILER_REGISTER_ADD_FEATURE _var _description)
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
        if(ROCPROFILER_REGISTER_BUILD_DOCS)
            set_property(
                GLOBAL APPEND PROPERTY ${PROJECT_NAME}_CMAKE_OPTIONS_DOC
                                       "${_var}` | ${_description}${EXTRA_DESC} |")
        endif()
    elseif("DOC" IN_LIST ARGN AND ROCPROFILER_REGISTER_BUILD_DOCS)
        set_property(GLOBAL APPEND PROPERTY ${PROJECT_NAME}_CMAKE_OPTIONS_DOC
                                            "${_var}` | ${_description}${EXTRA_DESC} |")
    endif()
endfunction()

# ----------------------------------------------------------------------------------------#
# function add_option(<OPTION_NAME> <DOCSRING> <DEFAULT_SETTING> [NO_FEATURE]) Add an
# option and add as a feature if NO_FEATURE is not provided
#
function(ROCPROFILER_REGISTER_ADD_OPTION _NAME _MESSAGE _DEFAULT)
    option(${_NAME} "${_MESSAGE}" ${_DEFAULT})
    if("NO_FEATURE" IN_LIST ARGN)
        mark_as_advanced(${_NAME})
    else()
        rocprofiler_register_add_feature(${_NAME} "${_MESSAGE}")
        if(ROCPROFILER_REGISTER_BUILD_DOCS)
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
# function rocprofiler_register_add_cache_option(<OPTION_NAME> <DOCSRING> <TYPE>
# <DEFAULT_VALUE> [NO_FEATURE] [ADVANCED] [CMAKE_DEFINE])
#
function(ROCPROFILER_REGISTER_ADD_CACHE_OPTION _NAME _DEFAULT _TYPE _MESSAGE)
    set(_FORCE)
    if("FORCE" IN_LIST ARGN)
        set(_FORCE FORCE)
    endif()

    set(${_NAME}
        "${_DEFAULT}"
        CACHE ${_TYPE} "${_MESSAGE}" ${_FORCE})

    if("NO_FEATURE" IN_LIST ARGN)
        mark_as_advanced(${_NAME})
    else()
        rocprofiler_register_add_feature(${_NAME} "${_MESSAGE}")

        if(ROCPROFILER_REGISTER_BUILD_DOCS)
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
# function rocprofiler_register_report_feature_changes() :: print changes in features
#
function(ROCPROFILER_REGISTER_REPORT_FEATURE_CHANGES)
    get_property(_features GLOBAL PROPERTY ${PROJECT_NAME}_FEATURES)
    if(NOT "${_features}" STREQUAL "")
        list(REMOVE_DUPLICATES _features)
        list(SORT _features)
    endif()
    foreach(_feature ${_features})
        if("${ARGN}" STREQUAL "")
            rocprofiler_register_watch_for_change(${_feature})
        elseif("${_feature}" IN_LIST ARGN)
            rocprofiler_register_watch_for_change(${_feature})
        endif()
    endforeach()
endfunction()

# ----------------------------------------------------------------------------------------#
# function print_enabled_features() Print enabled  features plus their docstrings.
#
function(ROCPROFILER_REGISTER_PRINT_ENABLED_FEATURES)
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
                    rocprofiler_register_capitalize("${_feature_tmp}" _feature_tmp_c)
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
function(ROCPROFILER_REGISTER_PRINT_DISABLED_FEATURES)
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
function(ROCPROFILER_REGISTER_PRINT_FEATURES)
    rocprofiler_register_report_feature_changes()
    rocprofiler_register_print_enabled_features()
    rocprofiler_register_print_disabled_features()
endfunction()

# ----------------------------------------------------------------------------------------#
# function watch_for_change() Print the value of variable if it changed
#
function(ROCPROFILER_REGISTER_WATCH_FOR_CHANGE _var)
    list(LENGTH ARGN _NUM_EXTRA_ARGS)
    if(_NUM_EXTRA_ARGS EQUAL 1)
        set(_VAR ${ARGN})
    else()
        set(_VAR)
    endif()

    macro(update_var _VAL)
        if(_VAR)
            set(${_VAR}
                ${_VAL}
                PARENT_SCOPE)
        endif()
    endmacro()

    update_var(OFF)

    set(_rocprofiler_register_watch_var_name ROCPROFILER_REGISTER_WATCH_VALUE_${_var})
    if(DEFINED ${_rocprofiler_register_watch_var_name})
        if("${${_var}}" STREQUAL "${${_rocprofiler_register_watch_var_name}}")
            return()
        else()
            rocprofiler_register_message(
                STATUS
                "${_var} changed :: ${${_rocprofiler_register_watch_var_name}} --> ${${_var}}"
                )
            update_var(ON)
        endif()
    else()
        if(NOT "${${_var}}" STREQUAL "")
            rocprofiler_register_message(STATUS "${_var} :: ${${_var}}")
            update_var(ON)
        endif()
    endif()

    # store the value for the next run
    set(${_rocprofiler_register_watch_var_name}
        "${${_var}}"
        CACHE INTERNAL "Last value of ${_var}" FORCE)
endfunction()

function(ROCPROFILER_REGISTER_FIND_STATIC_LIBRARY)
    set(CMAKE_FIND_LIBRARY_SUFFIXES ${CMAKE_STATIC_LIBRARY_SUFFIX})
    find_library(${ARGN})
endfunction()

function(ROCPROFILER_REGISTER_FIND_SHARED_LIBRARY)
    set(CMAKE_FIND_LIBRARY_SUFFIXES ${CMAKE_SHARED_LIBRARY_SUFFIX})
    find_library(${ARGN})
endfunction()

cmake_policy(POP)
