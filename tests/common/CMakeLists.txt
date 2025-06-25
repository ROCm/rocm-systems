#
# common utilities for tests
#

include(FetchContent)
include(CMakeParseArguments)

set(FETCHCONTENT_BASE_DIR ${PROJECT_BINARY_DIR}/external)

# default FAIL_REGULAR_EXPRESSION for tests
set(ROCPROFILER_DEFAULT_FAIL_REGEX
    "threw an exception|terminate called after throwing|Permission denied|Could not create logging file|failed with error code|Subprocess aborted"
    CACHE INTERNAL "Default FAIL_REGULAR_EXPRESSION for tests")

set(DEFAULT_GPU_TARGETS
    "gfx900"
    "gfx906"
    "gfx908"
    "gfx90a"
    "gfx942"
    "gfx950"
    "gfx1030"
    "gfx1010"
    "gfx1100"
    "gfx1101"
    "gfx1102")

set(GPU_TARGETS
    "${DEFAULT_GPU_TARGETS}"
    CACHE STRING "GPU targets to compile for")

set(AMDGPU_TARGETS
    "${GPU_TARGETS}"
    CACHE STRING
          "GPU targets to compile for AMDGPUs (update GPU_TARGETS, not this variable)"
          FORCE)

if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.24)
    cmake_policy(SET CMP0135 NEW)
endif()

if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.30)
    cmake_policy(SET CMP0167 NEW)
    cmake_policy(SET CMP0169 OLD)
endif()

find_package(rocprofiler-sdk REQUIRED)

# rocprofiler-sdk provides a Findlibdw.cmake
find_package(libdw REQUIRED)

# build flags
add_library(rocprofiler-sdk-tests-build-flags INTERFACE)
add_library(rocprofiler-sdk::tests-build-flags ALIAS rocprofiler-sdk-tests-build-flags)
target_compile_options(rocprofiler-sdk-tests-build-flags INTERFACE -W -Wall -Wextra
                                                                   -Wshadow)
target_compile_features(rocprofiler-sdk-tests-build-flags INTERFACE cxx_std_17)

if(ROCPROFILER_BUILD_CI OR ROCPROFILER_BUILD_WERROR)
    target_compile_options(rocprofiler-sdk-tests-build-flags INTERFACE -Werror)
endif()

# serialization library
if(NOT TARGET rocprofiler-sdk::rocprofiler-sdk-cereal)
    get_filename_component(ROCPROFILER_SOURCE_DIR "${PROJECT_SOURCE_DIR}/.." REALPATH)

    add_library(rocprofiler-sdk-cereal INTERFACE)
    add_library(rocprofiler-sdk::rocprofiler-sdk-cereal ALIAS rocprofiler-sdk-cereal)
    target_compile_definitions(rocprofiler-sdk-cereal
                               INTERFACE $<BUILD_INTERFACE:CEREAL_THREAD_SAFE=1>)

    if(EXISTS ${ROCPROFILER_SOURCE_DIR}/external AND COMMAND
                                                     rocprofiler_checkout_git_submodule)
        rocprofiler_checkout_git_submodule(
            RECURSIVE
            RELATIVE_PATH external/cereal
            WORKING_DIRECTORY ${ROCPROFILER_SOURCE_DIR}
            REPO_URL https://github.com/jrmadsen/cereal.git
            REPO_BRANCH "rocprofiler")

        target_include_directories(
            rocprofiler-sdk-cereal SYSTEM
            INTERFACE $<BUILD_INTERFACE:${ROCPROFILER_SOURCE_DIR}/external/cereal/include>
            )
    else()
        fetchcontent_declare(
            cereal
            GIT_REPOSITORY https://github.com/jrmadsen/cereal.git
            GIT_TAG rocprofiler
            SOURCE_DIR ${PROJECT_BINARY_DIR}/external/cereal BINARY_DIR
            ${PROJECT_BINARY_DIR}/external/build/cereal-build SUBBUILD_DIR
            ${PROJECT_BINARY_DIR}/external/build/cereal-subdir)

        fetchcontent_getproperties(cereal)

        if(NOT cereal_POPULATED)
            fetchcontent_populate(cereal)
        endif()

        target_include_directories(
            rocprofiler-sdk-cereal SYSTEM
            INTERFACE $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/external/cereal/include>)
    endif()
endif()

if(NOT TARGET rocprofiler-sdk::rocprofiler-sdk-perfetto)
    # perfetto
    fetchcontent_declare(
        perfetto
        GIT_REPOSITORY https://github.com/google/perfetto
        GIT_TAG v44.0
        SOURCE_DIR ${PROJECT_BINARY_DIR}/external/perfetto BINARY_DIR
        ${PROJECT_BINARY_DIR}/external/build/perfetto-build SUBBUILD_DIR
        ${PROJECT_BINARY_DIR}/external/build/perfetto-subdir)

    fetchcontent_getproperties(perfetto)

    if(NOT perfetto_POPULATED)
        fetchcontent_populate(perfetto)
    endif()

    add_library(rocprofiler-sdk-tests-perfetto STATIC)
    add_library(rocprofiler-sdk::tests-perfetto ALIAS rocprofiler-sdk-tests-perfetto)
    target_sources(
        rocprofiler-sdk-tests-perfetto
        PRIVATE ${PROJECT_BINARY_DIR}/external/perfetto/sdk/perfetto.h
                ${PROJECT_BINARY_DIR}/external/perfetto/sdk/perfetto.cc)
    target_include_directories(
        rocprofiler-sdk-tests-perfetto SYSTEM
        INTERFACE $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/external/perfetto/sdk>)
    set_target_properties(rocprofiler-sdk-tests-perfetto
                          PROPERTIES POSITION_INDEPENDENT_CODE ON)
else()
    add_library(rocprofiler-sdk-tests-perfetto INTERFACE)
    add_library(rocprofiler-sdk::tests-perfetto ALIAS rocprofiler-sdk-tests-perfetto)
    target_link_libraries(rocprofiler-sdk-tests-perfetto
                          INTERFACE rocprofiler-sdk::rocprofiler-sdk-perfetto)
endif()

# common utilities
cmake_path(GET CMAKE_CURRENT_SOURCE_DIR PARENT_PATH COMMON_LIBRARY_INCLUDE_DIR)

add_library(rocprofiler-sdk-tests-common-library INTERFACE)
add_library(rocprofiler-sdk::tests-common-library ALIAS
            rocprofiler-sdk-tests-common-library)
target_link_libraries(
    rocprofiler-sdk-tests-common-library
    INTERFACE rocprofiler-sdk::tests-build-flags rocprofiler-sdk::rocprofiler-sdk-cereal
              libdw::libdw)
target_compile_features(rocprofiler-sdk-tests-common-library INTERFACE cxx_std_17)
target_include_directories(rocprofiler-sdk-tests-common-library
                           INTERFACE ${COMMON_LIBRARY_INCLUDE_DIR})

set(EXTERNAL_SUBMODULE_DIR "${PROJECT_SOURCE_DIR}/../external")
cmake_path(ABSOLUTE_PATH EXTERNAL_SUBMODULE_DIR NORMALIZE)

if(EXISTS ${EXTERNAL_SUBMODULE_DIR}/filesystem/include/ghc/filesystem.hpp)
    target_compile_definitions(
        rocprofiler-sdk-tests-common-library
        INTERFACE $<BUILD_INTERFACE:ROCPROFILER_SAMPLES_HAS_GHC_LIB_FILESYSTEM=1>)
    target_include_directories(
        rocprofiler-sdk-tests-common-library SYSTEM
        INTERFACE $<BUILD_INTERFACE:${EXTERNAL_SUBMODULE_DIR}/filesystem/include>)
endif()

function(rocprofiler_configure_pytest_files)
    cmake_parse_arguments(RCPF "" "OUTPUT_DIRECTORY" "COPY;CONFIG" ${ARGN})

    if(NOT RCPF_OUTPUT_DIRECTORY)
        set(RCPF_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
    endif()

    foreach(FILENAME ${RCPF_COPY})
        configure_file(${CMAKE_CURRENT_SOURCE_DIR}/${FILENAME}
                       ${RCPF_OUTPUT_DIRECTORY}/${FILENAME} COPYONLY)
    endforeach()

    foreach(FILENAME ${RCPF_CONFIG})
        configure_file(${CMAKE_CURRENT_SOURCE_DIR}/${FILENAME}
                       ${RCPF_OUTPUT_DIRECTORY}/${FILENAME} @ONLY)
    endforeach()
endfunction()

function(rocprofiler_add_test)
    cmake_parse_arguments(
        arg
        "WILL_FAIL" # <options>
        "NAME;TARGET;WORKING_DIRECTORY;DEPENDS;TIMEOUT;LABELS;DISABLED" # <one_value_keywords>
        "ARGS;COMMAND;ATTACHED_FILES;ATTACHED_FILES_ON_FAIL;ENVIRONMENT;PASS_REGULAR_EXPRESSION;FAIL_REGULAR_EXPRESSION;SKIP_REGULAR_EXPRESSION" # <multi_value_keywords>
        ${ARGN})

    # Helper to set a property only if the arg is set
    function(_rocprofiler_set_test_property_if_arg TEST_NAME PROPERTY_NAME)
        if(DEFINED "arg_${PROPERTY_NAME}")
            set_tests_properties(${TEST_NAME} PROPERTIES ${PROPERTY_NAME}
                                                         "${arg_${PROPERTY_NAME}}")
        endif()
    endfunction()

    if(DEFINED arg_COMMAND AND DEFINED arg_TARGET)
        message(SEND_ERROR "Cannot specify both TARGET and COMMAND arguments at once")
        return()
    elseif(NOT DEFINED arg_COMMAND AND NOT DEFINED arg_TARGET)
        message(SEND_ERROR "One of COMMAND or TARGET must be specified")
        return()
    endif()

    if(DEFINED arg_COMMAND AND NOT DEFINED arg_DEPENDS)
        message(SEND_ERROR "COMMAND signature without DEPENDS specified")

    elseif(
        DEFINED arg_COMMAND
        AND DEFINED arg_DEPENDS
        AND TARGET ${arg_DEPENDS})

        list(JOIN arg_COMMAND " " joined_CMD)

        add_test(NAME "${arg_NAME}" COMMAND ${arg_COMMAND})
        _rocprofiler_set_test_property_if_arg(${arg_NAME} TIMEOUT)
        _rocprofiler_set_test_property_if_arg(${arg_NAME} LABELS)
        _rocprofiler_set_test_property_if_arg(${arg_NAME} WORKING_DIRECTORY)
        _rocprofiler_set_test_property_if_arg(${arg_NAME} ENVIRONMENT)
        _rocprofiler_set_test_property_if_arg(${arg_NAME} PASS_REGULAR_EXPRESSION)
        _rocprofiler_set_test_property_if_arg(${arg_NAME} FAIL_REGULAR_EXPRESSION)
        _rocprofiler_set_test_property_if_arg(${arg_NAME} SKIP_REGULAR_EXPRESSION)
        _rocprofiler_set_test_property_if_arg(${arg_NAME} ATTACHED_FILES)
        _rocprofiler_set_test_property_if_arg(${arg_NAME} ATTACHED_FILES_ON_FAIL)
        _rocprofiler_set_test_property_if_arg(${arg_NAME} DEPENDS)
        _rocprofiler_set_test_property_if_arg(${arg_NAME} DISABLED)

    elseif(DEFINED arg_TARGET AND TARGET ${arg_TARGET})
        list(JOIN arg_ARGS " " joined_ARGS)

        add_test(NAME "${arg_NAME}" COMMAND $<TARGET_FILE:${arg_TARGET}> ${arg_ARGS})
        _rocprofiler_set_test_property_if_arg(${arg_NAME} TIMEOUT)
        _rocprofiler_set_test_property_if_arg(${arg_NAME} LABELS)
        _rocprofiler_set_test_property_if_arg(${arg_NAME} WORKING_DIRECTORY)
        _rocprofiler_set_test_property_if_arg(${arg_NAME} ENVIRONMENT)
        _rocprofiler_set_test_property_if_arg(${arg_NAME} PASS_REGULAR_EXPRESSION)
        _rocprofiler_set_test_property_if_arg(${arg_NAME} FAIL_REGULAR_EXPRESSION)
        _rocprofiler_set_test_property_if_arg(${arg_NAME} SKIP_REGULAR_EXPRESSION)
        _rocprofiler_set_test_property_if_arg(${arg_NAME} ATTACHED_FILES)
        _rocprofiler_set_test_property_if_arg(${arg_NAME} ATTACHED_FILES_ON_FAIL)
        _rocprofiler_set_test_property_if_arg(${arg_NAME} DEPENDS)
        _rocprofiler_set_test_property_if_arg(${arg_NAME} DISABLED)
    else()
        add_test(
            NAME "${arg_NAME}"
            COMMAND
                ${CMAKE_COMMAND} -E echo
                "'${arg_TARGET}' with args '${arg_ARGS}' could not be executed because the target does not exist"
            )

        # If this target is accidentally run, report it as a failed test
        set_tests_properties("${arg_NAME}" PROPERTIES FAIL_REGULAR_EXPRESSION ".*"
                                                      DISABLED ${arg_DISABLED})
    endif()
endfunction()
