#
# common utilities for tests
#

include(FetchContent)
include(CMakeParseArguments)

set(FETCHCONTENT_BASE_DIR ${PROJECT_BINARY_DIR}/external)

# default FAIL_REGULAR_EXPRESSION for tests
set(ROCPROFILER_DEFAULT_FAIL_REGEX
    "threw an exception|Permission denied|Could not create logging file|failed with error code|Subprocess aborted"
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

    function(rocprofiler_configure_pytest_files_check_pytest_ini _FILENAME)
        if(_FILENAME MATCHES ".*pytest\\.ini$")
            file(READ ${_FILENAME} _pytest_ini_content)
            if(_pytest_ini_content MATCHES "addopts.* -[a-zA-Z]*v.*")
                message(
                    SEND_ERROR
                        "${_FILENAME} should not contain -v or -vv. This breaks test discovery in pytest_discover_tests"
                    )
            endif()
        endif()
    endfunction()

    if(NOT RCPF_OUTPUT_DIRECTORY)
        set(RCPF_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
    endif()

    foreach(FILENAME ${RCPF_COPY})
        rocprofiler_configure_pytest_files_check_pytest_ini(
            ${CMAKE_CURRENT_SOURCE_DIR}/${FILENAME})
        configure_file(${CMAKE_CURRENT_SOURCE_DIR}/${FILENAME}
                       ${RCPF_OUTPUT_DIRECTORY}/${FILENAME} COPYONLY)
    endforeach()

    foreach(FILENAME ${RCPF_CONFIG})
        rocprofiler_configure_pytest_files_check_pytest_ini(
            ${CMAKE_CURRENT_SOURCE_DIR}/${FILENAME})
        configure_file(${CMAKE_CURRENT_SOURCE_DIR}/${FILENAME}
                       ${RCPF_OUTPUT_DIRECTORY}/${FILENAME} @ONLY)
    endforeach()
endfunction()

function(rocprofiler_set_integration_test_name _VAR _BASE _CATEGORY)
    set(_NAME "${_BASE}")

    if("${_NAME}" MATCHES "^tests\\.integration\\.${_CATEGORY}.*")
        set(_NAME "${_NAME}")
    else()
        set(_NAME "tests.integration.${_CATEGORY}.${_NAME}")
    endif()

    set(${_VAR}
        "${_NAME}"
        PARENT_SCOPE)
endfunction()

function(rocprofiler_add_integration_execute_test NAME)
    set(_FLAG_OPTS
        "WILL_FAIL" # failure is expected in order for test to pass
        "UNSTABLE" # disable when ROCPROFILER_DISABLE_UNSTABLE_CTESTS=ON
        "DISABLED_CODECOV" # disable when ROCPROFILER_BUILD_CODECOV=ON
        )
    set(_SINGLE_OPTS "TARGET" "WORKING_DIRECTORY" "DEPENDS" "TIMEOUT" "LABELS" "DISABLED"
                     "DISABLED_MEMCHECKS")
    set(_MULTI_OPTS
        "ARGS"
        "COMMAND"
        "ATTACHED_FILES"
        "ATTACHED_FILES_ON_FAIL"
        "ENVIRONMENT"
        "PRELOAD"
        "FIXTURES_SETUP"
        "FIXTURES_REQUIRED"
        "FIXTURES_CLEANUP"
        "PASS_REGULAR_EXPRESSION"
        "FAIL_REGULAR_EXPRESSION"
        "SKIP_REGULAR_EXPRESSION")

    cmake_parse_arguments(
        arg
        "${_FLAG_OPTS}" # <options>
        "${_SINGLE_OPTS}" # <one_value_keywords>
        "${_MULTI_OPTS}" # <multi_value_keywords>
        ${ARGN})

    rocprofiler_set_integration_test_name(arg_NAME "${NAME}" "execute")

    if("${arg_ENVIRONMENT}" MATCHES "LD_PRELOAD=.*")
        message(
            FATAL_ERROR
                "rocprofiler_add_integration_execute_test: LD_PRELOAD is not allowed in ENVIRONMENT for ${NAME} (${arg_NAME}). Use PRELOAD instead."
            )
    endif()

    macro(set_arg_if_empty _VAR)
        if(NOT ${_VAR})
            set(${_VAR} ${ARGN})
        endif()
    endmacro()

    # provide default
    set_arg_if_empty(arg_FAIL_REGULAR_EXPRESSION "${ROCPROFILER_DEFAULT_FAIL_REGEX}")
    set_arg_if_empty(arg_TIMEOUT 60)
    set_arg_if_empty(arg_FIXTURES_SETUP ${NAME})

    # the test is marked as unstable
    if(ROCPROFILER_DISABLE_UNSTABLE_CTESTS AND arg_UNSTABLE)
        set(arg_DISABLED ON)
    endif()

    # the test shouldn't be run for matching memchecks
    if(arg_DISABLED_MEMCHECKS AND ROCPROFILER_MEMCHECK MATCHES
                                  "${arg_DISABLED_MEMCHECKS}")
        set(arg_DISABLED ON)
    endif()

    # the tests shouldn't be run when rocprofiler-sdk is built with code-coverage support
    if(ROCPROFILER_BUILD_CODECOV AND arg_DISABLED_CODECOV)
        set(arg_DISABLED ON)
    endif()

    # insert sanitizer libraries into LD_PRELOAD
    if(ROCPROFILER_MEMCHECK_PRELOAD_ENV_VALUE)
        list(INSERT arg_PRELOAD 0 "${ROCPROFILER_MEMCHECK_PRELOAD_ENV_VALUE}")
    endif()

    if(arg_PRELOAD)
        list(JOIN arg_PRELOAD ":" _preload)
        set(arg_ENVIRONMENT "LD_PRELOAD=${_preload}" ${arg_ENVIRONMENT})
    endif()

    # provide LD_LIBRARY_PATH by default if not specified
    if(NOT "${arg_ENVIRONMENT}" MATCHES "LD_LIBRARY_PATH=.*")
        list(
            APPEND
            arg_ENVIRONMENT
            "LD_LIBRARY_PATH=$<TARGET_FILE_DIR:rocprofiler-sdk::rocprofiler-sdk-shared-library>:$ENV{LD_LIBRARY_PATH}"
            )
    endif()

    if(NOT "integration-tests" IN_LIST arg_LABELS)
        list(APPEND arg_LABELS "integration-tests")
    endif()

    if(NOT "execution" IN_LIST arg_LABELS)
        list(APPEND arg_LABELS "execution")
    endif()

    # Helper to set a property only if the arg is set
    function(_rocprofiler_set_test_property_if_arg TEST_NAME PROPERTY_NAME)
        if(arg_${PROPERTY_NAME})
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

        add_test(NAME "${arg_NAME}" COMMAND ${arg_COMMAND})

        set_tests_properties(${arg_NAME} PROPERTIES RUN_SERIAL ON)
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
        _rocprofiler_set_test_property_if_arg(${arg_NAME} FIXTURES_SETUP)
        _rocprofiler_set_test_property_if_arg(${arg_NAME} FIXTURES_REQUIRED)
        _rocprofiler_set_test_property_if_arg(${arg_NAME} FIXTURES_CLEANUP)
        _rocprofiler_set_test_property_if_arg(${arg_NAME} WILL_FAIL)

    elseif(arg_TARGET AND TARGET ${arg_TARGET})

        add_test(NAME "${arg_NAME}" COMMAND $<TARGET_FILE:${arg_TARGET}> ${arg_ARGS})

        set_tests_properties(${arg_NAME} PROPERTIES RUN_SERIAL ON)
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
        _rocprofiler_set_test_property_if_arg(${arg_NAME} FIXTURES_SETUP)
        _rocprofiler_set_test_property_if_arg(${arg_NAME} FIXTURES_REQUIRED)
        _rocprofiler_set_test_property_if_arg(${arg_NAME} FIXTURES_CLEANUP)
        _rocprofiler_set_test_property_if_arg(${arg_NAME} WILL_FAIL)

    else()

        message(WARNING "Target '${arg_TARGET}' not found")

        add_test(
            NAME "${arg_NAME}"
            COMMAND
                ${CMAKE_COMMAND} -E echo
                "'${arg_TARGET}' with args '${arg_ARGS}' could not be executed because the target does not exist"
            )

        # If this target is accidentally run, report it as a failed test
        set_tests_properties("${arg_NAME}" PROPERTIES FAIL_REGULAR_EXPRESSION ".*"
                                                      DISABLED ON)
    endif()
endfunction()

function(rocprofiler_add_integration_validate_test NAME)
    set(_FLAG_OPTS
        "WILL_FAIL" # failure is expected in order for test to pass
        "UNSTABLE" # disable when ROCPROFILER_DISABLE_UNSTABLE_CTESTS=ON
        "DISABLED_CODECOV" # disable when ROCPROFILER_BUILD_CODECOV=ON
        "STRIP_PARAM_BRACKETS"
        "INCLUDE_FILE_PATH"
        "BUNDLE_TESTS")
    set(_SINGLE_OPTS "TARGET_DEPENDS" "WORKING_DIRECTORY" "DEPENDS" "TIMEOUT" "LABELS"
                     "DISABLED" "DISABLED_MEMCHECKS")
    set(_MULTI_OPTS
        "TEST_PATHS"
        "CONFIG"
        "COPY"
        "ARGS"
        "DISCOVERY_ARGS"
        "ATTACHED_FILES"
        "ATTACHED_FILES_ON_FAIL"
        "PYTHON_PATH"
        "ENVIRONMENT"
        "PRELOAD"
        "FIXTURES_SETUP"
        "FIXTURES_REQUIRED"
        "FIXTURES_CLEANUP"
        "PASS_REGULAR_EXPRESSION"
        "FAIL_REGULAR_EXPRESSION"
        "SKIP_REGULAR_EXPRESSION")

    cmake_parse_arguments(
        arg
        "${_FLAG_OPTS}" # <options>
        "${_SINGLE_OPTS}" # <one_value_keywords>
        "${_MULTI_OPTS}" # <multi_value_keywords>
        ${ARGN})

    find_package(rocprofiler_sdk_pytest REQUIRED)

    rocprofiler_set_integration_test_name(arg_NAME "${NAME}" "validate")

    if("${arg_ENVIRONMENT}" MATCHES "LD_PRELOAD=.*")
        message(
            FATAL_ERROR
                "rocprofiler_add_integration_validate_test: LD_PRELOAD is not allowed in ENVIRONMENT for ${NAME} (${arg_NAME}). Use PRELOAD instead."
            )
    endif()

    macro(set_arg_if_empty _VAR)
        if(NOT ${_VAR})
            set(${_VAR} ${ARGN})
        endif()
    endmacro()

    # provide default
    set_arg_if_empty(arg_FAIL_REGULAR_EXPRESSION "${ROCPROFILER_DEFAULT_FAIL_REGEX}")
    set_arg_if_empty(arg_WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
    set_arg_if_empty(arg_TIMEOUT 60)
    set_arg_if_empty(arg_FIXTURES_REQUIRED ${NAME})
    set_arg_if_empty(arg_DISABLED OFF)

    # the test is marked as unstable
    if(ROCPROFILER_DISABLE_UNSTABLE_CTESTS AND arg_UNSTABLE)
        set(arg_DISABLED ON)
    endif()

    # the test shouldn't be run for matching memchecks
    if(arg_DISABLED_MEMCHECKS AND ROCPROFILER_MEMCHECK MATCHES
                                  "${arg_DISABLED_MEMCHECKS}")
        set(arg_DISABLED ON)
    endif()

    # the test shouldn't be run if any of the TARGET_DEPENDS are not built
    foreach(_TARGET ${arg_TARGET_DEPENDS})
        if(NOT TARGET ${_TARGET})
            set(arg_DISABLED ON)
        endif()
    endforeach()

    # the tests shouldn't be run when rocprofiler-sdk is built with code-coverage support
    if(ROCPROFILER_BUILD_CODECOV AND arg_DISABLED_CODECOV)
        set(arg_DISABLED ON)
    endif()

    if(NOT "integration-tests" IN_LIST arg_LABELS)
        list(APPEND arg_LABELS "integration-tests")
    endif()

    if(NOT "validation" IN_LIST arg_LABELS)
        list(APPEND arg_LABELS "validation")
    endif()

    set(arg_PYTHON_PATH "${ROCPROFILER_SDK_TESTS_BINARY_DIR}/pytest-packages"
                        ${arg_PYTHON_PATH})

    set(arg_CONFIG ${arg_CONFIG} ${arg_TEST_PATHS} pytest.ini)
    list(REMOVE_DUPLICATES arg_CONFIG)

    rocprofiler_configure_pytest_files(COPY ${arg_COPY} CONFIG ${arg_CONFIG})

    foreach(_FLAG STRIP_PARAM_BRACKETS INCLUDE_FILE_PATH BUNDLE_TESTS)
        if(arg_${_FLAG})
            set(_${_FLAG} ${_FLAG})
        else()
            set(_${_FLAG})
        endif()

    endforeach()

    # Build properties list safely to avoid argument shifting
    set(_props)
    list(APPEND _props DISABLED "${arg_DISABLED}")
    list(APPEND _props TIMEOUT "${arg_TIMEOUT}")

    function(_append_prop _key _listvar)
        if(DEFINED ${_listvar})
            list(JOIN ${_listvar} ";" _joined)
            string(REPLACE ";" "\\;" _escaped "${_joined}")
            list(APPEND _props ${_key} "${_escaped}")
            set(_props
                "${_props}"
                PARENT_SCOPE)
        endif()
    endfunction()

    # Multi-valued properties (must be joined!)
    _append_prop(LABELS arg_LABELS)
    _append_prop(FAIL_REGULAR_EXPRESSION arg_FAIL_REGULAR_EXPRESSION)
    _append_prop(SKIP_REGULAR_EXPRESSION arg_SKIP_REGULAR_EXPRESSION)
    _append_prop(PASS_REGULAR_EXPRESSION arg_PASS_REGULAR_EXPRESSION)
    _append_prop(FIXTURES_REQUIRED arg_FIXTURES_REQUIRED)

    rocprofiler_sdk_pytest_discover_tests(
        ${arg_NAME}
        ${_STRIP_PARAM_BRACKETS}
        ${_INCLUDE_FILE_PATH}
        ${_BUNDLE_TESTS}
        TEST_PATHS
        ${arg_TEST_PATHS}
        WORKING_DIRECTORY
        ${arg_WORKING_DIRECTORY}
        TRIM_FROM_NAME
        "${arg_TRIM_FROM_NAME}"
        TRIM_FROM_FULL_NAME
        "${arg_TRIM_FROM_FULL_NAME}"
        LIBRARY_PATH_PREPEND
        ${arg_LIBRARY_PATH}
        PYTHON_PATH_PREPEND
        ${arg_PYTHON_PATH}
        ENVIRONMENT
        "${arg_ENVIRONMENT}"
        DEPENDS
        "${arg_DEPENDS}"
        EXTRA_ARGS
        ${arg_ARGS}
        --durations=20
        -rA
        -s
        -vv
        DISCOVERY_EXTRA_ARGS
        ${arg_DISCOVERY_ARGS}
        PROPERTIES
        ${_props})

endfunction()
