# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Shared build-tree and installed-test registration, initialized before test subdirectories.
include_guard(GLOBAL)

if(RJ_INSTALL_TESTS)
    set(RJ_INSTALL_TEST_DIR "${CMAKE_INSTALL_DATADIR}/rocjitsu/tests")
    set(RJ_INSTALL_TEST_BIN_DIR "${RJ_INSTALL_TEST_DIR}/bin")
    set(RJ_INSTALL_CTEST_FILE
        "${PROJECT_BINARY_DIR}/tests/install/CTestTestfile.cmake"
    )
    set(RJ_INSTALL_GTEST_GENERATOR
        "${PROJECT_SOURCE_DIR}/cmake/rj_generate_installed_gtest_ctest.cmake"
    )
    if(IS_ABSOLUTE "${CMAKE_INSTALL_BINDIR}")
        set(_rj_installed_bindir "${CMAKE_INSTALL_BINDIR}")
    else()
        set(_rj_installed_bindir "../../../${CMAKE_INSTALL_BINDIR}")
    endif()
    if(IS_ABSOLUTE "${CMAKE_INSTALL_DATADIR}")
        set(_rj_installed_datadir "${CMAKE_INSTALL_DATADIR}")
    else()
        set(_rj_installed_datadir "../..")
    endif()
    if(IS_ABSOLUTE "${CMAKE_INSTALL_LIBDIR}")
        set(_rj_installed_libdir "${CMAKE_INSTALL_LIBDIR}")
    else()
        set(_rj_installed_libdir "../../../${CMAKE_INSTALL_LIBDIR}")
    endif()
    file(MAKE_DIRECTORY "${PROJECT_BINARY_DIR}/tests/install")
    file(WRITE "${RJ_INSTALL_CTEST_FILE}" "# Installed rocjitsu CTest tests\n")
    file(
        APPEND "${RJ_INSTALL_CTEST_FILE}"
        "set(RJ_INSTALLED_BINDIR \"${_rj_installed_bindir}\")\n"
        "set(RJ_INSTALLED_CONFIG_DIR \"${_rj_installed_datadir}/rocjitsu/configs\")\n\n"
        "set(RJ_INSTALLED_KERNEL_DIR \"kernels\")\n\n"
        "set(RJ_INSTALLED_LIBDIR \"${_rj_installed_libdir}\")\n\n"
    )

    # Install a compiled test/tool target into the installed CTest bin directory.
    # The installed CTest file references these binaries with paths relative to
    # ${RJ_INSTALL_TEST_DIR}, e.g. bin/rocjitsu_tests.
    function(rj_install_test_target target)
        install(
            TARGETS ${target}
            RUNTIME DESTINATION ${RJ_INSTALL_TEST_BIN_DIR}
        )
    endfunction()

    # Install a non-CMake executable produced by a custom command, such as a HIP
    # test binary compiled directly with hipcc.
    function(rj_install_test_program path)
        install(PROGRAMS ${path} DESTINATION ${RJ_INSTALL_TEST_BIN_DIR})
    endfunction()

    # Append one installed CTest entry to ${RJ_INSTALL_CTEST_FILE}. This mirrors
    # the subset of add_test()/set_tests_properties() used by rj_add_test, but
    # writes plain CMake code because the file is consumed later by
    # `ctest --test-dir <installed tests dir>`.
    function(rj_install_ctest name)
        set(oneValueArgs TIMEOUT SKIP_REGULAR_EXPRESSION)
        set(multiValueArgs COMMAND ENVIRONMENT LABELS)
        cmake_parse_arguments(
            ARG
            "RUN_SERIAL"
            "${oneValueArgs}"
            "${multiValueArgs}"
            ${ARGN}
        )
        if(NOT ARG_COMMAND)
            message(FATAL_ERROR "rj_install_ctest(${name}) requires COMMAND")
        endif()

        set(_content "add_test(\"${name}\"")
        foreach(_arg IN LISTS ARG_COMMAND)
            string(APPEND _content " \"${_arg}\"")
        endforeach()
        string(APPEND _content ")\n")

        set(_props)
        if(ARG_TIMEOUT)
            string(APPEND _props " \"TIMEOUT\" \"${ARG_TIMEOUT}\"")
        endif()
        if(ARG_SKIP_REGULAR_EXPRESSION)
            string(
                APPEND _props
                " \"SKIP_REGULAR_EXPRESSION\" [=[${ARG_SKIP_REGULAR_EXPRESSION}]=]"
            )
        endif()
        if(ARG_LABELS)
            list(JOIN ARG_LABELS ";" _labels)
            string(APPEND _props " \"LABELS\" \"${_labels}\"")
        endif()
        if(ARG_RUN_SERIAL)
            string(APPEND _props " \"RUN_SERIAL\" \"TRUE\"")
        endif()
        if(ARG_ENVIRONMENT)
            list(JOIN ARG_ENVIRONMENT ";" _env)
            string(APPEND _props " \"ENVIRONMENT\" \"${_env}\"")
        endif()
        if(_props)
            string(
                APPEND _content
                "set_tests_properties(\"${name}\" PROPERTIES"
            )
            string(APPEND _content "${_props})\n")
        endif()

        file(APPEND "${RJ_INSTALL_CTEST_FILE}" "${_content}\n")
    endfunction()

    # Generate and install fine-grained CTest entries for every concrete
    # GoogleTest case in a target. The target executable must already exist, so
    # this is a build-time custom command that runs --gtest_list_tests and writes
    # an installed CTest fragment like rocjitsu_tests.cmake.
    function(rj_install_gtest_cases target)
        set(_fragment "${PROJECT_BINARY_DIR}/tests/install/${target}.cmake")
        cmake_parse_arguments(ARG "" "TEST_FILTER;TEST_PREFIX" "" ${ARGN})
        add_custom_command(
            OUTPUT "${_fragment}"
            COMMAND
                ${CMAKE_COMMAND} "-DTEST_EXECUTABLE=$<TARGET_FILE:${target}>"
                "-DINSTALLED_EXECUTABLE=bin/${target}"
                "-DTEST_FILTER=${ARG_TEST_FILTER}"
                "-DTEST_PREFIX=${ARG_TEST_PREFIX}" "-DOUTPUT_FILE=${_fragment}"
                -P "${RJ_INSTALL_GTEST_GENERATOR}"
            DEPENDS ${target} "${RJ_INSTALL_GTEST_GENERATOR}"
            COMMENT "Generating installed CTest entries for ${target}"
            VERBATIM
        )
        add_custom_target(${target}_installed_ctest ALL DEPENDS "${_fragment}")
        install(FILES "${_fragment}" DESTINATION ${RJ_INSTALL_TEST_DIR})
    endfunction()

    install(FILES "${RJ_INSTALL_CTEST_FILE}" DESTINATION ${RJ_INSTALL_TEST_DIR})
    install(
        FILES
            "${PROJECT_SOURCE_DIR}/cmake/rj_run_filtered_gtest.cmake"
            "${PROJECT_SOURCE_DIR}/cmake/rj_run_gtest_shard.cmake"
        DESTINATION ${RJ_INSTALL_TEST_DIR}
    )
    install(
        FILES "${PROJECT_SOURCE_DIR}/tests/tsan_hip_runtime.supp"
        DESTINATION ${RJ_INSTALL_TEST_DIR}
    )
endif()

include(GoogleTest)

set(_rj_installed_test_bin_dir "${CMAKE_INSTALL_DATADIR}/rocjitsu/tests/bin")
function(_rj_path_from_installed_test_bin out install_dir file_name)
    if(IS_ABSOLUTE "${install_dir}")
        set(_path "${install_dir}/${file_name}")
    elseif(IS_ABSOLUTE "${_rj_installed_test_bin_dir}")
        set(_path "${CMAKE_INSTALL_PREFIX}/${install_dir}/${file_name}")
    else()
        file(
            RELATIVE_PATH
            _rel_dir
            "/${_rj_installed_test_bin_dir}"
            "/${install_dir}"
        )
        set(_path "${_rel_dir}/${file_name}")
    endif()
    set(${out} "${_path}" PARENT_SCOPE)
endfunction()

function(rj_test_runtime_dir_env out name)
    string(REGEX REPLACE "[^A-Za-z0-9_.-]" "_" _runtime_name "${name}")
    set(${out}
        "ROCJITSU_RUNTIME_DIR=${CMAKE_CURRENT_BINARY_DIR}/runtime/${_runtime_name}"
        PARENT_SCOPE
    )
endfunction()

function(rj_installed_test_runtime_dir_env out name)
    string(REGEX REPLACE "[^A-Za-z0-9_.-]" "_" _runtime_name "${name}")
    set(${out} "ROCJITSU_RUNTIME_DIR=runtime/${_runtime_name}" PARENT_SCOPE)
endfunction()

function(rj_environment_has_runtime_dir out)
    set(_has_runtime_dir FALSE)
    foreach(_entry IN LISTS ARGN)
        if(_entry MATCHES "^ROCJITSU_RUNTIME_DIR=")
            set(_has_runtime_dir TRUE)
        endif()
    endforeach()
    set(${out} "${_has_runtime_dir}" PARENT_SCOPE)
endfunction()

function(rj_set_default_runtime_dir name)
    get_property(_env TEST "${name}" PROPERTY ENVIRONMENT)
    rj_environment_has_runtime_dir(_has_runtime_dir ${_env})
    if(NOT _has_runtime_dir)
        rj_test_runtime_dir_env(_runtime_dir_env "${name}")
        list(APPEND _env "${_runtime_dir_env}")
        set_tests_properties("${name}" PROPERTIES ENVIRONMENT "${_env}")
    endif()
endfunction()

# Register a test for both the build tree and, when RJ_INSTALL_TESTS is ON, the
# installed CTest tree. COMMAND is the normal build-tree command passed to
# add_test(). INSTALL_COMMAND is optional and should use paths relative to the
# installed test directory, e.g. bin/my_test or ${RJ_INSTALLED_BINDIR}/rocjitsu.
# Shared properties TIMEOUT, SKIP_REGULAR_EXPRESSION, RUN_SERIAL, and LABELS are
# applied to both registrations. ENVIRONMENT applies only to the build-tree test;
# INSTALL_ENVIRONMENT adds to the default installed test environment, which
# points installed tests at packaged configs, kernels, binaries, and runtimes.
function(rj_add_test)
    set(oneValueArgs NAME WORKING_DIRECTORY TIMEOUT SKIP_REGULAR_EXPRESSION)
    set(multiValueArgs
        COMMAND
        ENVIRONMENT
        INSTALL_COMMAND
        INSTALL_ENVIRONMENT
        LABELS
    )
    cmake_parse_arguments(
        ARG
        "RUN_SERIAL"
        "${oneValueArgs}"
        "${multiValueArgs}"
        ${ARGN}
    )
    if(NOT ARG_NAME)
        message(FATAL_ERROR "rj_add_test requires NAME")
    endif()
    if(NOT ARG_COMMAND)
        message(FATAL_ERROR "rj_add_test(${ARG_NAME}) requires COMMAND")
    endif()

    rj_environment_has_runtime_dir(_has_runtime_dir ${ARG_ENVIRONMENT})
    if(NOT _has_runtime_dir)
        # Each launcher test writes the active config into its runtime dir for
        # the preloaded interposer. Use a per-test default so ctest -j runs
        # cannot overwrite another test's config or daemon socket.
        rj_test_runtime_dir_env(_runtime_dir_env "${ARG_NAME}")
        list(APPEND ARG_ENVIRONMENT "${_runtime_dir_env}")
    endif()

    add_test(NAME "${ARG_NAME}" COMMAND ${ARG_COMMAND})

    if(ARG_WORKING_DIRECTORY)
        set_tests_properties(
            "${ARG_NAME}"
            PROPERTIES WORKING_DIRECTORY "${ARG_WORKING_DIRECTORY}"
        )
    endif()
    if(ARG_TIMEOUT)
        set_tests_properties("${ARG_NAME}" PROPERTIES TIMEOUT "${ARG_TIMEOUT}")
    endif()
    if(ARG_SKIP_REGULAR_EXPRESSION)
        set_tests_properties(
            "${ARG_NAME}"
            PROPERTIES SKIP_REGULAR_EXPRESSION "${ARG_SKIP_REGULAR_EXPRESSION}"
        )
    endif()
    if(ARG_LABELS)
        list(JOIN ARG_LABELS ";" _labels)
        set_tests_properties("${ARG_NAME}" PROPERTIES LABELS "${_labels}")
    endif()
    if(ARG_RUN_SERIAL)
        set_tests_properties("${ARG_NAME}" PROPERTIES RUN_SERIAL TRUE)
    endif()
    if(ARG_ENVIRONMENT)
        list(JOIN ARG_ENVIRONMENT ";" _env)
        set_tests_properties("${ARG_NAME}" PROPERTIES ENVIRONMENT "${_env}")
    endif()

    if(RJ_INSTALL_TESTS AND ARG_INSTALL_COMMAND)
        set(_install_args COMMAND ${ARG_INSTALL_COMMAND})
        if(ARG_TIMEOUT)
            list(APPEND _install_args TIMEOUT "${ARG_TIMEOUT}")
        endif()
        if(ARG_SKIP_REGULAR_EXPRESSION)
            list(
                APPEND _install_args
                SKIP_REGULAR_EXPRESSION
                "${ARG_SKIP_REGULAR_EXPRESSION}"
            )
        endif()
        if(ARG_LABELS)
            list(APPEND _install_args LABELS ${ARG_LABELS})
        endif()
        if(ARG_RUN_SERIAL)
            list(APPEND _install_args RUN_SERIAL)
        endif()
        set(_install_env
            "ROCJITSU_CONFIG_DIR=\${RJ_INSTALLED_CONFIG_DIR}"
            "ROCJITSU_KERNEL_DIR=\${RJ_INSTALLED_KERNEL_DIR}"
        )
        rj_environment_has_runtime_dir(
            _install_has_runtime_dir
            ${ARG_INSTALL_ENVIRONMENT}
        )
        if(NOT _install_has_runtime_dir)
            rj_installed_test_runtime_dir_env(
                _install_runtime_dir_env
                "${ARG_NAME}"
            )
            list(APPEND _install_env "${_install_runtime_dir_env}")
        endif()
        list(APPEND _install_env ${ARG_INSTALL_ENVIRONMENT})
        list(APPEND _install_args ENVIRONMENT ${_install_env})
        rj_install_ctest("${ARG_NAME}" ${_install_args})
    endif()
endfunction()

# Register a GoogleTest target with fine-grained tests in both the build tree
# and installed CTest tree. Build-tree discovery delegates to
# gtest_discover_tests(); installed discovery delegates to rj_install_gtest_cases
# so parameterized TEST_P cases are expanded from the built executable.
#
# TEST_FILTER reaches BOTH paths. It used to reach only the build tree, while the
# installed generator carried its own hard-coded copy, so a case excluded here to be
# registered manually was still discovered on the installed side and ended up
# registered twice under one name.
function(rj_add_gtest_tests target)
    cmake_parse_arguments(ARG "" "TEST_FILTER;TEST_PREFIX" "" ${ARGN})
    gtest_discover_tests(${target} ${ARGN})
    if(RJ_INSTALL_TESTS)
        rj_install_test_target(${target})
        rj_install_gtest_cases(
            ${target}
            TEST_FILTER "${ARG_TEST_FILTER}"
            TEST_PREFIX "${ARG_TEST_PREFIX}"
        )
    endif()
endfunction()
