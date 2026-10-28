# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

include(GoogleTest)
find_package(Python3 COMPONENTS Interpreter REQUIRED)

function(rocjitsu_cli_add_gtest target)
    if(COMMAND rj_add_gtest_tests)
        rj_add_gtest_tests(${target} TEST_PREFIX "cli." ${ARGN})
        if(RJ_INSTALL_TESTS)
            file(
                APPEND "${RJ_INSTALL_CTEST_FILE}"
                "include(\"${target}.cmake\")\n"
            )
        endif()
    else()
        gtest_discover_tests(${target} TEST_PREFIX "cli." ${ARGN})
    endif()
endfunction()

add_executable(rocjitsu_cli_test cli_test.cpp)
target_link_libraries(rocjitsu_cli_test PRIVATE rocjitsu_cli GTest::gtest_main)
rocjitsu_cli_configure_target(rocjitsu_cli_test)
rocjitsu_cli_add_gtest(rocjitsu_cli_test)

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(RJ_CLI_PLUGIN_FIXTURE_FILE
        "${CMAKE_SHARED_MODULE_PREFIX}rocjitsu_cli_plugin_fixture${CMAKE_SHARED_MODULE_SUFFIX}"
    )
    add_library(rocjitsu_cli_plugin_fixture MODULE plugin.cpp)
    target_link_libraries(rocjitsu_cli_plugin_fixture PRIVATE rocjitsu_cli)
    rocjitsu_cli_configure_target(rocjitsu_cli_plugin_fixture)
    # Match the simulator's archive-export policy. Exported GNU-unique symbols
    # can pin a module and mask bugs in the explicit lifetime guard.
    target_link_options(
        rocjitsu_cli_plugin_fixture
        PRIVATE "LINKER:--exclude-libs,ALL"
    )
    set_target_properties(
        rocjitsu_cli_plugin_fixture
        PROPERTIES CXX_VISIBILITY_PRESET hidden
    )
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        target_compile_options(
            rocjitsu_cli_plugin_fixture
            PRIVATE -fno-gnu-unique
        )
    endif()

    add_executable(rocjitsu_cli_plugin_test plugin_test.cpp)
    target_link_libraries(
        rocjitsu_cli_plugin_test
        PRIVATE rocjitsu_cli GTest::gtest_main ${CMAKE_DL_LIBS}
    )
    target_compile_definitions(
        rocjitsu_cli_plugin_test
        PRIVATE RJ_CLI_PLUGIN_PATH="bin/${RJ_CLI_PLUGIN_FIXTURE_FILE}"
    )
    rocjitsu_cli_configure_target(rocjitsu_cli_plugin_test)
    add_dependencies(rocjitsu_cli_plugin_test rocjitsu_cli_plugin_fixture)
    rocjitsu_cli_add_gtest(
        rocjitsu_cli_plugin_test
        PROPERTIES ENVIRONMENT "RJ_CLI_PLUGIN_PATH=$<TARGET_FILE:rocjitsu_cli_plugin_fixture>"
    )

    add_executable(rj_cli_probe probe.cpp)
    target_link_libraries(rj_cli_probe PRIVATE rocjitsu_cli ${CMAKE_DL_LIBS})
    rocjitsu_cli_configure_target(rj_cli_probe)
    add_dependencies(rj_cli_probe rocjitsu_cli_plugin_fixture)

    if(COMMAND rj_add_test)
        rj_add_test(
            NAME cli.probe
            COMMAND
                ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_probe.py
                $<TARGET_FILE:rj_cli_probe>
                $<TARGET_FILE:rocjitsu_cli_plugin_fixture>
            INSTALL_COMMAND
                ${Python3_EXECUTABLE} test_probe.py
                bin/rj_cli_probe
                bin/${RJ_CLI_PLUGIN_FIXTURE_FILE}
        )
    else()
        add_test(
            NAME cli.probe
            COMMAND
                ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_probe.py
                $<TARGET_FILE:rj_cli_probe>
                $<TARGET_FILE:rocjitsu_cli_plugin_fixture>
        )
    endif()

    if(RJ_INSTALL_TESTS)
        rj_install_test_target(rj_cli_probe)
        install(
            TARGETS rocjitsu_cli_plugin_fixture
            LIBRARY DESTINATION ${RJ_INSTALL_TEST_BIN_DIR}
        )
        install(FILES test_probe.py DESTINATION ${RJ_INSTALL_TEST_DIR})
    endif()
endif()
