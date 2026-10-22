# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

add_executable(rocjitsu_cli_test cli_test.cpp)
target_link_libraries(rocjitsu_cli_test PRIVATE util_cli GTest::gtest_main)
include(GoogleTest)
gtest_discover_tests(rocjitsu_cli_test TEST_PREFIX "cli.")

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    add_library(rocjitsu_cli_plugin_fixture MODULE plugin.cpp)
    target_link_libraries(rocjitsu_cli_plugin_fixture PRIVATE util_cli)
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
        PRIVATE util_cli GTest::gtest_main ${CMAKE_DL_LIBS}
    )
    target_compile_definitions(
        rocjitsu_cli_plugin_test
        PRIVATE RJ_CLI_PLUGIN_PATH="$<TARGET_FILE:rocjitsu_cli_plugin_fixture>"
    )
    add_dependencies(rocjitsu_cli_plugin_test rocjitsu_cli_plugin_fixture)
    gtest_discover_tests(rocjitsu_cli_plugin_test TEST_PREFIX "cli.")
    add_executable(rj_cli_probe probe.cpp)
    target_link_libraries(rj_cli_probe PRIVATE util_cli ${CMAKE_DL_LIBS})
    add_dependencies(rj_cli_probe rocjitsu_cli_plugin_fixture)
    find_package(Python3 COMPONENTS Interpreter REQUIRED)
    add_test(
        NAME cli.probe
        COMMAND
            ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_probe.py
            $<TARGET_FILE:rj_cli_probe>
            $<TARGET_FILE:rocjitsu_cli_plugin_fixture>
    )
endif()
