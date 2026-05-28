## Copyright (c) Advanced Micro Devices, Inc.
## SPDX-License-Identifier:  MIT

set(target test-roctx-recordfn-leaf-context)

add_executable(${target} test_leaf_context.cpp)
target_link_libraries(${target} PRIVATE gtest_main)
target_include_directories(${target} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)

add_test(NAME ${target} COMMAND ${target})
set_tests_properties(${target} PROPERTIES LABELS "profile")

if(INSTALL_TESTS)
    install(
        TARGETS ${target}
        RUNTIME
            DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/${CMAKE_PROJECT_NAME}/tests
            COMPONENT tests
    )
endif()
