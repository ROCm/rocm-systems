## Copyright (c) Advanced Micro Devices, Inc.
## SPDX-License-Identifier:  MIT
set(target test-csv)

add_executable(${target} main.cpp test_csv.h test_csv.cpp)

target_link_libraries(${target} PRIVATE csv gtest_main)
target_include_directories(${target} PUBLIC .)

if(INSTALL_TESTS)
    install(
        TARGETS ${target}
        RUNTIME
            DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/${CMAKE_PROJECT_NAME}/tests
            COMPONENT tests
    )
endif()
