## Copyright (c) Advanced Micro Devices, Inc.
## SPDX-License-Identifier:  MIT
set(target test-compression)

add_executable(${target} main.cpp test_gzip_output_stream.h test_gzip_output_stream.cpp)

# ZLIB for independent inflate in tests, not via the writer.
target_link_libraries(${target} PRIVATE compression gtest_main ZLIB::ZLIB)
target_include_directories(${target} PUBLIC .)

if(INSTALL_TESTS)
    install(
        TARGETS ${target}
        RUNTIME
            DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/${CMAKE_PROJECT_NAME}/tests
            COMPONENT tests
    )
endif()
