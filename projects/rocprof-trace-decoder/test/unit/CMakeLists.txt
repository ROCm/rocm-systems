# ######################################################################################################################
# # Copyright (c) 2025 Advanced Micro Devices, Inc.
# ######################################################################################################################

cmake_minimum_required(VERSION 3.16.0)

project(att-decoder-unit-tests LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Fetch GoogleTest
include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.14.0)
set(INSTALL_GTEST
    OFF
    CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()

# Source directory for including internal headers
set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../source)
set(INC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../include)

# Test source files
set(TEST_FILES
    trie_test.cpp
    segment_test.cpp
    stitch_utils_test.cpp
    stitch_test.cpp
    trace_parser_test.cpp
    gfx9_test.cpp
    gfx10_test.cpp
    gfx11_test.cpp
    gfx12_test.cpp
    token_stream_test.cpp
    sqttparser_test.cpp)

# Function to create a test executable with specific sanitizer options Re-list library sources for sanitizer builds
# (must be recompiled with sanitizer flags)
file(
    GLOB
    UNIT_TEST_LIB_SOURCES
    ${SRC_DIR}/trie.cpp
    ${SRC_DIR}/sqttparser.cpp
    ${SRC_DIR}/stitch/stitch.cpp
    ${SRC_DIR}/stitch/pctranslator.cpp
    ${SRC_DIR}/trace_parser.cpp
    ${SRC_DIR}/gfx9/*.cpp
    ${SRC_DIR}/gfx10/*.cpp
    ${SRC_DIR}/gfx11/*.cpp
    ${SRC_DIR}/gfx12/*.cpp)

function(add_unit_test_executable TARGET_NAME SANITIZER_TYPE)
    if(SANITIZER_TYPE STREQUAL "regular")
        add_executable(${TARGET_NAME} ${TEST_FILES})
        target_link_libraries(${TARGET_NAME} PRIVATE rocprof-trace-decoder-static)
    else()
        add_executable(${TARGET_NAME} ${TEST_FILES} ${UNIT_TEST_LIB_SOURCES})
    endif()

    target_include_directories(${TARGET_NAME} PRIVATE ${SRC_DIR} ${INC_DIR})

    target_link_libraries(${TARGET_NAME} PRIVATE GTest::gtest_main GTest::gmock_main)

    if(SANITIZER_TYPE STREQUAL "asan")
        target_compile_options(${TARGET_NAME} PRIVATE -fsanitize=address -fno-omit-frame-pointer)
        target_link_options(${TARGET_NAME} PRIVATE -fsanitize=address)
    elseif(SANITIZER_TYPE STREQUAL "ubsan")
        target_compile_options(${TARGET_NAME} PRIVATE -fsanitize=undefined -fno-omit-frame-pointer)
        target_link_options(${TARGET_NAME} PRIVATE -fsanitize=undefined)
    endif()
endfunction()

# Create test executables
add_unit_test_executable(unit_tests "regular")
add_unit_test_executable(unit_tests_asan "asan")
add_unit_test_executable(unit_tests_ubsan "ubsan")

include(GoogleTest)

gtest_discover_tests(
    unit_tests
    PROPERTIES LABELS "unit-tests;regular"
    TEST_PREFIX "regular/")

gtest_discover_tests(
    unit_tests_asan
    PROPERTIES LABELS "unit-tests;asan"
    TEST_PREFIX "asan/")

gtest_discover_tests(
    unit_tests_ubsan
    PROPERTIES LABELS "unit-tests;ubsan"
    TEST_PREFIX "ubsan/")
