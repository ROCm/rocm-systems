# ######################################################################################################################
# # Copyright (c) 2025 Advanced Micro Devices, Inc.
# ######################################################################################################################

cmake_minimum_required(VERSION 3.16.0)

project(att-decoder-unit-tests LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

function(create_system_gtest_targets)
    find_path(GTEST_INCLUDE_DIR NAMES gtest/gtest.h)
    find_library(GTEST_LIBRARY NAMES gtest)
    find_library(GTEST_MAIN_LIBRARY NAMES gtest_main)
    find_library(GMOCK_LIBRARY NAMES gmock)
    find_library(GMOCK_MAIN_LIBRARY NAMES gmock_main)

    if(GTEST_INCLUDE_DIR AND GTEST_LIBRARY)
        if(NOT TARGET GTest::gtest)
            add_library(GTest::gtest UNKNOWN IMPORTED)
            set_target_properties(GTest::gtest PROPERTIES IMPORTED_LOCATION "${GTEST_LIBRARY}"
                                                          INTERFACE_INCLUDE_DIRECTORIES "${GTEST_INCLUDE_DIR}")
        endif()

        if(GTEST_MAIN_LIBRARY AND NOT TARGET GTest::gtest_main)
            add_library(GTest::gtest_main UNKNOWN IMPORTED)
            set_target_properties(
                GTest::gtest_main
                PROPERTIES IMPORTED_LOCATION "${GTEST_MAIN_LIBRARY}"
                           INTERFACE_INCLUDE_DIRECTORIES "${GTEST_INCLUDE_DIR}"
                           INTERFACE_LINK_LIBRARIES GTest::gtest)
        endif()
    endif()

    if(GTEST_INCLUDE_DIR AND GMOCK_LIBRARY)
        if(NOT TARGET GTest::gmock)
            add_library(GTest::gmock UNKNOWN IMPORTED)
            set_target_properties(
                GTest::gmock
                PROPERTIES IMPORTED_LOCATION "${GMOCK_LIBRARY}"
                           INTERFACE_INCLUDE_DIRECTORIES "${GTEST_INCLUDE_DIR}"
                           INTERFACE_LINK_LIBRARIES GTest::gtest)
        endif()

        if(GMOCK_MAIN_LIBRARY AND NOT TARGET GTest::gmock_main)
            add_library(GTest::gmock_main UNKNOWN IMPORTED)
            set_target_properties(
                GTest::gmock_main
                PROPERTIES IMPORTED_LOCATION "${GMOCK_MAIN_LIBRARY}"
                           INTERFACE_INCLUDE_DIRECTORIES "${GTEST_INCLUDE_DIR}"
                           INTERFACE_LINK_LIBRARIES GTest::gmock)
        elseif(
            TARGET GTest::gmock
            AND TARGET GTest::gtest_main
            AND NOT TARGET GTest::gmock_main)
            add_library(GTest::gmock_main INTERFACE IMPORTED)
            set_target_properties(GTest::gmock_main PROPERTIES INTERFACE_LINK_LIBRARIES
                                                               "GTest::gmock;GTest::gtest_main")
        endif()
    endif()
endfunction()

function(create_gtest_aliases)
    if(TARGET gtest AND NOT TARGET GTest::gtest)
        add_library(GTest::gtest ALIAS gtest)
    endif()
    if(TARGET gtest_main AND NOT TARGET GTest::gtest_main)
        add_library(GTest::gtest_main ALIAS gtest_main)
    endif()
    if(TARGET gmock AND NOT TARGET GTest::gmock)
        add_library(GTest::gmock ALIAS gmock)
    endif()
    if(TARGET gmock_main AND NOT TARGET GTest::gmock_main)
        add_library(GTest::gmock_main ALIAS gmock_main)
    endif()
endfunction()

find_package(GTest QUIET CONFIG)
if(NOT TARGET GTest::gtest_main OR NOT TARGET GTest::gmock_main)
    find_package(GTest QUIET MODULE)
endif()
if(NOT TARGET GTest::gtest_main OR NOT TARGET GTest::gmock_main)
    create_system_gtest_targets()
endif()
if(NOT TARGET GTest::gtest_main OR NOT TARGET GTest::gmock_main)
    find_path(
        GOOGLETEST_SOURCE_DIR
        NAMES CMakeLists.txt
        PATHS /usr/src/googletest /usr/local/src/googletest
        NO_DEFAULT_PATH)
    if(GOOGLETEST_SOURCE_DIR)
        set(INSTALL_GTEST
            OFF
            CACHE BOOL "" FORCE)
        add_subdirectory(${GOOGLETEST_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR}/googletest EXCLUDE_FROM_ALL)
        create_gtest_aliases()
    endif()
endif()
if(NOT TARGET GTest::gtest_main OR NOT TARGET GTest::gmock_main)
    include(FetchContent)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.14.0)
    set(INSTALL_GTEST
        OFF
        CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
endif()

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
    quick_scan_api_test.cpp
    sqttparser_test.cpp
    mi400_test.cpp
    funcmap_test.cpp)

# Function to create a test executable with specific sanitizer options Re-list library sources for sanitizer builds
# (must be recompiled with sanitizer flags)
file(
    GLOB
    UNIT_TEST_LIB_SOURCES
    ${SRC_DIR}/trie.cpp
    ${SRC_DIR}/rocprof_trace_decoder.cpp
    ${SRC_DIR}/quick_scan_export.cpp
    ${SRC_DIR}/stitch/stitch.cpp
    ${SRC_DIR}/stitch/pctranslator.cpp
    ${SRC_DIR}/trace_parser.cpp
    ${SRC_DIR}/gfx9/*.cpp
    ${SRC_DIR}/gfx10/*.cpp
    ${SRC_DIR}/gfx11/*.cpp
    ${SRC_DIR}/gfx12/*.cpp
    ${SRC_DIR}/mi400/*.cpp)

function(add_unit_test_executable TARGET_NAME SANITIZER_TYPE)
    if(SANITIZER_TYPE STREQUAL "regular")
        add_executable(${TARGET_NAME} ${TEST_FILES})
        target_link_libraries(${TARGET_NAME} PRIVATE rocprof-trace-decoder-static)
    else()
        add_executable(${TARGET_NAME} ${TEST_FILES} ${UNIT_TEST_LIB_SOURCES})
    endif()

    target_include_directories(${TARGET_NAME} PRIVATE ${SRC_DIR} ${INC_DIR})

    if(NOT SANITIZER_TYPE STREQUAL "regular")
        if(DISABLE_COMGR)
            target_compile_definitions(${TARGET_NAME} PRIVATE ROCPROF_TRACE_DECODER_COMGR_DISABLED)
        else()
            if(DEFINED ROCM_INCLUDE_DIR)
                target_include_directories(${TARGET_NAME} PRIVATE ${ROCM_INCLUDE_DIR})
            endif()
            if(DEFINED COMGR_LIB)
                target_link_libraries(${TARGET_NAME} PRIVATE ${COMGR_LIB})
            endif()
        endif()
    endif()

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

add_executable(unit_tests_logging rdna_logging_test.cpp)
target_include_directories(unit_tests_logging PRIVATE ${SRC_DIR} ${INC_DIR})
target_link_libraries(unit_tests_logging PRIVATE rocprof-trace-decoder-static GTest::gtest_main GTest::gmock_main)

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

gtest_discover_tests(
    unit_tests_logging
    PROPERTIES LABELS "unit-tests;regular;logging" ENVIRONMENT "SQTT_LOGGING=1"
    TEST_PREFIX "logging/")
