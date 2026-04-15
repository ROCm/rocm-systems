cmake_minimum_required(VERSION 3.18)
project(rocdecode_unit_tests LANGUAGES CXX)

find_package(GTest REQUIRED)

add_executable(surface_format_selection_test
    surface_format_selection_test.cpp
)

target_include_directories(surface_format_selection_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../..
)

target_link_libraries(surface_format_selection_test PRIVATE GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(surface_format_selection_test)
