include(GoogleTest)
find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})

add_executable(utility_tests)
SET(AQLPROFILE_UTILITY_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/util_tests.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../hsa_rsrc_factory.cpp
)

target_sources(utility_tests PRIVATE ${AQLPROFILE_UTILITY_SOURCES})
target_include_directories(utility_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${LIB_DIR} ${LIB_DIR}/core/include)
target_link_libraries(
    utility_tests
    PRIVATE
            hsa-runtime64::hsa-runtime64
            GTest::gtest
            GTest::gtest_main)


gtest_add_tests(
    TARGET utility_tests
    SOURCES ${AQLPROFILE_UTILITY_SOURCES}
    TEST_LIST utility_tests_TESTS
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

set_tests_properties(
    ${utility_tests_TESTS} PROPERTIES TIMEOUT 45 LABELS "unittests" FAIL_REGULAR_EXPRESSION
                                 "${AQLPROFILE_DEFAULT_FAIL_REGEX}")