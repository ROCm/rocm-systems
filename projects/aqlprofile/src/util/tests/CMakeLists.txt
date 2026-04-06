include(GoogleTest)
find_package(GTest QUIET)
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
            $<$<NOT:$<BOOL:${WIN32}>>:hsa-runtime64::hsa-runtime64>
            $<$<BOOL:${WIN32}>:${AMDHIP64_LIB}>
            GTest::gtest
            GTest::gtest_main
            $<$<NOT:$<BOOL:${WIN32}>>:dl>)

gtest_discover_tests(
    utility_tests
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    PROPERTIES TIMEOUT 45 LABELS "unittests" FAIL_REGULAR_EXPRESSION "${AQLPROFILE_DEFAULT_FAIL_REGEX}")
