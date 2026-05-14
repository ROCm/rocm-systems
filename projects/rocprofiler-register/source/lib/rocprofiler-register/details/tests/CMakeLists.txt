#
# Unit tests for rocprofiler-register details/* helpers.
#
# These tests are gated on GTest being available. The repository does not
# currently vendor GTest; once GTest is plumbed in (likely as part of broader
# unit-test infrastructure), this directory will compile its sources into a
# single executable per CONTRIBUTING.md ("compile all unit-test sources in a
# folder into one executable; register with gtest_add_tests").
#
find_package(GTest QUIET)

if(NOT GTest_FOUND)
    message(
        STATUS
            "rocprofiler-register details unit tests skipped: GTest not found")
    return()
endif()

add_executable(rocprofiler-register-details-tests)
# Per CONTRIBUTING.md, all unit-test sources in this folder compile into one
# executable. Tests pull in the impl translation units they exercise.
set(rocprofiler_register_details_test_sources environment.cpp loader.cpp
                                              logging.cpp pe_parser.cpp)

# utility.cpp is required because loader_posix.cpp calls utility::delimit,
# whose explicit template instantiations live there (extern template in
# utility.hpp). It is platform-agnostic and harmless to compile in on Windows.
set(rocprofiler_register_details_test_impls ../environment.cpp ../logging.cpp
                                            ../utility.cpp)

if(WIN32)
    list(APPEND rocprofiler_register_details_test_impls
         ../platform/loader_win.cpp ../platform/pe_parser_win.cpp)
else()
    list(APPEND rocprofiler_register_details_test_impls
         ../platform/loader_posix.cpp ../platform/pe_parser_posix.cpp)
endif()

target_sources(
    rocprofiler-register-details-tests
    PRIVATE ${rocprofiler_register_details_test_sources}
            ${rocprofiler_register_details_test_impls})

target_link_libraries(
    rocprofiler-register-details-tests
    PRIVATE rocprofiler-register::headers rocprofiler-register::build-flags
            GTest::gtest GTest::gtest_main fmt::fmt glog::glog)

if(WIN32)
    target_link_libraries(rocprofiler-register-details-tests PRIVATE psapi)
endif()

target_include_directories(
    rocprofiler-register-details-tests
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../.. ${PROJECT_SOURCE_DIR}/source/lib
            ${PROJECT_SOURCE_DIR}/source ${PROJECT_BINARY_DIR}/source)

include(GoogleTest)
gtest_add_tests(
    TARGET rocprofiler-register-details-tests TEST_LIST details_tests SOURCES
    ${rocprofiler_register_details_test_sources})

# Per CONTRIBUTING.md, unit tests may opt out of clang-tidy.
if(COMMAND rocprofiler_register_deactivate_clang_tidy)
    rocprofiler_register_deactivate_clang_tidy(rocprofiler-register-details-tests)
endif()
