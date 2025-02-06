rocprofiler_deactivate_clang_tidy()

include(GoogleTest)

set(ROCPROFILER_LIB_PC_SAMPLING_TEST_SOURCES
    configure_service.cpp cid_manager.cpp
    # samples_processing.cpp
    pc_sampling_vs_counter_collection.cpp query_configuration.cpp)
set(ROCPROFILER_LIB_PC_SAMPLING_TEST_HEADERS pc_sampling_internals.hpp)

add_executable(pcs-test)

target_sources(pcs-test PRIVATE ${ROCPROFILER_LIB_PC_SAMPLING_TEST_SOURCES}
                                ${ROCPROFILER_LIB_PC_SAMPLING_TEST_HEADERS})

target_link_libraries(
    pcs-test
    PRIVATE rocprofiler-sdk::rocprofiler-sdk-common-library
            rocprofiler-sdk::rocprofiler-sdk-static-library GTest::gtest
            GTest::gtest_main)

gtest_add_tests(
    TARGET pcs-test
    SOURCES ${ROCPROFILER_LIB_COUNTER_TEST_SOURCES}
    TEST_LIST pcs-tests_TESTS
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

set_tests_properties(
    ${pcs-tests_TESTS}
    PROPERTIES TIMEOUT 45 LABELS "unittests;pc-sampling" SKIP_REGULAR_EXPRESSION
               "PC sampling unavailable" FAIL_REGULAR_EXPRESSION
               "${ROCPROFILER_DEFAULT_FAIL_REGEX}")
