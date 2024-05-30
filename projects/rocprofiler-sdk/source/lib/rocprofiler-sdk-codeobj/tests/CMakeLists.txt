rocprofiler_deactivate_clang_tidy()

include(GoogleTest)
add_executable(codeobj-library-test)

set(CODEOBJ_LIB_TEST_SOURCES "codeobj_library_test.cpp")
target_sources(codeobj-library-test PRIVATE ${CODEOBJ_LIB_TEST_SOURCES})

target_link_libraries(
    codeobj-library-test
    PRIVATE rocprofiler-sdk::rocprofiler-static-library
            rocprofiler-sdk::rocprofiler-glog
            rocprofiler-sdk::rocprofiler-hsa-runtime
            rocprofiler-sdk::rocprofiler-hip
            rocprofiler-sdk::rocprofiler-common-library
            GTest::gtest
            GTest::gtest_main
            rocprofiler-sdk::rocprofiler-sdk-codeobj)

gtest_add_tests(
    TARGET codeobj-library-test
    SOURCES ${CODEOBJ_LIB_TEST_SOURCES}
    TEST_LIST codeobj-library-test_TESTS
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

set_tests_properties(${codeobj-library-test_TESTS} PROPERTIES TIMEOUT 10 LABELS
                                                              "unittests")

target_compile_definitions(codeobj-library-test
                           PRIVATE -DCODEOBJ_BINARY_DIR=\"${CMAKE_CURRENT_BINARY_DIR}/\")

configure_file(smallkernel.b smallkernel.b COPYONLY)
configure_file(hipcc_output.s hipcc_output.s COPYONLY)
