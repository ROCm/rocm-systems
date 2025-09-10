#
#
#
project(rocprofiler-sdk-tests-buffering LANGUAGES C CXX)

include(GoogleTest)

set(buffering_sources buffering-serial.cpp buffering-parallel.cpp buffering-save-load.cpp)

add_executable(buffering-tests)
target_sources(buffering-tests PRIVATE ${buffering_sources})
target_link_libraries(
    buffering-tests
    PRIVATE rocprofiler-sdk::rocprofiler-sdk-headers
            rocprofiler-sdk::rocprofiler-sdk-common-library GTest::gtest
            GTest::gtest_main)

rocprofiler_add_unit_test(TARGET buffering-tests SOURCES ${buffering_sources})
