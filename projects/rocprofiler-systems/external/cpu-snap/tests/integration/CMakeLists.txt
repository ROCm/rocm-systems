# Integration tests that link more than one component (e.g. core + sampling).
# Pure-core tests live in tests/core/.

if(CPU_SNAP_BUILD_SAMPLING AND CPU_SNAP_BUILD_CORE AND CPU_SNAP_BUILD_PROCESSING)
    add_executable(integration_tests
        test_reader.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../test_coverage_extras.cpp)
    target_link_libraries(integration_tests PRIVATE
        cpu_snap::core cpu_snap::sampling cpu_snap::processing cpu_snap_test_support
        GTest::gtest GTest::gtest_main)
    target_compile_options(integration_tests PRIVATE -Wall -Wextra -Wpedantic -O2 -g)
    target_compile_definitions(integration_tests PRIVATE
        SAMPLING_READER_FIXTURES="${CMAKE_SOURCE_DIR}/tests/fixtures/reader")
    gtest_discover_tests(integration_tests PROPERTIES TIMEOUT 30)
endif()
