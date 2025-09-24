find_package(GTest REQUIRED)
find_package(PkgConfig REQUIRED)

set(smi_tests_source 
    ${CMAKE_CURRENT_SOURCE_DIR}/smi/service_tests.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/smi/processor_tests.cpp
)

add_executable(rocprof-sys-unit-tests ${smi_tests_source})

target_link_libraries(${CMAKE_PROJECT_NAME}-tests PRIVATE smi-library GTest::gtest GTest::gtest_main GTest::gmock)

# Set test output directory
set_target_properties(${CMAKE_PROJECT_NAME}-tests PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests/bin
)

if(ENABLE_COVERAGE)
    target_compile_options(${CMAKE_PROJECT_NAME}-tests PRIVATE --coverage)
    target_link_libraries(${CMAKE_PROJECT_NAME}-tests PRIVATE --coverage)
endif()

include(GoogleTest)
gtest_discover_tests(example-tests
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/tests/bin
    PROPERTIES
        LABELS "unit"
        TIMEOUT 60
)

# Add custom test targets
add_custom_target(run-tests
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure --parallel ${CMAKE_BUILD_PARALLEL_LEVEL}
    DEPENDS example-tests
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running all tests"
)
