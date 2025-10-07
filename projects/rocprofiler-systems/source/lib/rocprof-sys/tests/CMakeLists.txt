find_package(GTest REQUIRED)
find_package(PkgConfig REQUIRED)

set(smi_tests_source 
    ${CMAKE_CURRENT_SOURCE_DIR}/amd_smi/service_tests.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/amd_smi/processor_tests.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/amd_smi/amd_smi_tests.cpp
)

add_executable(library-unit-tests ${smi_tests_source})
target_link_libraries(
    library-unit-tests PRIVATE 
    rocprofiler-systems::rocprofiler-systems-common-library
    rocprofiler-systems::rocprofiler-systems-core
    rocprofiler-systems::rocprofiler-systems-headers
    rocprofiler-systems::rocprofiler-systems-interface-library
    rocprofiler-systems::rocprofiler-systems-binary
    GTest::gtest 
    GTest::gtest_main 
    GTest::gmock
    amd-smi::amd-smi
)

# Set test output directory
set_target_properties(library-unit-tests PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests/bin
)

include(GoogleTest)
gtest_discover_tests(library-unit-tests
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/tests/bin
    PROPERTIES
        LABELS "unit"
        TIMEOUT 60
)

# # Add custom test targets
# add_custom_target(run-tests
#     COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure --parallel ${CMAKE_BUILD_PARALLEL_LEVEL}
#     DEPENDS library-tests
#     WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
#     COMMENT "Running all tests"
# )
