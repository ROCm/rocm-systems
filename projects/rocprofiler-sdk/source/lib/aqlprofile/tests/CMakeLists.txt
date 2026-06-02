#
#
#
rocprofiler_deactivate_clang_tidy()

include(GoogleTest)

add_library(aqlprofile-ctrl-tests-flags INTERFACE)
target_compile_options(aqlprofile-ctrl-tests-flags
                       INTERFACE -Wno-shadow -Wno-unused-parameter -Wno-unused-variable)

set(AQLPROFILE_CTRL_TEST_SOURCES aql_profile_v2_ctrl_tests.cpp)

add_executable(aqlprofile-ctrl-tests)

target_sources(
    aqlprofile-ctrl-tests
    PRIVATE ${AQLPROFILE_CTRL_TEST_SOURCES}
            ${CMAKE_CURRENT_SOURCE_DIR}/simple_convolution/simple_convolution.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/util/hsa_rsrc_factory.cpp)

# Copy pre-built hsaco files to the build directories (ctest runs from
# CMAKE_CURRENT_BINARY_DIR; direct invocation runs from CMAKE_RUNTIME_OUTPUT_DIRECTORY)
file(GLOB HSACO_FILES ${CMAKE_CURRENT_SOURCE_DIR}/simple_convolution/*.hsaco)
foreach(HSACO_FILE ${HSACO_FILES})
    configure_file(${HSACO_FILE} ${CMAKE_CURRENT_BINARY_DIR}/ COPYONLY)
    configure_file(${HSACO_FILE} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/ COPYONLY)
endforeach()
target_include_directories(
    aqlprofile-ctrl-tests
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}
            ${CMAKE_CURRENT_SOURCE_DIR}/../core ${CMAKE_CURRENT_SOURCE_DIR}/..
            ${CMAKE_CURRENT_SOURCE_DIR}/ctrl)
set_target_properties(aqlprofile-ctrl-tests PROPERTIES C_STANDARD 11 C_STANDARD_REQUIRED
                                                                     ON)
target_link_options(aqlprofile-ctrl-tests PRIVATE -Wl,--allow-multiple-definition)

target_link_libraries(
    aqlprofile-ctrl-tests
    PRIVATE -Wl,--start-group
            rocprofiler-sdk::rocprofiler-sdk-static-library
            rocprofiler-sdk::rocprofiler-sdk-common-library
            rocprofiler-sdk::rocprofiler-sdk-aqlprofile
            -Wl,--end-group
            rocprofiler-sdk::rocprofiler-sdk-amd-comgr
            rocprofiler-sdk::rocprofiler-sdk-drm
            aqlprofile-ctrl-tests-flags
            hsa-runtime64::hsa-runtime64
            GTest::gtest
            GTest::gtest_main
            GTest::gmock
            GTest::gmock_main)

# This test has a custom main() (not gtest TEST macros), so register it manually for ctest
add_test(
    NAME rocprofiler_sdk.unit.aqlprofile.ctrl.main
    COMMAND aqlprofile-ctrl-tests
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
set_tests_properties(
    rocprofiler_sdk.unit.aqlprofile.ctrl.main
    PROPERTIES TIMEOUT 30 LABELS "unittests;aqlprofile" FAIL_REGULAR_EXPRESSION
               "${ROCPROFILER_DEFAULT_FAIL_REGEX}")
