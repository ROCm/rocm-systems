cmake_minimum_required(VERSION 3.24 FATAL_ERROR)

project(rocprofiler-tool-test-app-sdma-copy LANGUAGES CXX)

find_package(Threads REQUIRED)
if(NOT TARGET hsa-runtime64::hsa-runtime64)
    find_package(hsa-runtime64 REQUIRED)
endif()

add_executable(sdma-copy sdma-copy.cpp)
target_compile_features(sdma-copy PRIVATE cxx_std_17)
target_compile_options(sdma-copy PRIVATE -Wpedantic)
target_link_libraries(sdma-copy PRIVATE Threads::Threads hsa-runtime64::hsa-runtime64)

if(PROJECT_IS_TOP_LEVEL)
    enable_testing()
endif()

add_test(NAME sdma-copy-profiling COMMAND sdma-copy)
set_tests_properties(
    sdma-copy-profiling
    PROPERTIES TIMEOUT 60
               ENVIRONMENT "HSA_ENABLE_SDMA=1"
               SKIP_RETURN_CODE 77
               LABELS "integration-tests;sdma")
