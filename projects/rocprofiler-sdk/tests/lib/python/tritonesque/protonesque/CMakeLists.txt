#
#
#
cmake_minimum_required(VERSION 3.21.0 FATAL_ERROR)

project(rocprofiler-sdk-tests-tools-protonesque LANGUAGES CXX)

find_package(rocprofiler-sdk REQUIRED)

rocprofiler_tests_configure_python_package_sources(tritonesque/protonesque __init__.py)

add_library(protonesque-tool SHARED)
target_sources(protonesque-tool PRIVATE protonesque-tool.cpp)
target_link_libraries(protonesque-tool PRIVATE rocprofiler-sdk::rocprofiler-sdk
                                               rocprofiler-sdk::tests-build-flags)
set_target_properties(
    protonesque-tool
    PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib/rocprofiler-sdk"
               INSTALL_RPATH "\$ORIGIN:\$ORIGIN/..")
