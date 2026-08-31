#
#
#
cmake_minimum_required(VERSION 3.21.0 FATAL_ERROR)

project(rocprofiler-sdk-tests-tools-kinetoesque LANGUAGES CXX)

find_package(rocprofiler-sdk REQUIRED)

rocprofiler_tests_configure_python_package_sources(torchesque/kinetoesque __init__.py)

add_library(kinetoesque-tool SHARED)
target_sources(kinetoesque-tool PRIVATE kinetoesque-tool.cpp)
target_link_libraries(kinetoesque-tool PRIVATE rocprofiler-sdk::rocprofiler-sdk
                                               rocprofiler-sdk::tests-build-flags)
set_target_properties(
    kinetoesque-tool
    PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib/rocprofiler-sdk"
               INSTALL_RPATH "\$ORIGIN:\$ORIGIN/..")
