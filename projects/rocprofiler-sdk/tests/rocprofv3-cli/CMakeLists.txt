#
# rocprofv3 CLI argument-parsing unit tests
#
# These tests exercise the pure-Python argument parsing in the rocprofv3 script
# and require no GPU or ROCm runtime.
#
cmake_minimum_required(VERSION 3.21.0 FATAL_ERROR)

project(
    rocprofiler-sdk-tests-rocprofv3-cli
    LANGUAGES CXX
    VERSION 0.0.0)

find_package(Python3 REQUIRED)

add_test(NAME rocprofv3-cli-argument-parsing
         COMMAND ${Python3_EXECUTABLE} -m pytest -v
                 ${CMAKE_CURRENT_SOURCE_DIR}/test_argument_parsing.py)

set_tests_properties(rocprofv3-cli-argument-parsing PROPERTIES LABELS "unit-tests"
                                                               TIMEOUT 60)
