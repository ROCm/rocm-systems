#
# Integration test applications
#
cmake_minimum_required(VERSION 3.21.0 FATAL_ERROR)

project(rocprofiler-test-apps LANGUAGES C CXX)

# applications used by integration tests
add_subdirectory(reproducible-runtime)
add_subdirectory(transpose)
