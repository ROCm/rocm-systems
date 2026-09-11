# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
# Standalone host validation; no GPU, driver or installed runtime is required.
cmake_minimum_required(VERSION 3.16)
project(aql_resident_host_tests LANGUAGES CXX)
enable_testing()
set(CMAKE_CXX_STANDARD 17)
get_filename_component(backend "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)
foreach(name aql_resident_program aql_compiled_bindings)
  add_executable(${name}_test ${name}_test.cpp "${backend}/aql_resident_program.cpp")
  target_include_directories(${name}_test PRIVATE "${backend}")
  # Tests use assertions even in Release builds.
  target_compile_options(${name}_test PRIVATE -UNDEBUG)
  add_test(NAME ${name} COMMAND ${name}_test)
endforeach()
