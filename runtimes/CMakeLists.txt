cmake_minimum_required(VERSION 3.25)
project(ROCmRuntimes LANGUAGES C CXX)

if(NOT CMAKE_GENERATOR STREQUAL "Ninja" OR CMAKE_CONFIGURATION_TYPES)
  message(FATAL_ERROR "The initial runtimes build requires single-configuration Ninja")
endif()
if(CMAKE_CROSSCOMPILING OR NOT CMAKE_SYSTEM_NAME STREQUAL "Linux"
    OR NOT CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|aarch64|AMD64)$")
  message(FATAL_ERROR "The current runtime implementations require native Linux x86-64 or AArch64")
endif()
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug CACHE STRING "Runtime build configuration" FORCE)
endif()
set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS Debug Release RelWithDebInfo MinSizeRel)

include(CTest)
include(cmake/RuntimeRust.cmake)
runtime_rust_initialize()
add_subdirectory(api-headers)
add_subdirectory(ddi)
add_subdirectory(hsa)
runtime_rust_finalize()

if(BUILD_TESTING)
  add_test(NAME runtimes.rust
    COMMAND ${_runtime_cargo_command} test --workspace --all-targets --all-features
      --frozen --profile "${_runtime_profile}")
  set_tests_properties(runtimes.rust PROPERTIES
    WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}" LABELS "host;rust"
    RESOURCE_LOCK runtime_cargo)
endif()
