# These tests can also be configured on their own without an installed ROCm
# stack: cmake -S rocrtst/suites/functional/code_object -B build-code-object
cmake_minimum_required(VERSION 3.16)

get_filename_component(ROCRTST_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../.." ABSOLUTE)
get_filename_component(HSA_RUNTIME_ROOT "${ROCRTST_ROOT}/../runtime/hsa-runtime" ABSOLUTE)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(rocr-code-object-tests LANGUAGES CXX)
  enable_testing()
  set(GOOGLE_TEST_FRWK_NAME google-test-frwk64)
  add_subdirectory(${ROCRTST_ROOT}/gtest ${CMAKE_CURRENT_BINARY_DIR}/gtest)
endif()

find_package(Threads REQUIRED)
find_package(LibElf QUIET CONFIG)
if(NOT LibElf_FOUND)
  list(APPEND CMAKE_MODULE_PATH "${HSA_RUNTIME_ROOT}/cmake_modules")
  find_package(LibElf REQUIRED)
endif()

# Compile the production ELF reader and ISA registry directly. No HSA runtime
# initialization or GPU is needed to reproduce invalid ISA-name lookups.
add_executable(code_object_isa_test
  code_object_isa_test.cc
  ${HSA_RUNTIME_ROOT}/libamdhsacode/amd_elf_image.cpp
  ${HSA_RUNTIME_ROOT}/libamdhsacode/amd_hsa_code.cpp
  ${HSA_RUNTIME_ROOT}/libamdhsacode/amd_hsa_code_util.cpp
  ${HSA_RUNTIME_ROOT}/core/runtime/isa.cpp)
target_compile_features(code_object_isa_test PRIVATE cxx_std_17)
target_compile_definitions(code_object_isa_test PRIVATE LITTLEENDIAN_CPU=1)
target_include_directories(code_object_isa_test PRIVATE
  ${HSA_RUNTIME_ROOT}
  ${HSA_RUNTIME_ROOT}/libamdhsacode
  ${ROCRTST_ROOT}/gtest/include)
target_link_libraries(code_object_isa_test PRIVATE
  ${GOOGLE_TEST_FRWK_NAME} elf::elf Threads::Threads)
add_test(NAME code_object_isa_test COMMAND code_object_isa_test)
set_tests_properties(code_object_isa_test PROPERTIES
  LABELS "quick;standard;pr;pre-commit"
  TIMEOUT 60)
