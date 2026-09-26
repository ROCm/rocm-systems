# Configure these GPU-free tests independently:
# cmake -S shared/amdgpu-windows-interop/sc/HSAIL/ext/libamdhsacode/tests -B build-pal-code-object
cmake_minimum_required(VERSION 3.16)
project(pal-code-object-tests LANGUAGES CXX)
enable_testing()

find_package(Threads REQUIRED)
find_package(GTest REQUIRED)
find_path(LIBELF_INCLUDE_DIR libelf.h)
find_library(LIBELF_LIBRARY elf)
if(NOT LIBELF_INCLUDE_DIR OR NOT LIBELF_LIBRARY)
  message(FATAL_ERROR "The code-object tests require libelf headers and library")
endif()

get_filename_component(HSAIL_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../.." ABSOLUTE)
set(CODE_READER_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/..")

# Compile the same reader used by CLR's PAL backend, using system libelf and
# disabling only SP3 disassembly. No PAL driver or HSA runtime is required.
add_executable(pal_code_object_isa_test
  code_object_isa_test.cc
  ${CODE_READER_ROOT}/amd_elf_image.cpp
  ${CODE_READER_ROOT}/amd_hsa_code.cpp
  ${CODE_READER_ROOT}/amd_hsa_code_util.cpp)
target_compile_features(pal_code_object_isa_test PRIVATE cxx_std_17)
target_compile_definitions(pal_code_object_isa_test PRIVATE
  LITTLEENDIAN_CPU=1 NO_SI_SP3 GFX11_BUILD GFX12_BUILD)
target_include_directories(pal_code_object_isa_test PRIVATE
  ${HSAIL_ROOT}/include
  ${HSAIL_ROOT}/hsail-tools/libHSAIL
  ${CODE_READER_ROOT}
  ${LIBELF_INCLUDE_DIR})
target_link_libraries(pal_code_object_isa_test PRIVATE
  GTest::Main ${LIBELF_LIBRARY} Threads::Threads)
add_test(NAME pal_code_object_isa_test COMMAND pal_code_object_isa_test)
set_tests_properties(pal_code_object_isa_test PROPERTIES TIMEOUT 60)
