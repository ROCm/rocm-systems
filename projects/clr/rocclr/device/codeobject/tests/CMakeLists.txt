# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# codeobject_test — Catch2 tests for the code object formats read without comgr: offload bundle
# selection, compressed bundles (CCOB), ELF image size, notes and .dynsym names, metadata notes
# and MessagePack. The inputs are built in the tests, so no GPU, compiler or comgr is needed.
#
# codeobject_fuzz — libFuzzer harness over the same parsers; built only with clang.

include(FetchContent)
FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG        v3.6.0
  GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(Catch2)
list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
include(Catch)

add_executable(codeobject_test
  rocclr_stubs.cpp
  test_offload_bundle.cpp
  test_compressed_bundle.cpp
  test_elf.cpp
  test_metadata_notes.cpp
  test_msgpack.cpp
)
set_target_properties(codeobject_test PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)
target_link_libraries(codeobject_test PRIVATE rocclr Catch2::Catch2WithMain ${CMAKE_DL_LIBS})
catch_discover_tests(codeobject_test)

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  # The parsers are compiled in again so libFuzzer's coverage instrumentation sees them; the
  # rest (ISA table, Os) comes from the library.
  add_executable(codeobject_fuzz
    fuzz_codeobject.cpp
    rocclr_stubs.cpp
    ${ROCCLR_SRC_DIR}/device/codeobject/compressed_bundle.cpp
    ${ROCCLR_SRC_DIR}/device/codeobject/elf_header.cpp
    ${ROCCLR_SRC_DIR}/device/codeobject/metadata_notes.cpp
    ${ROCCLR_SRC_DIR}/device/codeobject/msgpack.cpp
    ${ROCCLR_SRC_DIR}/device/codeobject/offload_bundle.cpp)
  set_target_properties(codeobject_fuzz PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)
  target_compile_options(codeobject_fuzz PRIVATE -fsanitize=fuzzer,address)
  target_link_options(codeobject_fuzz PRIVATE -fsanitize=fuzzer,address)
  target_link_libraries(codeobject_fuzz PRIVATE rocclr ${CMAKE_DL_LIBS})
endif()
