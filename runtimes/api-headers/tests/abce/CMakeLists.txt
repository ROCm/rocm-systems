# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

# ABCE tests. No GPU, driver or HSA runtime is needed: ABCE is logic over plain
# memory, so everything here runs on any machine. See TEST_PLAN.md for what is
# covered and what is planned.
#
#   cmake -S runtimes/api-headers/tests/abce -B build/abce-tests
#   cmake --build build/abce-tests -j && ctest --test-dir build/abce-tests --output-on-failure
#
#   harness/  fake ring over plain memory, orchestrator fixture, frame walker
#   unit/     GoogleTest suites
#
# Besides the suites, every public header is compiled on its own as C11 and as
# C++17 (target abce_header_self_containment), which is what backs the README's
# claim that the interface is strict C.

cmake_minimum_required(VERSION 3.20)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(abce_tests LANGUAGES C CXX)
  include(CTest)
endif()

find_package(GTest REQUIRED)
find_package(Threads REQUIRED)

set(ABCE_INCLUDE_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../include")
# abce headers include each other as "abce_*.h", and abce_hsa.h finds <hsa/hsa.h>
# from the sibling directory.
set(ABCE_INCLUDE_DIRS "${ABCE_INCLUDE_ROOT}/abce" "${ABCE_INCLUDE_ROOT}")

# Optional: -DABCE_TESTS_SANITIZER=thread (or address, undefined). The ring suite
# is the one that benefits from thread. On kernels with high mmap entropy
# ThreadSanitizer aborts at startup ("unexpected memory mapping"); run it with
# ASLR off:  setarch "$(uname -m)" -R ctest --test-dir <build>
set(ABCE_TESTS_SANITIZER "" CACHE STRING "Sanitizer for the test targets: thread, address, undefined, or empty")
set(ABCE_SANITIZER_FLAGS "")
if(ABCE_TESTS_SANITIZER)
  set(ABCE_SANITIZER_FLAGS -fsanitize=${ABCE_TESTS_SANITIZER} -fno-omit-frame-pointer)
endif()

#===------------------------------------------------------------------------===#
# Unit suites
#===------------------------------------------------------------------------===#

add_executable(abce_unit_tests
  unit/orchestrator_test.cpp
  unit/packet_fields_test.cpp
  unit/profiling_test.cpp
  unit/ring_test.cpp
  unit/validation_test.cpp)
set_target_properties(abce_unit_tests PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON
                                                 CXX_EXTENSIONS OFF)
target_include_directories(abce_unit_tests PRIVATE harness ${ABCE_INCLUDE_DIRS})
target_compile_options(abce_unit_tests PRIVATE -Wall -Wextra ${ABCE_SANITIZER_FLAGS})
target_link_options(abce_unit_tests PRIVATE ${ABCE_SANITIZER_FLAGS})
target_link_libraries(abce_unit_tests PRIVATE GTest::gtest_main Threads::Threads)

include(GoogleTest)
# PRE_TEST: list the tests when ctest runs, not at build time, so a sanitizer build
# does not have to execute the binary while compiling.
gtest_discover_tests(abce_unit_tests DISCOVERY_MODE PRE_TEST)

#===------------------------------------------------------------------------===#
# Header self-containment: each header alone, as C11 and as C++17
#===------------------------------------------------------------------------===#

# abce_device.h needs the HIP device compiler and is covered by the device tier
# in TEST_PLAN.md. abce_cxx.h is C++-only.
set(ABCE_C_HEADERS
  abce_builder.h abce_config.h abce_decode.h abce_frame.h abce_host.h abce_hsa.h
  abce_ring_core.h abce_ring_host.h abce_topology.h abce_types.h sdma_packets.h)
set(ABCE_CXX_HEADERS ${ABCE_C_HEADERS} abce_cxx.h)

set(ABCE_GENERATED_DIR "${CMAKE_CURRENT_BINARY_DIR}/header_tus")
set(ABCE_SELF_CONTAINED_SOURCES "")
foreach(header IN LISTS ABCE_CXX_HEADERS)
  string(REPLACE "." "_" stem "${header}")
  if(NOT header STREQUAL "abce_cxx.h")
    set(c_source "${ABCE_GENERATED_DIR}/${stem}_c11.c")
    file(WRITE "${c_source}" "#include \"${header}\"\ntypedef int ${stem}_c11_anchor_t;\n")
    list(APPEND ABCE_SELF_CONTAINED_SOURCES "${c_source}")
  endif()
  set(cxx_source "${ABCE_GENERATED_DIR}/${stem}_cxx17.cpp")
  file(WRITE "${cxx_source}" "#include \"${header}\"\ntypedef int ${stem}_cxx17_anchor_t;\n")
  list(APPEND ABCE_SELF_CONTAINED_SOURCES "${cxx_source}")
endforeach()

add_library(abce_header_self_containment OBJECT ${ABCE_SELF_CONTAINED_SOURCES})
set_target_properties(abce_header_self_containment PROPERTIES
  C_STANDARD 11 C_STANDARD_REQUIRED ON C_EXTENSIONS OFF
  CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF)
target_include_directories(abce_header_self_containment PRIVATE ${ABCE_INCLUDE_DIRS})
target_compile_options(abce_header_self_containment PRIVATE -Wall -Wextra -Werror)
add_dependencies(abce_unit_tests abce_header_self_containment)
