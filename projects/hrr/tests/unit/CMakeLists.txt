# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

# HRR unit tests.
#
# These exercise the archive wire format, crash-truncation recovery in the
# reader, and CPU-side playback invariants. They do not issue HIP runtime calls,
# but their sources include <hip/hip_runtime.h> (via the local test harness and
# the playback helper header), so HIP headers are required to build them.

add_executable(hrr-unit-tests
  hrr_format_test.cc
  hrr_limits_test.cc
  hrr_recovery_test.cc
  hrr_repair_test.cc
  hrr_playback_test.cc
)
set_target_properties(hrr-unit-tests PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON)

# ${CMAKE_CURRENT_SOURCE_DIR}/.. makes the local harness headers
# (hrr_test_common.hh, hrr_test_process.hh) available. hrr_reader.h,
# hip_playback.h, and hrr/hrr_api_args.h come transitively from hrr::reader.
target_include_directories(hrr-unit-tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)

target_link_libraries(hrr-unit-tests PRIVATE
  hrr::reader
  hip::host
  Catch2::Catch2WithMain)

# HRR_API_ARGS_IMPLEMENTATION emits the hrr_api_names[] table with internal
# linkage per translation unit, matching hrr_reader; keep it defined here so the
# test TUs use their own copy rather than the extern declaration.
target_compile_definitions(hrr-unit-tests PRIVATE
  HRR_API_ARGS_IMPLEMENTATION
  __HIP_PLATFORM_AMD__
  ROCM_BIN_PATH="${ROCM_PATH}/bin"
)

# hrr_recovery_test.cc spawns hrr-playback for its --repair / --info cases.
#
#   * When the playback tool is built in this same tree (HRR_BUILD_PLAYBACK=ON),
#     point at the just-built binary via its absolute target path so the tests
#     exercise the in-tree hrr-playback rather than any installed copy, and make
#     the unit target depend on it so it is built first.
#   * Otherwise fall back to the bare name "hrr-playback", which SpawnProc
#     resolves from PATH (the test prepends ROCM_BIN_PATH). This keeps the
#     CPU-only reader/format tests buildable with HRR_BUILD_PLAYBACK=OFF; the two
#     recovery cases that invoke hrr-playback then require an hrr-playback on
#     PATH at run time (build with HRR_BUILD_PLAYBACK=ON to exercise them in
#     tree).
if(TARGET hrr-playback)
  target_compile_definitions(hrr-unit-tests PRIVATE
    HRR_PLAYBACK_EXE="$<TARGET_FILE:hrr-playback>")
  add_dependencies(hrr-unit-tests hrr-playback)
else()
  target_compile_definitions(hrr-unit-tests PRIVATE
    HRR_PLAYBACK_EXE="hrr-playback")
endif()

add_test(NAME hrr-unit-tests COMMAND hrr-unit-tests)

# Handler-level tests: playback handlers and helpers in hip_playback.cpp (and
# the generated handlers), linked in and called with crafted records or stand-ins
# for the HIP call. They link the HIP runtime like hrr-playback does but reach no
# HIP call, so they need no GPU. Each change that adds handler code adds its test
# file to HRR_HANDLER_TEST_SOURCES.
#
# Built with AddressSanitizer by default so an over-read of a record aborts the
# run instead of passing by luck; set HRR_HANDLER_TESTS_ASAN=OFF for a
# toolchain without the sanitizer runtime.
if(TARGET hrr-playback)
  option(HRR_HANDLER_TESTS_ASAN "Build hrr-handler-tests with AddressSanitizer" ON)

  set(HRR_HANDLER_TEST_SOURCES
    hrr_query_retry_test.cc
    hrr_host_budget_test.cc
  )

  add_executable(hrr-handler-tests
    ${HRR_HANDLER_TEST_SOURCES}
    ${CMAKE_CURRENT_SOURCE_DIR}/../../playback/hip_playback.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../../playback/hip_playback_generated.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../../playback/hrr_region_map.cpp
  )
  set_target_properties(hrr-handler-tests PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON)
  target_include_directories(hrr-handler-tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../playback)
  target_compile_definitions(hrr-handler-tests PRIVATE
    HIP_ABI_IMPL
    __HIP_PLATFORM_AMD__)
  find_package(Threads REQUIRED)
  target_link_libraries(hrr-handler-tests PRIVATE
    hrr::reader hip::host Threads::Threads Catch2::Catch2WithMain)
  if(HRR_HANDLER_TESTS_ASAN AND NOT WIN32)
    target_compile_options(hrr-handler-tests PRIVATE -fsanitize=address -fno-omit-frame-pointer)
    target_link_options(hrr-handler-tests PRIVATE -fsanitize=address)
  endif()
  add_test(NAME hrr-handler-tests COMMAND hrr-handler-tests)
endif()
