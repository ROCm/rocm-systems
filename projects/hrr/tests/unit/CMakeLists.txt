# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT


add_executable(hrr-unit-tests
  hrr_format_test.cc
  hrr_host_rect_test.cc
  hrr_limits_test.cc
  hrr_recovery_test.cc
  hrr_repair_test.cc
  hrr_slots_test.cc
  hrr_playback_test.cc
)
set_target_properties(hrr-unit-tests PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON)

# The code object scrubber needs only <cstring>. Without the CLR sources its cases register as
# skipped, so the binary keeps matching tests/config/unit.yaml.
target_sources(hrr-unit-tests PRIVATE hrr_scrub_test.cc)
set(HRR_CAPTURE_SRC ${CMAKE_CURRENT_SOURCE_DIR}/../../../clr/hipamd/src/hrr)
if(EXISTS ${HRR_CAPTURE_SRC}/hip_capture_scrub.cpp)
  target_sources(hrr-unit-tests PRIVATE ${HRR_CAPTURE_SRC}/hip_capture_scrub.cpp)
  target_include_directories(hrr-unit-tests PRIVATE ${HRR_CAPTURE_SRC})
  target_compile_definitions(hrr-unit-tests PRIVATE HRR_HAVE_CAPTURE_SCRUB=1)
endif()

target_include_directories(hrr-unit-tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../include)

# hrr_test_process.hh starts a std::thread watchdog. Before glibc 2.34 the pthread symbols live in
# libpthread, so the link needs Threads::Threads (-pthread).
find_package(Threads REQUIRED)

target_link_libraries(hrr-unit-tests PRIVATE
  hrr::reader
  hip::host
  Threads::Threads
  Catch2::Catch2WithMain)

# HRR_API_ARGS_IMPLEMENTATION emits hrr_api_names[] with internal linkage per translation unit,
# matching hrr_reader. Define it so each test TU uses its own copy.
target_compile_definitions(hrr-unit-tests PRIVATE
  HRR_API_ARGS_IMPLEMENTATION
  __HIP_PLATFORM_AMD__
  ROCM_BIN_PATH="${ROCM_PATH}/bin"
)

# hrr_recovery_test.cc spawns hrr-playback for --repair and --info. Without the in-tree target
# (HRR_BUILD_PLAYBACK=OFF) those cases need hrr-playback on PATH.
if(TARGET hrr-playback)
  target_compile_definitions(hrr-unit-tests PRIVATE
    HRR_PLAYBACK_EXE="$<TARGET_FILE:hrr-playback>")
  add_dependencies(hrr-unit-tests hrr-playback)
else()
  target_compile_definitions(hrr-unit-tests PRIVATE
    HRR_PLAYBACK_EXE="hrr-playback")
endif()

add_test(NAME hrr-unit-tests COMMAND hrr-unit-tests)

# Handler tests link hip_playback.cpp and the generated handlers and call them with crafted
# records, so they need no GPU. AddressSanitizer is on by default to catch over-reads
# (HRR_HANDLER_TESTS_ASAN=OFF without the runtime). POSIX only: the util header uses mkstemp and dup2.
if(TARGET hrr-playback AND NOT WIN32)
  option(HRR_HANDLER_TESTS_ASAN "Build hrr-handler-tests with AddressSanitizer" ON)

  set(HRR_HANDLER_TEST_SOURCES
    hrr_query_retry_test.cc
    hrr_host_budget_test.cc
    hrr_graph_deps_test.cc
    hrr_refusals_test.cc
    hrr_blob_refusal_test.cc
    hrr_access_policy_window_test.cc
    hrr_capture_state_test.cc
    hrr_unmapped_note_test.cc
    hrr_symbol_copy_bounds_test.cc
    hrr_unregister_fail_test.cc
    hrr_snapshot_order_test.cc
    hrr_kernel_args_signature_test.cc
    hrr_memcpy3d_array_test.cc
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
