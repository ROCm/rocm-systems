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
  hrr_recovery_test.cc
  hrr_playback_test.cc
)
set_target_properties(hrr-unit-tests PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON)

# ../include makes the local harness headers (hrr_test_common.hh,
# hrr_test_process.hh) available. hrr_reader.h, hip_playback.h, and
# hrr/hrr_api_args.h come transitively from hrr::reader.
target_include_directories(hrr-unit-tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../include)

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

# HIP entry points load dispatch table slots with hip::LoadDispatchSlot from
# projects/clr/hipamd/src/hip_dispatch_load.hpp, which HRR capture relies on when it
# installs shims into the live table.
set(HRR_HIPAMD_SRC "${CMAKE_CURRENT_SOURCE_DIR}/../../../clr/hipamd/src")
if(EXISTS "${HRR_HIPAMD_SRC}/hip_dispatch_load.hpp")
  # Every entry point in hip_table_interface.cpp goes through that load.
  add_test(NAME hrr-unit-tests-dispatch-readers
    COMMAND ${CMAKE_COMMAND} -DSOURCE=${HRR_HIPAMD_SRC}/hip_table_interface.cpp
            -P ${CMAKE_CURRENT_SOURCE_DIR}/check_dispatch_readers.cmake)

  # The load and HRR's slot store race cleanly under ThreadSanitizer. The test only
  # means something with TSan, so it is built when the compiler supports it.
  include(CheckCXXSourceCompiles)
  set(CMAKE_REQUIRED_FLAGS -fsanitize=thread)
  set(CMAKE_REQUIRED_LINK_OPTIONS -fsanitize=thread)
  check_cxx_source_compiles("int main() { return 0; }" HRR_HAVE_TSAN)
  unset(CMAKE_REQUIRED_FLAGS)
  unset(CMAKE_REQUIRED_LINK_OPTIONS)
  if(HRR_HAVE_TSAN)
    add_executable(hrr-unit-tests-tsan hrr_dispatch_load_tsan_test.cc)
    set_target_properties(hrr-unit-tests-tsan PROPERTIES
      CXX_STANDARD 17
      CXX_STANDARD_REQUIRED ON)
    target_include_directories(hrr-unit-tests-tsan PRIVATE "${HRR_HIPAMD_SRC}")
    target_compile_definitions(hrr-unit-tests-tsan PRIVATE __HIP_PLATFORM_AMD__)
    target_compile_options(hrr-unit-tests-tsan PRIVATE -fsanitize=thread)
    target_link_options(hrr-unit-tests-tsan PRIVATE -fsanitize=thread)
    target_link_libraries(hrr-unit-tests-tsan PRIVATE hip::host)
    # Building the unit tests builds this one too, so a plain ctest run includes it.
    add_dependencies(hrr-unit-tests hrr-unit-tests-tsan)
    add_test(NAME hrr-unit-tests-tsan COMMAND hrr-unit-tests-tsan)
  endif()
endif()
