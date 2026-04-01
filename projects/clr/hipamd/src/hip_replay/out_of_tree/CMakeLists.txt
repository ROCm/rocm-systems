cmake_minimum_required(VERSION 3.21)
project(hip_replay_out_of_tree LANGUAGES C CXX)

# All static libs linked into shared objects need PIC
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ============================================================================
# Out-of-tree HIP Record & Replay
#
# Builds:
#   Linux:   libhrr_record.so  (LD_PRELOAD interposer)
#   Windows: amdhip64_7.dll    (proxy DLL)
#   Both:    hrr-replay, hrr-bench, hrr-info tools
# ============================================================================

# --- Shared code object parser (C, no dependencies) ---
add_library(hrr_code_object STATIC hrr_code_object.c)
target_include_directories(hrr_code_object PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# --- Shared trace writer (C, no dependencies) ---
add_library(hrr_trace_writer STATIC hrr_trace_writer.c)
target_link_libraries(hrr_trace_writer PRIVATE hrr_code_object)
target_include_directories(hrr_trace_writer PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
if(NOT WIN32)
  target_link_libraries(hrr_trace_writer PRIVATE pthread)
endif()

# --- Platform-specific recording library ---
if(WIN32)
  # Windows proxy DLL
  add_library(amdhip64_7 SHARED hrr_proxy_win.c)
  target_link_libraries(amdhip64_7 PRIVATE hrr_trace_writer hrr_code_object)
  set_target_properties(amdhip64_7 PROPERTIES
    OUTPUT_NAME "amdhip64_7"
    PREFIX "")
else()
  # Linux LD_PRELOAD interposer
  # Include the C++ extension to intercept hipExtModuleLaunchKernel (MIGraphX path)
  add_library(hrr_record SHARED hrr_interposer_linux.c hrr_interposer_cxx.cpp)
  target_link_libraries(hrr_record PRIVATE hrr_trace_writer hrr_code_object
                        dl pthread)
  set_target_properties(hrr_record PROPERTIES
    OUTPUT_NAME "hrr_record"
    PREFIX "lib")
endif()

# --- Tools (require HIP + shared sources in parent directory) ---
# When deployed standalone (out_of_tree/ only), the ../hrr_reader.cpp etc. won't
# exist. In that case, skip building the tools here — use HRR_BENCH_PATH to point
# at a pre-built hrr-bench for the integration test.
find_package(hip QUIET)
set(_hrr_parent_src "${CMAKE_CURRENT_SOURCE_DIR}/..")
if(hip_FOUND AND EXISTS "${_hrr_parent_src}/hrr_reader.cpp")
  # Archive reader (shared by tools)
  add_library(hrr_reader STATIC ${_hrr_parent_src}/hrr_reader.cpp)
  target_include_directories(hrr_reader PUBLIC ${_hrr_parent_src})
  target_compile_features(hrr_reader PUBLIC cxx_std_17)

  add_executable(hrr-replay ${_hrr_parent_src}/hrr_replay.cpp)
  target_link_libraries(hrr-replay PRIVATE hrr_reader hip::host)
  target_compile_features(hrr-replay PRIVATE cxx_std_17)

  add_executable(hrr-bench ${_hrr_parent_src}/hrr_bench.cpp)
  target_link_libraries(hrr-bench PRIVATE hrr_reader hip::host)
  target_compile_features(hrr-bench PRIVATE cxx_std_17)

  add_executable(hrr-info ${_hrr_parent_src}/hrr_info.cpp)
  target_link_libraries(hrr-info PRIVATE hrr_reader)
  target_compile_features(hrr-info PRIVATE cxx_std_17)

  install(TARGETS hrr-replay hrr-bench hrr-info RUNTIME DESTINATION bin)
elseif(NOT hip_FOUND)
  message(STATUS "[HRR] HIP not found - building recording library only (no tools)")
else()
  message(STATUS "[HRR] Standalone deploy (no parent sources) - building recording library only (no tools)")
endif()

# Install recording library
if(WIN32)
  install(TARGETS amdhip64_7 RUNTIME DESTINATION bin)
else()
  install(TARGETS hrr_record LIBRARY DESTINATION lib)
endif()

# ============================================================================
# MIGraphX / ONNX replay integration tests
#
# Run with: ctest --test-dir <build_dir> -L migraphx -V
#
# The test skips gracefully (exit 0) when migraphx, onnx, or the built
# binaries are not available — it only fails on actual GPU faults or
# unexpected hrr-bench errors.
#
# To point at an hrr-bench built elsewhere:
#   cmake -DHRR_BENCH_PATH=/path/to/hrr-bench ...
# ============================================================================
find_package(Python3 COMPONENTS Interpreter QUIET)

if(Python3_FOUND)
  enable_testing()

  # Prefer the locally-built hrr-bench; fall back to an explicit path.
  if(TARGET hrr-bench)
    set(_hrr_bench "$<TARGET_FILE:hrr-bench>")
  elseif(DEFINED HRR_BENCH_PATH AND EXISTS "${HRR_BENCH_PATH}")
    set(_hrr_bench "${HRR_BENCH_PATH}")
  else()
    set(_hrr_bench "hrr-bench-NOT-FOUND")
  endif()

  if(WIN32)
    set(_libhrr "$<TARGET_FILE:amdhip64_7>")
  else()
    set(_libhrr "$<TARGET_FILE:hrr_record>")
  endif()

  add_test(
    NAME hrr_migraphx_onnx_replay
    COMMAND
      ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/test_migraphx_replay.py
      --hrr-bench     ${_hrr_bench}
      --libhrr        ${_libhrr}
      --record-script ${CMAKE_CURRENT_SOURCE_DIR}/record_model.py
      --iterations    20
      --warmup        5
  )
  set_tests_properties(hrr_migraphx_onnx_replay PROPERTIES
    TIMEOUT 600
    LABELS  "hrr;migraphx;integration"
  )
endif()
