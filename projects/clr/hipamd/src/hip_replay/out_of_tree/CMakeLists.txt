cmake_minimum_required(VERSION 3.21)
project(hip_replay_out_of_tree LANGUAGES C CXX)

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
  add_library(hrr_record SHARED hrr_interposer_linux.c)
  target_link_libraries(hrr_record PRIVATE hrr_trace_writer hrr_code_object
                        dl pthread)
  set_target_properties(hrr_record PROPERTIES
    OUTPUT_NAME "hrr_record"
    PREFIX "lib")
endif()

# --- Tools (require HIP) ---
find_package(hip QUIET)
if(hip_FOUND)
  # Archive reader (shared by tools)
  add_library(hrr_reader STATIC ../hrr_reader.cpp)
  target_include_directories(hrr_reader PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..)
  target_compile_features(hrr_reader PUBLIC cxx_std_17)

  add_executable(hrr-replay ../hrr_replay.cpp)
  target_link_libraries(hrr-replay PRIVATE hrr_reader hip::host hip::device)
  target_compile_features(hrr-replay PRIVATE cxx_std_17)

  add_executable(hrr-bench ../hrr_bench.cpp)
  target_link_libraries(hrr-bench PRIVATE hrr_reader hip::host hip::device)
  target_compile_features(hrr-bench PRIVATE cxx_std_17)

  add_executable(hrr-info ../hrr_info.cpp)
  target_link_libraries(hrr-info PRIVATE hrr_reader)
  target_compile_features(hrr-info PRIVATE cxx_std_17)

  install(TARGETS hrr-replay hrr-bench hrr-info RUNTIME DESTINATION bin)
else()
  message(STATUS "[HRR] HIP not found - building recording library only (no tools)")
endif()

# Install recording library
if(WIN32)
  install(TARGETS amdhip64_7 RUNTIME DESTINATION bin)
else()
  install(TARGETS hrr_record LIBRARY DESTINATION lib)
endif()
