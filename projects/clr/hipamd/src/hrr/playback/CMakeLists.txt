# HRR Playback Tools
# Built as part of the main HIP runtime CMake tree.
# HIP_ROOT / HIP_INCLUDE_DIR / HIP_LIBRARY are already resolved by the parent.

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# hrr_api_args.h and capture headers are one directory up (hipamd/src/hrr/).
set(HRR_DIR ${CMAKE_CURRENT_SOURCE_DIR}/..)

if(NOT DEFINED HIP_INCLUDE_DIR)
  set(HIP_INCLUDE_DIR "${HIP_ROOT}/include")
endif()

# Always built in-tree alongside amdhip64 — link against the CMake target directly.
set(HIP_LIBRARY amdhip64)

# ---------------------------------------------------------------------------
# hrr_reader — archive loader (STATIC)
# ---------------------------------------------------------------------------
add_library(hrr_reader STATIC hrr_reader.cpp)
target_include_directories(hrr_reader PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${HRR_DIR}
  ${HIP_INCLUDE_DIR}
)
target_compile_definitions(hrr_reader PRIVATE
  HRR_API_ARGS_IMPLEMENTATION
  $<$<BOOL:${WIN32}>:NOMINMAX WIN32_LEAN_AND_MEAN>
)

# ---------------------------------------------------------------------------
# hrr_playback_lib — manual shims + generated dispatch table (STATIC)
# ---------------------------------------------------------------------------
add_library(hrr_playback_lib STATIC
  hip_playback.cpp
  hip_playback_generated.cpp
)
target_include_directories(hrr_playback_lib PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${HRR_DIR}
  ${HIP_INCLUDE_DIR}
)
target_compile_definitions(hrr_playback_lib PRIVATE
  __HIP_PLATFORM_AMD__
  $<$<BOOL:${WIN32}>:NOMINMAX WIN32_LEAN_AND_MEAN>
)
target_link_libraries(hrr_playback_lib PUBLIC hrr_reader ${HIP_LIBRARY})

# ---------------------------------------------------------------------------
# hrr-playback — full replay tool
# ---------------------------------------------------------------------------
add_executable(hrr-playback hrr_playback.cpp)
target_link_libraries(hrr-playback PRIVATE hrr_playback_lib)
target_include_directories(hrr-playback PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${HRR_DIR}
  ${HIP_INCLUDE_DIR}
)
target_compile_definitions(hrr-playback PRIVATE
  __HIP_PLATFORM_AMD__
  $<$<BOOL:${WIN32}>:NOMINMAX WIN32_LEAN_AND_MEAN>
)

install(TARGETS hrr-playback RUNTIME DESTINATION bin)

# ---------------------------------------------------------------------------
# test_stride1 — standalone kernel launch test (diagnostic only, not installed)
# ---------------------------------------------------------------------------
add_executable(test_stride1 test_stride1.cpp)
target_include_directories(test_stride1 PRIVATE
  ${HIP_INCLUDE_DIR}
)
target_compile_definitions(test_stride1 PRIVATE
  __HIP_PLATFORM_AMD__
  $<$<BOOL:${WIN32}>:NOMINMAX WIN32_LEAN_AND_MEAN>
)
target_link_libraries(test_stride1 PRIVATE ${HIP_LIBRARY})

# ---------------------------------------------------------------------------
# test_stride2_then_stride1 — sequence test (diagnostic only, not installed)
# ---------------------------------------------------------------------------
add_executable(test_stride2_then_stride1 test_stride2_then_stride1.cpp)
target_include_directories(test_stride2_then_stride1 PRIVATE ${HIP_INCLUDE_DIR})
target_compile_definitions(test_stride2_then_stride1 PRIVATE
  __HIP_PLATFORM_AMD__
  $<$<BOOL:${WIN32}>:NOMINMAX WIN32_LEAN_AND_MEAN>
)
target_link_libraries(test_stride2_then_stride1 PRIVATE ${HIP_LIBRARY})

# ---------------------------------------------------------------------------
# time_stride1 — wall-clock / GPU-event timing at various grid sizes (diagnostic)
# ---------------------------------------------------------------------------
add_executable(time_stride1 time_stride1.cpp)
target_include_directories(time_stride1 PRIVATE ${HIP_INCLUDE_DIR})
target_compile_definitions(time_stride1 PRIVATE
  __HIP_PLATFORM_AMD__
  $<$<BOOL:${WIN32}>:NOMINMAX WIN32_LEAN_AND_MEAN>
)
target_link_libraries(time_stride1 PRIVATE ${HIP_LIBRARY})
