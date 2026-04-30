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
if(NOT DEFINED HIP_LIBRARY)
  find_library(HIP_LIBRARY amdhip64 HINTS "${HIP_ROOT}/lib" REQUIRED)
endif()

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
