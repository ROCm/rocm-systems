# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

# HRR Playback Tools
# Built as part of the standalone projects/hrr CMake tree. HIP runtime linkage
# comes from the public HIP CMake package (hip::host), not from CLR internals.

# C++20 on Windows (MSVC requires it for some std features), C++17 on Linux.
if(WIN32)
  set(HRR_CXX_STANDARD 20)
else()
  set(HRR_CXX_STANDARD 17)
endif()

# ---------------------------------------------------------------------------
# hrr_reader — archive loader (STATIC)
#
# CPU-only: depends solely on the header-only wire format (hrr::api_args) and the
# C++ standard library. It is defined unconditionally so the unit tests can link
# it even when the HIP-dependent playback tool is not built
# (HRR_BUILD_PLAYBACK=OFF); it must not gain a HIP dependency.
# ---------------------------------------------------------------------------
add_library(hrr_reader STATIC hrr_reader.cpp)
add_library(hrr::reader ALIAS hrr_reader)
set_target_properties(hrr_reader PROPERTIES
  CXX_STANDARD ${HRR_CXX_STANDARD}
  CXX_STANDARD_REQUIRED ON)
target_link_libraries(hrr_reader PUBLIC hrr::api_args)
target_include_directories(hrr_reader PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<INSTALL_INTERFACE:include/hrr/playback>
)
target_compile_definitions(hrr_reader PRIVATE
  HRR_API_ARGS_IMPLEMENTATION
  $<$<BOOL:${WIN32}>:NOMINMAX WIN32_LEAN_AND_MEAN>
)

# ---------------------------------------------------------------------------
# hrr-playback — full replay tool (requires HIP)
# ---------------------------------------------------------------------------
if(HRR_BUILD_PLAYBACK)
  find_package(Threads REQUIRED)
  find_package(hip CONFIG REQUIRED)

  add_executable(hrr-playback
    hrr_playback.cpp
    hip_playback.cpp
    hip_playback_generated.cpp
  )
  set_target_properties(hrr-playback PROPERTIES
    CXX_STANDARD ${HRR_CXX_STANDARD}
    CXX_STANDARD_REQUIRED ON)
  target_include_directories(hrr-playback PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
  )
  target_compile_definitions(hrr-playback PRIVATE
    __HIP_PLATFORM_AMD__
    $<$<BOOL:${WIN32}>:NOMINMAX WIN32_LEAN_AND_MEAN>
  )
  target_link_libraries(hrr-playback PRIVATE hrr::reader hip::host Threads::Threads)

  install(TARGETS hrr-playback RUNTIME DESTINATION bin)
  install(FILES hrr_reader.h hip_playback.h DESTINATION include/hrr/playback)
endif()
