# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

# Defines `clr::warnings`, the single source of truth for strict-warning flags.
# First-party CLR targets opt in with:
#
#   target_link_libraries(<target> PRIVATE clr::warnings)

include_guard(GLOBAL)

option(CLR_ENABLE_WERROR
  "Build first-party CLR targets with -Wall -Werror (/W4 /WX on Windows)" OFF)

add_library(clr_warnings INTERFACE)
add_library(clr::warnings ALIAS clr_warnings)

if(NOT CLR_ENABLE_WERROR)
  return()
endif()

# Windows: MSVC's cl, or clang-cl (any Clang driving the MSVC frontend).
# /W4 is the strict-but-sane level; /Wall additionally turns on a large body of
# off-by-default noise from the system headers themselves, so it is not used.
if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC"
    OR CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC")
  # CMake seeds CMAKE_CXX_FLAGS with /W3. CLR's top-level CMakeLists.txt already
  # strips it, and in any case target options are appended last, so /W4 wins.
  target_compile_options(clr_warnings INTERFACE /W4 /WX)

  # C4714: "function marked as __forceinline not inlined". Whether the compiler
  # honours __forceinline is its own decision and not something the source can
  # act on, so this reports a non-defect. It also dominates the Windows warning
  # count: every entry of the HIP dispatch table in hip_table_interface.cpp
  # trips it, for 542 of the 1517 /W4 sites in CLR. No GCC/Clang analogue.
  target_compile_options(clr_warnings INTERFACE /wd4714)

  if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    # clang-cl understands the GNU-style spellings, so keep the Linux policy of
    # letting deprecations warn without failing the build.
    target_compile_options(clr_warnings INTERFACE
      -Wno-error=deprecated-declarations)
  endif()

  message(STATUS "CLR warnings: /W4 /WX enabled")
  return()
endif()

# Everything else: GCC/Clang on Linux.
if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux"
    OR NOT CMAKE_CXX_COMPILER_ID MATCHES "^(GNU|Clang)$")
  message(WARNING
    "CLR_ENABLE_WERROR is supported for GCC/Clang on Linux and MSVC/clang-cl "
    "on Windows only; ignoring it for ${CMAKE_CXX_COMPILER_ID} on "
    "${CMAKE_SYSTEM_NAME}.")
  return()
endif()

# Deprecations are a warning on purpose: CLR must keep compiling against ROCm
# headers that deprecate an API ahead of CLR dropping its uses of it.
target_compile_options(clr_warnings INTERFACE
  -Wall -Werror -Wno-error=deprecated-declarations)

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 12)
  target_compile_options(clr_warnings INTERFACE -Wno-attributes)
endif()

message(STATUS "CLR warnings: -Wall -Werror enabled")
