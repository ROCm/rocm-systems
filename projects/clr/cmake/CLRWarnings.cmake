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

if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC"
    OR CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC")
  target_compile_options(clr_warnings INTERFACE /W4 /WX)

  # Ignore C4714: "function marked as __forceinline not inlined". Whether the compiler
  # honours __forceinline is its own decision, so this can be ignored.
  target_compile_options(clr_warnings INTERFACE /wd4714)

  # Ignore C4201: "nonstandard extension used: nameless struct/union". This
  # is required for our vector implementation.
  target_compile_options(clr_warnings INTERFACE /wd4201)

  # Ignore C4324: "structure was padded due to alignment specifier". Padding is
  # fine and not a problem
  target_compile_options(clr_warnings INTERFACE /wd4324)

  # Ingore C4702: "unreachable code". Every site is a defensive trailing return after
  # a construct MSVC can prove already returns on all paths.
  target_compile_options(clr_warnings INTERFACE /wd4702)

  # Ignore C4200: "nonstandard extension used: zero-sized array in struct/union".
  # rocglinterop.hpp's metadata_amd_ci_vi_t ends in a flexible array member
  # overlaid on a hardware descriptor.
  target_compile_options(clr_warnings INTERFACE /wd4200)

  if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
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
