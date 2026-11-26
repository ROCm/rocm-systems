# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

# Defines `clr::warnings`, the single source of truth for strict-warning flags.
# First-party CLR targets opt in with:
#
#   target_link_libraries(<target> PRIVATE clr::warnings)

include_guard(GLOBAL)

option(CLR_ENABLE_WERROR "Build first-party CLR targets with -Wall -Werror" OFF)

add_library(clr_warnings INTERFACE)
add_library(clr::warnings ALIAS clr_warnings)

if(NOT CLR_ENABLE_WERROR)
  return()
endif()

# Only supporting Linux for now
if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux"
    OR NOT CMAKE_CXX_COMPILER_ID MATCHES "^(GNU|Clang)$")
  message(WARNING
    "CLR_ENABLE_WERROR is supported for GCC/Clang on Linux only; ignoring it "
    "for ${CMAKE_CXX_COMPILER_ID} on ${CMAKE_SYSTEM_NAME}.")
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
