# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

# The vocabulary matches TheRock's THEROCK_SANITIZER so a standalone build and a superbuild
# are configured the same way.
set(ENABLE_SANITIZER
    "OFF"
    CACHE STRING "Sanitizer mode: OFF, ASAN (host and device), or HOST_ASAN (host only)")
set_property(CACHE ENABLE_SANITIZER PROPERTY STRINGS OFF ASAN HOST_ASAN)

# TheRock injects THEROCK_SANITIZER variable. It takes precedence over ENABLE_SANITIZER.
if(THEROCK_SANITIZER STREQUAL "ASAN" OR THEROCK_SANITIZER STREQUAL "HOST_ASAN")
  # TheRock puts the host -fsanitize= flags on the compile line itself. Fail if they are
  # absent, because hip-tests adds none of its own in this mode and would build uninstrumented.
  if(NOT (CMAKE_CXX_FLAGS_INIT MATCHES "-fsanitize=" OR CMAKE_CXX_FLAGS MATCHES "-fsanitize="))
    message(FATAL_ERROR
      "THEROCK_SANITIZER='${THEROCK_SANITIZER}' but no -fsanitize= reaches the C++ compile "
      "line. Hip-tests does not add host flags when THEROCK_SANITIZER is set. For a standalone "
      "build pass -DENABLE_SANITIZER=${THEROCK_SANITIZER} instead.")
  endif()
  set(ENABLE_SANITIZER "${THEROCK_SANITIZER}"
      CACHE STRING "Sanitizer mode (driven by THEROCK_SANITIZER)" FORCE)
endif()

set(_HIP_TESTS_SANITIZER_VALID OFF ASAN HOST_ASAN)
if(NOT ENABLE_SANITIZER IN_LIST _HIP_TESTS_SANITIZER_VALID)
  list(JOIN _HIP_TESTS_SANITIZER_VALID ", " _HIP_TESTS_SANITIZER_VALID)
  message(FATAL_ERROR
    "ENABLE_SANITIZER='${ENABLE_SANITIZER}' is not one of: ${_HIP_TESTS_SANITIZER_VALID}")
endif()
unset(_HIP_TESTS_SANITIZER_VALID)

# ENABLE_ADDRESS_SANITIZER was the previous switch and could not express a host-only build.
if(ENABLE_ADDRESS_SANITIZER)
  message(FATAL_ERROR
    "ENABLE_ADDRESS_SANITIZER has been replaced by ENABLE_SANITIZER. Pass "
    "-DENABLE_SANITIZER=ASAN for the previous behaviour, or -DENABLE_SANITIZER=HOST_ASAN "
    "to instrument only host code.")
endif()

# Device instrumentation needs xnack+, a KFD feature.
if(ENABLE_SANITIZER STREQUAL "ASAN" AND NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
  message(FATAL_ERROR
    "ENABLE_SANITIZER=ASAN requires Linux, found '${CMAKE_SYSTEM_NAME}'. "
    "Use -DENABLE_SANITIZER=HOST_ASAN to instrument only host code.")
endif()

if(ENABLE_SANITIZER STREQUAL "ASAN" AND NOT CMAKE_HIP_COMPILER_ID MATCHES "Clang")
  message(FATAL_ERROR
    "ENABLE_SANITIZER=ASAN requires a Clang HIP compiler, found "
    "'${CMAKE_HIP_COMPILER_ID}'. Use -DENABLE_SANITIZER=HOST_ASAN to instrument only "
    "host code.")
endif()

# Device instrumentation only applies on xnack+ targets; Rewrites GPU_TARGETS 
# to use xnack+ when possible.
if(ENABLE_SANITIZER STREQUAL "ASAN" AND DEFINED GPU_TARGETS)
  list(TRANSFORM GPU_TARGETS REPLACE "^(gfx942|gfx950)$" "\\1:xnack+")
  message(STATUS "Device instrumentation targets: ${GPU_TARGETS}")
endif()

if(NOT ENABLE_SANITIZER STREQUAL "OFF")
  # Host instrumentation. Skipped under TheRock, which already instruments the C and C++
  # compile lines through CMAKE_CXX_FLAGS_INIT.
  if(NOT THEROCK_SANITIZER)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fsanitize=address -fno-omit-frame-pointer -shared-libasan -g")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -fno-omit-frame-pointer -shared-libasan -g")
  endif()

  # The HIP flags are always set here: these sources compile as the HIP language, which
  # TheRock does not instrument.
  if(ENABLE_SANITIZER STREQUAL "ASAN")
    set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -fsanitize=address -fno-omit-frame-pointer -shared-libasan -g")
    add_compile_definitions(ENABLE_DEVICE_ADDRESS_SANITIZER)
  else()
    set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -Xarch_host -fsanitize=address -Xarch_host -fno-omit-frame-pointer -shared-libasan -g")
  endif()

  add_compile_definitions(ENABLE_ADDRESS_SANITIZER)
  message(STATUS "Building catch tests with Address Sanitizer options (${ENABLE_SANITIZER})")
endif()
