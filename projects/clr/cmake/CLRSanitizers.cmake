# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

# The vocabulary matches TheRock's THEROCK_SANITIZER so a standalone build and a superbuild
# are configured the same way. HOST_ASAN is a distinct value because CLR compiles and JITs
# device code.
set(ENABLE_SANITIZER
    "OFF"
    CACHE STRING "Sanitizer mode: OFF, ASAN (host and device), or HOST_ASAN (host only)")
set_property(CACHE ENABLE_SANITIZER PROPERTY STRINGS OFF ASAN HOST_ASAN)

if(THEROCK_SANITIZER AND NOT ENABLE_SANITIZER STREQUAL "OFF")
  message(FATAL_ERROR
    "THEROCK_SANITIZER='${THEROCK_SANITIZER}' already selects the sanitizer for this build, "
    "so ENABLE_SANITIZER must be left at OFF, found '${ENABLE_SANITIZER}'. Pass "
    "-DENABLE_SANITIZER=OFF, or configure without THEROCK_SANITIZER for a standalone build.")
endif()

# TheRock injects THEROCK_SANITIZER variable. It takes precedence over ENABLE_SANITIZER.
if(THEROCK_SANITIZER STREQUAL "ASAN" OR THEROCK_SANITIZER STREQUAL "HOST_ASAN")
  # TheRock puts the host -fsanitize=address flags on the compile line itself. Fail if they
  # are absent, because CLR adds none of its own in this mode and would build uninstrumented.
  if(NOT (CMAKE_CXX_FLAGS_INIT MATCHES "-fsanitize=address"
          OR CMAKE_CXX_FLAGS MATCHES "-fsanitize=address"))
    message(FATAL_ERROR
      "THEROCK_SANITIZER='${THEROCK_SANITIZER}' but no -fsanitize=address reaches the C++ "
      "compile line. CLR does not add host flags when THEROCK_SANITIZER is set. For a "
      "standalone build pass -DENABLE_SANITIZER=${THEROCK_SANITIZER} instead.")
  endif()
  set(ENABLE_SANITIZER "${THEROCK_SANITIZER}")
endif()

set(_clr_sanitizer_valid OFF ASAN HOST_ASAN)
if(NOT ENABLE_SANITIZER IN_LIST _clr_sanitizer_valid)
  list(JOIN _clr_sanitizer_valid ", " _clr_sanitizer_valid)
  message(FATAL_ERROR
    "ENABLE_SANITIZER='${ENABLE_SANITIZER}' is not one of: ${_clr_sanitizer_valid}")
endif()
unset(_clr_sanitizer_valid)

# ADDRESS_SANITIZER (hipamd, amdocl) and ENABLE_ADDRESS_SANITIZER (opencl tests) were the
# previous switches and neither could express a host-only build.
if(ADDRESS_SANITIZER OR ENABLE_ADDRESS_SANITIZER)
  message(FATAL_ERROR
    "ADDRESS_SANITIZER and ENABLE_ADDRESS_SANITIZER have been replaced by ENABLE_SANITIZER. "
    "Pass -DENABLE_SANITIZER=ASAN for the previous behaviour, or -DENABLE_SANITIZER=HOST_ASAN "
    "to instrument only host code.")
endif()

# Device instrumentation depends on xnack+, a KFD feature, and the host-only mode has not
# been brought up anywhere else, so both modes are Linux only for now.
if(NOT ENABLE_SANITIZER STREQUAL "OFF" AND NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
  message(FATAL_ERROR
    "ENABLE_SANITIZER='${ENABLE_SANITIZER}' is not supported on '${CMAKE_SYSTEM_NAME}'. "
    "Sanitizer builds are currently Linux only.")
endif()

if(ENABLE_SANITIZER STREQUAL "ASAN" AND NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  message(FATAL_ERROR
    "ENABLE_SANITIZER=ASAN requires a Clang compiler, found '${CMAKE_CXX_COMPILER_ID}'. "
    "Use -DENABLE_SANITIZER=HOST_ASAN to instrument only host code.")
endif()

if(ENABLE_SANITIZER STREQUAL "ASAN")
  add_compile_definitions(DEVICE_ADDRESS_SANITIZER=1)
else()
  add_compile_definitions(DEVICE_ADDRESS_SANITIZER=0)
endif()

# Host instrumentation. Skipped under TheRock, which already instruments the C and C++
# compile lines through CMAKE_CXX_FLAGS_INIT.
if(NOT ENABLE_SANITIZER STREQUAL "OFF" AND NOT THEROCK_SANITIZER)
  set(_linker_flags "-fsanitize=address")
  set(_compiler_flags "-fno-omit-frame-pointer -fsanitize=address")
  if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    if(BUILD_SHARED_LIBS)
      string(APPEND _compiler_flags " -shared-libsan")
      string(APPEND _linker_flags " -shared-libsan")
    else()
      string(APPEND _linker_flags " -static-libsan")
    endif()
  endif()

  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${_compiler_flags}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${_compiler_flags}")
  set(CMAKE_EXE_LINKER_FLAGS
      "${CMAKE_EXE_LINKER_FLAGS} ${_linker_flags} -Wl,--build-id=sha1")
  set(CMAKE_SHARED_LINKER_FLAGS
      "${CMAKE_SHARED_LINKER_FLAGS} ${_linker_flags} -Wl,--build-id=sha1")
  unset(_compiler_flags)
  unset(_linker_flags)
  message(STATUS "CLR sanitizer: ${ENABLE_SANITIZER} (host flags applied by CLR)")
elseif(NOT ENABLE_SANITIZER STREQUAL "OFF")
  message(STATUS "CLR sanitizer: ${ENABLE_SANITIZER} (host flags from the enclosing build)")
endif()
