# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

# The vocabulary matches TheRock's THEROCK_SANITIZER so a standalone build and a superbuild
# are configured the same way.
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
  # are absent, because hip-tests adds none of its own in this mode and would build
  # uninstrumented.
  if(NOT (CMAKE_CXX_FLAGS_INIT MATCHES "-fsanitize=address"
          OR CMAKE_CXX_FLAGS MATCHES "-fsanitize=address"))
    message(FATAL_ERROR
      "THEROCK_SANITIZER='${THEROCK_SANITIZER}' but no -fsanitize=address reaches the C++ "
      "compile line. Hip-tests does not add host flags when THEROCK_SANITIZER is set. For a "
      "standalone build pass -DENABLE_SANITIZER=${THEROCK_SANITIZER} instead.")
  endif()
  set(ENABLE_SANITIZER "${THEROCK_SANITIZER}")
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

# Device instrumentation needs xnack+, a KFD feature, and the host-only mode has not been
# brought up anywhere else, so both modes are Linux only for now.
if(NOT ENABLE_SANITIZER STREQUAL "OFF" AND NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
  message(FATAL_ERROR
    "ENABLE_SANITIZER='${ENABLE_SANITIZER}' is not supported on '${CMAKE_SYSTEM_NAME}'. "
    "Sanitizer builds are currently Linux only.")
endif()

if(ENABLE_SANITIZER STREQUAL "ASAN" AND NOT CMAKE_HIP_COMPILER_ID MATCHES "Clang")
  message(FATAL_ERROR
    "ENABLE_SANITIZER=ASAN requires a Clang HIP compiler, found "
    "'${CMAKE_HIP_COMPILER_ID}'. Use -DENABLE_SANITIZER=HOST_ASAN to instrument only "
    "host code.")
endif()

if(NOT ENABLE_SANITIZER STREQUAL "OFF")
  # Host instrumentation. Skipped under TheRock, which already instruments the C and C++
  # compile lines through CMAKE_CXX_FLAGS_INIT.
  if(NOT THEROCK_SANITIZER)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fsanitize=address -fno-omit-frame-pointer -g")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -fno-omit-frame-pointer -g")

    # -shared-libasan is a Clang spelling, and GCC rejects it outright. GCC links the
    # shared ASan runtime by default, so omitting it there gives the same linkage.
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
      string(APPEND CMAKE_C_FLAGS " -shared-libasan")
      string(APPEND CMAKE_CXX_FLAGS " -shared-libasan")
    endif()
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

# Device instrumentation is only emitted for xnack+ targets. Adds xnack+ to the targets known
# to support it, and throws a warning for the rest.
function(hip_tests_sanitizer_resolve_offload_archs offload_arch_str_var)
  separate_arguments(_archs UNIX_COMMAND "${${offload_arch_str_var}}")
  list(TRANSFORM _archs REPLACE "^--offload-arch=(gfx942|gfx950)$" "--offload-arch=\\1:xnack+")

  set(_no_xnack ${_archs})
  list(FILTER _no_xnack EXCLUDE REGEX ":xnack\\+")
  # ASan on SPIR-V is already handled by hip_tests_sanitizer_drop_device_flags below.
  list(FILTER _no_xnack EXCLUDE REGEX "^--offload-arch=amdgcnspirv$")
  if(_no_xnack)
    list(TRANSFORM _no_xnack REPLACE "^--offload-arch=" "")
    list(JOIN _no_xnack ", " _no_xnack)
    message(WARNING
      "ENABLE_SANITIZER=ASAN instruments device code only on xnack+ offload targets, and "
      "does not work on '${_no_xnack}'. Use -DENABLE_SANITIZER=HOST_ASAN to instrument "
      "only host code.")
  endif()

  string(JOIN " " _rewritten ${_archs})
  set(${offload_arch_str_var} "${_rewritten}" PARENT_SCOPE)
endfunction()

function(hip_tests_sanitizer_drop_device_flags flags_var)
  set(_flags ${${flags_var}})
  list(FILTER _flags EXCLUDE REGEX "-fsanitize=address")
  list(FILTER _flags EXCLUDE REGEX "-fno-omit-frame-pointer")
  list(FILTER _flags EXCLUDE REGEX "-shared-libasan")
  if(ENABLE_SANITIZER STREQUAL "HOST_ASAN")
    list(FILTER _flags EXCLUDE REGEX "-Xarch_host")
  endif()

  set(${flags_var} "${_flags}" PARENT_SCOPE)
endfunction()
