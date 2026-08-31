# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

# Included at directory scope so rocclr, hipamd and opencl inherit the policy.

include_guard()

option(CLR_ENABLE_HARDENING
  "Build CLR with compiler and linker hardening mitigations (Linux only)" ON)

if(NOT CLR_ENABLE_HARDENING)
  return()
endif()

# GCC/Clang driver and ELF linker syntax, which MSVC and clang-cl reject.
if(NOT UNIX)
  return()
endif()

# FORTIFY_SOURCE reports false positives under sanitizers.
if(ADDRESS_SANITIZER OR THEROCK_SANITIZER OR ENABLE_ADDRESS_SANITIZER)
  message(STATUS "CLR hardening: disabled for sanitizer build")
  return()
endif()

include(CheckCXXCompilerFlag)
include(CheckCXXSourceCompiles)

# gcc and clang reject -fstack-clash-protection outright where unsupported.
foreach(_clr_hardening_flag -fstack-protector-strong -fstack-clash-protection)
  string(MAKE_C_IDENTIFIER "CLR_HAVE${_clr_hardening_flag}" _clr_hardening_var)
  check_cxx_compiler_flag("${_clr_hardening_flag}" ${_clr_hardening_var})
  if(${_clr_hardening_var})
    add_compile_options("${_clr_hardening_flag}")
  endif()
endforeach()

unset(_clr_hardening_var)

# Not probed: linkers warn and ignore an unknown -z keyword, so a probe passes either way.
add_link_options("-Wl,-z,relro,-z,now" "-Wl,-z,noexecstack")

# Ubuntu 24.04 and others default to level 3; pinning the report's 2 would remove
# fortification. Probed under -Werror so level-2-only toolchains fall back.
set(CMAKE_REQUIRED_FLAGS "-Werror -O2 -U_FORTIFY_SOURCE -D_FORTIFY_SOURCE=3")
check_cxx_source_compiles("#include <string.h>\nint main(void) { return 0; }"
  CLR_HAVE_fortify_source_3)
unset(CMAKE_REQUIRED_FLAGS)
if(CLR_HAVE_fortify_source_3)
  set(_clr_fortify_level 3)
else()
  set(_clr_fortify_level 2)
endif()

# Needs an optimizing build, and hipamd compiles with -Werror. Undefine first so
# toolchains that predefine it do not warn.
add_compile_options(
  "$<$<NOT:$<CONFIG:Debug>>:-U_FORTIFY_SOURCE>"
  "$<$<NOT:$<CONFIG:Debug>>:-D_FORTIFY_SOURCE=${_clr_fortify_level}>")

message(STATUS "CLR hardening: enabled (_FORTIFY_SOURCE=${_clr_fortify_level})")

unset(_clr_fortify_level)
