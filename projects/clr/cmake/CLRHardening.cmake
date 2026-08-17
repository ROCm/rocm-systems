# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

# Binary hardening mitigations for the CLR host runtime.
#
# Included at directory scope from the top-level CMakeLists.txt so that rocclr,
# hipamd and opencl inherit them. Device code is produced by explicit clang
# invocations in add_custom_command(), so these host flags do not reach offload
# compilation.

include_guard()

option(CLR_ENABLE_HARDENING
  "Build CLR with compiler and linker hardening mitigations (Linux only)" ON)

if(NOT CLR_ENABLE_HARDENING)
  return()
endif()

# Everything below is GCC/Clang driver and ELF linker syntax. UNIX is the proxy
# for that: on Windows CLR builds with MSVC or clang-cl, which reject it.
if(NOT UNIX)
  return()
endif()

# Sanitizers replace the allocator and stack layout these mitigations rely on,
# and report false positives when combined with FORTIFY_SOURCE.
# ENABLE_ADDRESS_SANITIZER is the opencl subtree's own switch for the same
# thing. Any truthy value disables, so that a sanitizer added later does not
# silently inherit a conflicting configuration.
if(ADDRESS_SANITIZER OR THEROCK_SANITIZER OR ENABLE_ADDRESS_SANITIZER)
  message(STATUS "CLR hardening: disabled for sanitizer build")
  return()
endif()

include(CheckCXXCompilerFlag)
include(CheckCXXSourceCompiles)

# Every flag is probed before use. Not all targets and toolchain versions
# implement all of them, and an unsupported flag is diagnosed on every
# translation unit.
foreach(_clr_hardening_flag -fstack-protector-strong -fstack-clash-protection)
  string(MAKE_C_IDENTIFIER "CLR_HAVE${_clr_hardening_flag}" _clr_hardening_var)
  check_cxx_compiler_flag("${_clr_hardening_flag}" ${_clr_hardening_var})
  if(${_clr_hardening_var})
    add_compile_options("${_clr_hardening_flag}")
  endif()
endforeach()

foreach(_clr_hardening_flag "-Wl,-z,relro,-z,now" "-Wl,-z,noexecstack")
  string(MAKE_C_IDENTIFIER "CLR_HAVE${_clr_hardening_flag}" _clr_hardening_var)
  # check_linker_flag() would be the natural fit but needs CMake 3.18, above
  # the 3.16.8 floor this project declares.
  set(CMAKE_REQUIRED_LINK_OPTIONS "${_clr_hardening_flag}")
  check_cxx_source_compiles("int main(void) { return 0; }" ${_clr_hardening_var})
  unset(CMAKE_REQUIRED_LINK_OPTIONS)
  if(${_clr_hardening_var})
    add_link_options("${_clr_hardening_flag}")
  endif()
endforeach()

unset(_clr_hardening_var)

# Distributions such as Ubuntu 24.04 already default to level 3, so pinning the
# level 2 named in the report would remove fortification rather than add it.
# Probe the real construct under -Werror so that a compiler or C library
# implementing only level 2 falls back instead of warning on every file.
set(CMAKE_REQUIRED_FLAGS "-Werror -O2 -U_FORTIFY_SOURCE -D_FORTIFY_SOURCE=3")
check_cxx_source_compiles("#include <string.h>\nint main(void) { return 0; }"
  CLR_HAVE_fortify_source_3)
unset(CMAKE_REQUIRED_FLAGS)
if(CLR_HAVE_fortify_source_3)
  set(_clr_fortify_level 3)
else()
  set(_clr_fortify_level 2)
endif()

# FORTIFY_SOURCE needs an optimizing build; at -O0 the C library warns, and
# hipamd compiles with -Werror. Undefining first keeps toolchains that already
# predefine it from warning about the redefinition.
add_compile_options(
  "$<$<NOT:$<CONFIG:Debug>>:-U_FORTIFY_SOURCE>"
  "$<$<NOT:$<CONFIG:Debug>>:-D_FORTIFY_SOURCE=${_clr_fortify_level}>")

# CMake maps this to -fPIC for libraries and -fPIE/-pie for executables, which
# is why the property is set instead of passing a literal flag that would apply
# the wrong one to shared libraries.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

message(STATUS "CLR hardening: enabled (_FORTIFY_SOURCE=${_clr_fortify_level})")

unset(_clr_fortify_level)
