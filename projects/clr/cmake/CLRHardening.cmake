# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

# Included at directory scope so rocclr, hipamd and opencl inherit the policy.

include_guard()

option(CLR_ENABLE_HARDENING "Build CLR with full RELRO (Linux only)" ON)

if(NOT CLR_ENABLE_HARDENING)
  return()
endif()

# ELF linker syntax: MSVC and clang-cl reject it, and -z is not valid for Mach-O.
if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
  return()
endif()

# Resolve the GOT at load time and map it read-only, so a stray write cannot
# repoint an entry. Not probed: linkers warn about and ignore an unknown -z
# keyword, so a link probe would pass either way.
add_link_options("-Wl,-z,relro,-z,now")

message(STATUS "CLR hardening: full RELRO enabled")
