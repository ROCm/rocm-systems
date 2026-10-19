# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

# Fails when a HIP entry point in SOURCE (hip_table_interface.cpp) reads a dispatch
# table slot directly instead of through HIP_LOAD_DISPATCH_TABLE_FN or
# HIP_LOAD_COMPILER_TABLE_FN, which load the slot atomically.
file(READ "${SOURCE}" text)
# clang-format can break a long read before "->", so allow whitespace there.
string(REGEX MATCHALL "Get[A-Za-z]*DispatchTable\\(\\)[ \t\r\n]*->[A-Za-z0-9_]+_fn" plain "${text}")
string(REGEX MATCHALL "HIP_LOAD_(DISPATCH|COMPILER)_TABLE_FN\\([A-Za-z0-9_]+_fn\\)" loaded "${text}")
list(LENGTH plain n_plain)
list(LENGTH loaded n_loaded)
if(n_plain GREATER 0)
  list(GET plain 0 first)
  message(FATAL_ERROR "${n_plain} entry points read their dispatch slot directly, e.g. ${first}")
endif()
if(n_loaded EQUAL 0)
  message(FATAL_ERROR "no HIP_LOAD_*_TABLE_FN call sites found in ${SOURCE}")
endif()
# The call sites are only as good as the macros: each one must load through
# hip::LoadDispatchSlot, not read the slot with a plain ->fn.
foreach(table DISPATCH COMPILER)
  string(REGEX MATCH "#define HIP_LOAD_${table}_TABLE_FN\\(([^\n]*\\\\\n)*[^\n]*" def "${text}")
  if(NOT def MATCHES "hip::LoadDispatchSlot\\(")
    message(FATAL_ERROR "HIP_LOAD_${table}_TABLE_FN does not load through hip::LoadDispatchSlot: '${def}'")
  endif()
endforeach()
message(STATUS "${n_loaded} entry points load their dispatch slot atomically")
