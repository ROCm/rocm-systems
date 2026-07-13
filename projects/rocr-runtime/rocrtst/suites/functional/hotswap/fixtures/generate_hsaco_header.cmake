# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

if(NOT DEFINED INPUT)
  message(FATAL_ERROR "INPUT is required")
endif()
if(NOT DEFINED OUTPUT)
  message(FATAL_ERROR "OUTPUT is required")
endif()

file(READ "${INPUT}" _hsaco_hex HEX)
string(REGEX REPLACE "(..)" "0x\\1," _hsaco_bytes "${_hsaco_hex}")
file(WRITE "${OUTPUT}"
  "// Generated from ${INPUT}. Do not edit.\n"
  "static const unsigned char kGfx1250MinCo[] = {${_hsaco_bytes}};\n")
