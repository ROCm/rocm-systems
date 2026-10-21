# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

# Post-build check invoked as:
#   cmake -DFILE=<shared object> -P check_asan_instrumentation.cmake
# A link-only -fsanitize=address DSO has weak __asan_report_* symbols and no
# __asan_init. Instrumented objects reference __asan_init.

if(NOT FILE)
  message(FATAL_ERROR "FILE must be the shared object to check")
endif()

find_program(NM_TOOL NAMES nm llvm-nm REQUIRED)
execute_process(
  COMMAND ${NM_TOOL} -D ${FILE}
  RESULT_VARIABLE _nm_rc
  OUTPUT_VARIABLE _nm_out
  ERROR_VARIABLE _nm_err)
if(NOT _nm_rc EQUAL 0)
  message(FATAL_ERROR "nm -D ${FILE} failed (${_nm_rc}): ${_nm_err}")
endif()
if(NOT _nm_out MATCHES "(^|[ \t])__asan_init([ \t]|$)")
  message(FATAL_ERROR
    "${FILE} is not instrumented with ASan (dynamic symbol __asan_init is missing). "
    "The library was linked with the ASan runtime but its objects were not compiled "
    "with -fsanitize=address.")
endif()
