# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

foreach(kind HEADER CAPTURE PLAYBACK)
  set(generated "${GENERATED_${kind}}")
  set(expected "${EXPECTED_${kind}}")

  if(NOT EXISTS "${generated}")
    message(FATAL_ERROR "HRR code generation did not produce ${generated}")
  endif()
  if(NOT EXISTS "${expected}")
    message(FATAL_ERROR "HRR checked-in generated file is missing: ${expected}")
  endif()

  execute_process(
    COMMAND "${CMAKE_COMMAND}" -E compare_files "${generated}" "${expected}"
    RESULT_VARIABLE compare_result)
  if(NOT compare_result EQUAL 0)
    message(FATAL_ERROR
      "HRR generated ${kind} file is stale: ${expected}\n"
      "Run from the repository root to update checked-in generated files:\n"
      "  python3 projects/hrr/tools/gen_hrr_api_args.py --check-hrr-coverage")
  endif()
endforeach()
