# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

include_guard(GLOBAL)

set(_hrr_codegen_module_dir "${CMAKE_CURRENT_LIST_DIR}")

function(hrr_add_codegen_check)
  cmake_parse_arguments(ARG
    ""
    "TARGET;HRR_SOURCE_DIR;CLR_SOURCE_DIR;HIP_SOURCE_DIR;OUTPUT_DIR"
    ""
    ${ARGN})

  foreach(required TARGET HRR_SOURCE_DIR CLR_SOURCE_DIR HIP_SOURCE_DIR OUTPUT_DIR)
    if(NOT ARG_${required})
      message(FATAL_ERROR "hrr_add_codegen_check requires ${required}")
    endif()
  endforeach()

  find_package(Python3 COMPONENTS Interpreter REQUIRED)

  set(generator "${ARG_HRR_SOURCE_DIR}/tools/gen_hrr_api_args.py")
  set(api_trace "${ARG_CLR_SOURCE_DIR}/hipamd/include/hip/amd_detail/hip_api_trace.hpp")
  set(public_header "${ARG_HIP_SOURCE_DIR}/include/hip/hip_runtime_api.h")
  set(header "${ARG_OUTPUT_DIR}/include/hrr/hrr_api_args.h")
  set(capture "${ARG_OUTPUT_DIR}/hip_capture_generated.cpp")
  set(playback "${ARG_OUTPUT_DIR}/hip_playback_generated.cpp")
  set(expected_header "${ARG_HRR_SOURCE_DIR}/include/hrr/hrr_api_args.h")
  set(expected_capture "${ARG_CLR_SOURCE_DIR}/hipamd/src/hrr/hip_capture_generated.cpp")
  set(expected_playback "${ARG_HRR_SOURCE_DIR}/playback/hip_playback_generated.cpp")

  add_custom_command(
    OUTPUT "${header}" "${capture}" "${playback}"
    COMMAND ${Python3_EXECUTABLE} "${generator}"
      --input "${api_trace}"
      --public-header "${public_header}"
      --output-header "${header}"
      --output-capture "${capture}"
      --output-playback "${playback}"
      --check-hrr-coverage
    COMMAND ${CMAKE_COMMAND}
      "-DGENERATED_HEADER=${header}"
      "-DEXPECTED_HEADER=${expected_header}"
      "-DGENERATED_CAPTURE=${capture}"
      "-DEXPECTED_CAPTURE=${expected_capture}"
      "-DGENERATED_PLAYBACK=${playback}"
      "-DEXPECTED_PLAYBACK=${expected_playback}"
      -P "${_hrr_codegen_module_dir}/HrrCodegenVerify.cmake"
    DEPENDS
      "${generator}"
      "${api_trace}"
      "${public_header}"
      "${_hrr_codegen_module_dir}/HrrCodegenVerify.cmake"
      "${expected_header}"
      "${expected_capture}"
      "${expected_playback}"
    COMMENT "Generating and verifying HRR API sources"
    VERBATIM)
  set_source_files_properties("${header}" "${capture}" "${playback}" PROPERTIES GENERATED TRUE)
  add_custom_target(${ARG_TARGET} DEPENDS "${header}" "${capture}" "${playback}")
endfunction()
