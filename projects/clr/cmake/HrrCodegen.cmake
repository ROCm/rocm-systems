# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

include_guard(GLOBAL)

function(hrr_add_codegen)
  cmake_parse_arguments(ARG
    "CAPTURE;PLAYBACK"
    "TARGET;HRR_SOURCE_DIR;CLR_SOURCE_DIR;HIP_SOURCE_DIR;OUTPUT_DIR"
    ""
    ${ARGN})

  foreach(required TARGET HRR_SOURCE_DIR CLR_SOURCE_DIR HIP_SOURCE_DIR OUTPUT_DIR)
    if(NOT ARG_${required})
      message(FATAL_ERROR "hrr_add_codegen requires ${required}")
    endif()
  endforeach()

  if(NOT ARG_CAPTURE AND NOT ARG_PLAYBACK)
    message(FATAL_ERROR "hrr_add_codegen requires CAPTURE and/or PLAYBACK")
  endif()

  find_package(Python3 COMPONENTS Interpreter REQUIRED)

  set(generator "${ARG_CLR_SOURCE_DIR}/hipamd/src/hrr/tools/gen_hrr_api_args.py")
  set(api_trace "${ARG_CLR_SOURCE_DIR}/hipamd/include/hip/amd_detail/hip_api_trace.hpp")
  set(public_header "${ARG_HIP_SOURCE_DIR}/include/hip/hip_runtime_api.h")
  set(header "${ARG_OUTPUT_DIR}/include/hrr/hrr_api_args.h")
  set(regions_source "${ARG_HRR_SOURCE_DIR}/include/hrr/hrr_regions.h")
  set(regions_header "${ARG_OUTPUT_DIR}/include/hrr/hrr_regions.h")
  set(capture "${ARG_OUTPUT_DIR}/hip_capture_generated.cpp")
  set(playback "${ARG_OUTPUT_DIR}/hip_playback_generated.cpp")

  set(outputs "${header}" "${regions_header}")
  set(generator_args
    --input "${api_trace}"
    --public-header "${public_header}"
    --output-header "${header}"
    --output-capture "${capture}"
    --output-playback "${playback}"
    --check-hrr-coverage)

  if(ARG_CAPTURE)
    list(APPEND outputs "${capture}")
  else()
    list(APPEND generator_args --skip-capture)
  endif()

  if(ARG_PLAYBACK)
    list(APPEND outputs "${playback}")
  else()
    list(APPEND generator_args --skip-playback)
  endif()

  add_custom_command(
    OUTPUT ${outputs}
    COMMAND ${Python3_EXECUTABLE} "${generator}" ${generator_args}
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${regions_source}" "${regions_header}"
    DEPENDS "${generator}" "${api_trace}" "${public_header}" "${regions_source}"
    COMMENT "Generating HRR API capture/playback sources"
    VERBATIM)
  set_source_files_properties(${outputs} PROPERTIES GENERATED TRUE)
  add_custom_target(${ARG_TARGET} DEPENDS ${outputs})
endfunction()
