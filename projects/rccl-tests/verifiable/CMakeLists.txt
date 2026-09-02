#
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Modifications Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE.txt for license information
#

if(TARGET roc::rccl)
  find_program(HIPIFY_PERL_EXECUTABLE hipify-perl)
  if(NOT HIPIFY_PERL_EXECUTABLE)
    message(FATAL_ERROR "hipify-perl is required to build verifiable with RCCL")
  endif()

  set(VERIFIABLE_HIPIFY_DIR "${CMAKE_CURRENT_BINARY_DIR}/hipify")
  set(VERIFIABLE_HIPIFIED_SOURCE "${VERIFIABLE_HIPIFY_DIR}/verifiable.cu.cpp")
  set(VERIFIABLE_HIPIFIED_HEADER "${VERIFIABLE_HIPIFY_DIR}/verifiable.h")
  set(VERIFIABLE_HIPIFIED_FLOAT8 "${VERIFIABLE_HIPIFY_DIR}/rccl_float8.h")

  add_custom_command(
    OUTPUT
      "${VERIFIABLE_HIPIFIED_SOURCE}"
      "${VERIFIABLE_HIPIFIED_HEADER}"
      "${VERIFIABLE_HIPIFIED_FLOAT8}"
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${VERIFIABLE_HIPIFY_DIR}"
    COMMAND "${HIPIFY_PERL_EXECUTABLE}" -quiet-warnings
            "${CMAKE_CURRENT_SOURCE_DIR}/verifiable.cu"
            -o "${VERIFIABLE_HIPIFIED_SOURCE}"
    COMMAND "${HIPIFY_PERL_EXECUTABLE}" -quiet-warnings
            "${CMAKE_CURRENT_SOURCE_DIR}/verifiable.h"
            -o "${VERIFIABLE_HIPIFIED_HEADER}"
    COMMAND "${HIPIFY_PERL_EXECUTABLE}" -quiet-warnings
            "${CMAKE_SOURCE_DIR}/src/rccl_float8.h"
            -o "${VERIFIABLE_HIPIFIED_FLOAT8}"
    DEPENDS
      "${CMAKE_CURRENT_SOURCE_DIR}/verifiable.cu"
      "${CMAKE_CURRENT_SOURCE_DIR}/verifiable.h"
      "${CMAKE_SOURCE_DIR}/src/rccl_float8.h"
    COMMENT "Hipifying verifiable sources"
    VERBATIM
  )

  add_custom_target(verifiable_hipify
    DEPENDS
      "${VERIFIABLE_HIPIFIED_SOURCE}"
      "${VERIFIABLE_HIPIFIED_HEADER}"
      "${VERIFIABLE_HIPIFIED_FLOAT8}"
  )

  add_library(verifiable STATIC "${VERIFIABLE_HIPIFIED_SOURCE}")
  add_dependencies(verifiable verifiable_hipify)
  set_target_properties(verifiable PROPERTIES POSITION_INDEPENDENT_CODE ON)
  rccl_tests_apply_offload_flags(verifiable)

  target_link_libraries(verifiable
    PUBLIC
      nccl_tests_options
      roc::rccl
      hip::device
  )

  target_include_directories(verifiable
    PUBLIC
      "${VERIFIABLE_HIPIFY_DIR}"
  )
else()
  add_library(verifiable STATIC verifiable.cu)

  target_link_libraries(verifiable
    PUBLIC
      nccl_tests_options
      nccl
      CUDA::cudart
  )

  target_include_directories(verifiable
    PUBLIC
      "${CMAKE_CURRENT_SOURCE_DIR}"
  )
endif()
