#
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Modifications Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE.txt for license information
#

if(TARGET roc::rccl)
  set(VERIFIABLE_HIPIFY_DIR "${CMAKE_CURRENT_BINARY_DIR}/hipify")
  set(VERIFIABLE_HIPIFIED_SOURCE "${VERIFIABLE_HIPIFY_DIR}/verifiable.cu.cpp")
  set(VERIFIABLE_HIPIFIED_HEADER "${VERIFIABLE_HIPIFY_DIR}/verifiable.h")
  set(VERIFIABLE_HIPIFIED_FLOAT8 "${VERIFIABLE_HIPIFY_DIR}/rccl_float8.h")

  rccl_tests_hipify("${VERIFIABLE_HIPIFIED_SOURCE}" "${CMAKE_CURRENT_SOURCE_DIR}/verifiable.cu")
  rccl_tests_hipify("${VERIFIABLE_HIPIFIED_HEADER}" "${CMAKE_CURRENT_SOURCE_DIR}/verifiable.h")
  rccl_tests_hipify("${VERIFIABLE_HIPIFIED_FLOAT8}" "${PROJECT_SOURCE_DIR}/src/rccl_float8.h")

  add_custom_target(verifiable_hipify
    DEPENDS
      "${VERIFIABLE_HIPIFIED_SOURCE}"
      "${VERIFIABLE_HIPIFIED_HEADER}"
      "${VERIFIABLE_HIPIFIED_FLOAT8}"
  )

  add_library(verifiable STATIC "${VERIFIABLE_HIPIFIED_SOURCE}")
  add_dependencies(verifiable verifiable_hipify)
  set_target_properties(verifiable PROPERTIES POSITION_INDEPENDENT_CODE ON)

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
