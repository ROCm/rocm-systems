/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef HIP_LOADER_HIP_LOADER_ABI_H
#define HIP_LOADER_HIP_LOADER_ABI_H

#include <stdint.h>

#if defined(_WIN32)
#if defined(HIP_LOADER_BUILDING_LIBRARY)
#define HIP_LOADER_EXPORT __declspec(dllexport)
#else
#define HIP_LOADER_EXPORT __declspec(dllimport)
#endif
#else
#define HIP_LOADER_EXPORT __attribute__((visibility("default")))
#endif

typedef struct hip_loader_backend_info_v1 {
  uint32_t struct_size;
  uint32_t loader_backend_abi_version;
  uint32_t backend_api_major;
  uint32_t backend_api_minor;
  uint32_t flags;
  const char* backend_name;
} hip_loader_backend_info_v1;

#define HIP_LOADER_BACKEND_API_MAJOR 7u
#define HIP_LOADER_BACKEND_API_MINOR 0u
#define HIP_LOADER_BACKEND_ABI_VERSION 1u

typedef struct hip_loader_test_call_v1 {
  uint32_t struct_size;
  const char* symbol;
  uint32_t phase;
  hipError_t default_result;
  void* args[8];
} hip_loader_test_call_v1;

typedef hipError_t (*hip_loader_test_api_callback_t)(hip_loader_test_call_v1* call);

#endif
