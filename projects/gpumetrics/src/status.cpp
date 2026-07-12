// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#include "gpumetrics/types.h"

extern "C" const char* gpum_status_string(gpum_status s) {
  switch (s) {
    case GPUM_OK: return "ok";
    case GPUM_ERR_INVALID_ARG: return "invalid argument";
    case GPUM_ERR_NOT_FOUND: return "not found";
    case GPUM_ERR_UNSUPPORTED: return "unsupported";
    case GPUM_ERR_NOT_INITIALIZED: return "not initialized";
    case GPUM_ERR_ALREADY_EXISTS: return "already exists";
    case GPUM_ERR_NO_DATA: return "no data";
    case GPUM_ERR_TIMEOUT: return "timeout";
    case GPUM_ERR_BACKEND: return "backend error";
    case GPUM_ERR_ABI: return "abi mismatch";
    case GPUM_ERR_INTERNAL: return "internal error";
  }
  return "unknown";
}
