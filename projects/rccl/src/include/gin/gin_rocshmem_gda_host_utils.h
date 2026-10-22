/*************************************************************************
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 ************************************************************************/

// Private host-side helpers for the compiled-in rocSHMEM providers
// (provider_gda_mlx5.cpp). NOT installed — internal to librccl.so, pulled in by
// a build-time sed that replaces that file's rocSHMEM util.hpp include.
//
// These CHECK_* abort macros used to live in rocSHMEM's util.hpp, but rccl's
// util.hpp is an INSTALLED public device header reached by user code through the
// GIN device API (nccl_device.h -> ... -> queue_pair_mux.hpp -> util.hpp). Macros
// with generic names like CHECK_HIP must never leak into a consumer's TU, so they
// are quarantined here instead.

#ifndef GIN_ROCSHMEM_GDA_HOST_UTILS_H_
#define GIN_ROCSHMEM_GDA_HOST_UTILS_H_

#include <hip/hip_runtime.h>
#include <hsa/hsa_ext_amd.h>  // hsa_status_t for CHECK_HSA

#include <cerrno>
#include <cstring>

#include "log.hpp"  // LOG_ERROR_ABORT / LOG_ERROR_EXIT (rccl host shim)

namespace rocshmem {

#ifndef UNLIKELY
#define UNLIKELY(X) __builtin_expect(X, 0)
#endif

#define CHECK_NNULL(value, fn_str) do {                                        \
  if (UNLIKELY(nullptr == (value))) {                                          \
    LOG_ERROR_ABORT("%s: %s (%d)", fn_str, strerror(errno), errno);            \
  }                                                                            \
} while(0)

#define CHECK_ZERO(value, fn_str) do {                                         \
  if (UNLIKELY(0 != (value))) {                                                \
    LOG_ERROR_ABORT("%s: %s (%d)", fn_str, strerror(errno), errno);            \
  }                                                                            \
} while(0)

#define CHECK_HIP(instr) do {                                                  \
  hipError_t error = (instr);                                                  \
  if (error != hipSuccess) {                                                   \
    LOG_ERROR_ABORT(#instr ": %s (%d)", hipGetErrorString(error), error);      \
  }                                                                            \
} while(0)

#define CHECK_HSA(cmd) do {                                                    \
  hsa_status_t error = cmd;                                                     \
  if (error != HSA_STATUS_SUCCESS) {                                           \
    LOG_ERROR_EXIT(#cmd ": %d", error);                                        \
  }                                                                            \
} while (0)

}  // namespace rocshmem

#endif  // GIN_ROCSHMEM_GDA_HOST_UTILS_H_
