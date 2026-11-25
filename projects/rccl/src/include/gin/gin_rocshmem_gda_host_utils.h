/*************************************************************************
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 ************************************************************************/

// Private, non-installed host helpers for the compiled-in providers
// (provider_gda_mlx5.cpp), pulled in by a build-time sed that rewrites that
// file's rocSHMEM util.hpp include. These generically-named CHECK_* macros are
// quarantined here so they can't leak into a consumer TU — the installed public
// util.hpp (reached via the GIN device API) must not define them.

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
