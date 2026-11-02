/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_DV_DLOPEN_H_
#define NCCL_DV_DLOPEN_H_

#include <dlfcn.h>

// DV libraries (libionic, libmlx5, libbnxt_re) are also libibverbs providers,
// whose drivers libibverbs cannot unregister, so they must never be unmapped.
constexpr int kDvDlopenFlags = RTLD_NODELETE;

static inline void* ncclDvDlopen(const char* path, int mode) {
  return dlopen(path, mode | kDvDlopenFlags);
}

#endif
