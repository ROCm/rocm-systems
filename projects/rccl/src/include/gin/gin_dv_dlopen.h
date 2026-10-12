/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef GIN_DV_DLOPEN_H_
#define GIN_DV_DLOPEN_H_

#include <dlfcn.h>

// The DV libraries are also libibverbs providers. If dlopen resolves to a
// different file than the one libibverbs loaded, the copy registers its own
// driver, which libibverbs cannot unregister, so it must never be unmapped.
static inline void* gin_dv_dlopen(const char* path) {
  return dlopen(path, RTLD_LAZY | RTLD_NODELETE);
}

#endif
