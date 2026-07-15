/*************************************************************************
 * Copyright (c) 2017-2022, NVIDIA CORPORATION. All rights reserved.
 * Modifications Copyright (c) 2019-2023 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_PARAM_H_
#define NCCL_PARAM_H_

#include <stdint.h>
#include "compiler.h"

const char* userHomeDir();
void setEnvFile(const char* fileName);
void initEnv();
const char *ncclGetEnv(const char *name);

int64_t ncclLoadParam(char const* env, int64_t deftVal, int64_t uninitialized, int64_t* cache, int8_t* noCache);

#define NCCL_PARAM(name, env, deftVal) \
  int64_t ncclParam##name() { \
    constexpr int64_t uninitialized = INT64_MIN; \
    static int8_t noCache = /*uninitialized*/ -1; \
    static_assert(deftVal != uninitialized, "default value cannot be the uninitialized value."); \
    static int64_t cache = uninitialized; \
    if (COMPILER_EXPECT(COMPILER_ATOMIC_LOAD(&cache, std::memory_order_relaxed) == uninitialized, false)) { \
      return ncclLoadParam("NCCL_" env, deftVal, uninitialized, &cache, &noCache); \
    } \
    return cache; \
  }

#define RCCL_PARAM_DECLARE(name) \
int64_t rcclParam##name()

#define RCCL_PARAM(name, env, deftVal) \
pthread_mutex_t rcclParamMutex##name = PTHREAD_MUTEX_INITIALIZER; \
int64_t rcclParam##name() { \
    constexpr int64_t uninitialized = INT64_MIN; \
    static_assert(deftVal != uninitialized, "default value cannot be the uninitialized value."); \
    static int64_t cache = uninitialized; \
    static int8_t noCache = /*uninitialized*/ -1; \
    if (__builtin_expect(__atomic_load_n(&cache, __ATOMIC_RELAXED) == uninitialized, false)) { \
      ncclLoadParam("RCCL_" env, deftVal, uninitialized, &cache, &noCache); \
    } \
    return cache; \
  }

// RCCL_PARAM variant that also accepts the NCCL_ prefix as an alias.
// Checks RCCL_<env> first; if unset, falls back to NCCL_<env>.
#define RCCL_PARAM_NCCL_ALIAS(name, env, deftVal) \
pthread_mutex_t rcclParamMutex##name = PTHREAD_MUTEX_INITIALIZER; \
int64_t rcclParam##name() { \
    constexpr int64_t uninitialized = INT64_MIN; \
    static_assert(deftVal != uninitialized, "default value cannot be the uninitialized value."); \
    static int64_t cache = uninitialized; \
    static int8_t noCache = /*uninitialized*/ -1; \
    if (__builtin_expect(__atomic_load_n(&cache, __ATOMIC_RELAXED) == uninitialized, false)) { \
      const char* _s = ncclGetEnv("RCCL_" env); \
      if (_s && strlen(_s) > 0) \
        ncclLoadParam("RCCL_" env, deftVal, uninitialized, &cache, &noCache); \
      else \
        ncclLoadParam("NCCL_" env, deftVal, uninitialized, &cache, &noCache); \
    } \
    return cache; \
  }

#endif
