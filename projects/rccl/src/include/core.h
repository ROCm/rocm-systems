/*************************************************************************
 * Copyright (c) 2015-2021, NVIDIA CORPORATION. All rights reserved.
 * Modifications Copyright (c) 2019-2021 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_CORE_H_
#define NCCL_CORE_H_

#include <pthread.h>
#include <unistd.h>
#include <stdlib.h>
#include <stdint.h>
#include <algorithm> // For std::min/std::max
#include "nccl.h"

#if defined(NCCL_OS_LINUX)
#ifdef PROFAPI
#define NCCL_API(ret, func, args...) \
  extern "C" __attribute__((visibility("default"))) __attribute__((alias(#func))) ret p##func(args); \
  extern "C" __attribute__((visibility("default"))) __attribute__((weak)) ret func(args)
#else
#define NCCL_API(ret, func, args...) extern "C" __attribute__((visibility("default"))) ret func(args)
#endif // end PROFAPI
#elif defined(NCCL_OS_WINDOWS)
#define NCCL_STRINGIFY_(x) #x
#define NCCL_STRINGIFY(x) NCCL_STRINGIFY_(x)
#if defined(NCCL_BUILD_SHARED_LIB)
#define NCCL_API(ret, func, ...) \
  __pragma(comment(linker, "/export:" NCCL_STRINGIFY(func))) extern "C" ret func(__VA_ARGS__)
#else
#define NCCL_API(ret, func, ...) extern "C" ret func(__VA_ARGS__)
#endif
#else
/* Other non-Linux platforms: use standard variadic macro (no visibility attribute) */
#define NCCL_API(ret, func, ...) extern "C" ret func(__VA_ARGS__)
#endif // end NCCL_OS_LINUX

#include "debug.h"
#include "checks.h"
#include "rocmwrap.h"
#include "alloc.h"
#include "utils.h"
#include "param.h"
#ifdef NVTX_NO_IMPL
#include "nvtx_stub.h"
#else
#include "nvtx.h"
#endif

#endif // end include guard
