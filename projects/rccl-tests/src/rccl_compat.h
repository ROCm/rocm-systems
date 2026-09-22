/* ************************************************************************
 * Copyright (C) 2016-2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
 * ies of the Software, and to permit persons to whom the Software is furnished
 * to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
 * PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
 * CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 * ************************************************************************ */

#ifndef RCCL_COMPAT_H
#define RCCL_COMPAT_H

// Weak symbol forward declarations for runtime compatibility across RCCL versions
// These allow the code to compile with older headers but run with newer runtime libraries
extern "C" {
// For nccl.h < 2.13 since we define a weak fallback
#if NCCL_VERSION_CODE < NCCL_VERSION(2,13,0)
  char const* ncclGetLastError(ncclComm_t comm);
#endif

// Ensures backward compatibility for FP8 datatypes
#if NCCL_VERSION_CODE < NCCL_VERSION(2,24,3)
  #define ncclFloat8e4m3 ncclFp8E4M3
  #define ncclFloat8e5m2 ncclFp8E5M2
#endif

#if NCCL_VERSION_CODE < NCCL_VERSION(2,27,0)
  struct ncclWindow;
  typedef struct ncclWindow *ncclWindow_t;
  ncclResult_t ncclCommWindowRegister(ncclComm_t, void*, size_t, ncclWindow_t*, int) __attribute__((weak));
  ncclResult_t ncclCommWindowDeregister(ncclComm_t, ncclWindow_t) __attribute__((weak));
#endif
#if !(defined(ENABLE_DEVICE_API) && NCCL_VERSION_CODE >= NCCL_VERSION(2,28,0))
  struct ncclDevComm;          // Device API from 2.28
  struct ncclDevCommRequirements;
  ncclResult_t ncclDevCommCreate(ncclComm_t, ncclDevCommRequirements*, ncclDevComm*) __attribute__((weak));
  ncclResult_t ncclDevCommDestroy(ncclDevComm) __attribute__((weak));
#endif
#if NCCL_VERSION_CODE < NCCL_VERSION(2,29,0)
  struct ncclCommProperties;   // Available from 2.29
  ncclResult_t ncclCommQueryProperties(ncclComm_t, ncclCommProperties*) __attribute__((weak));
#endif

// Weak fallback for AlltoAll/AlltoAllv APIs (runtime compatibility)
#if NCCL_VERSION_CODE < NCCL_VERSION(2,28,0)
  ncclResult_t ncclAlltoAll(const void*, void*, size_t, ncclDataType_t, ncclComm_t, hipStream_t) __attribute__((weak));
  ncclResult_t ncclAlltoAllv(const void*, const size_t*, const size_t*, void*, const size_t*, const size_t*, ncclDataType_t, ncclComm_t, hipStream_t) __attribute__((weak));
#else
  ncclResult_t ncclAllToAll(const void*, void*, size_t, ncclDataType_t, ncclComm_t, hipStream_t) __attribute__((weak));
  ncclResult_t ncclAllToAllv(const void*, const size_t*, const size_t*, void*, const size_t*, const size_t*, ncclDataType_t, ncclComm_t, hipStream_t) __attribute__((weak));
#endif
}

#endif