/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2015-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NCCL_INT_DEBUG_H_
#define NCCL_INT_DEBUG_H_

#include "nccl.h"
#include "nccl_common.h"
#include <stdio.h>
#include <thread>
#include "compiler.h"

// Conform to pthread and NVTX standard
#define NCCL_THREAD_NAMELEN 16

extern uint32_t ncclDebugLevelMask;
extern uint64_t ncclDebugMask;
extern FILE* ncclDebugFile;

#define NCCL_DEBUG_LEVEL_MASK_UNINITIALIZED (~0u)
#define NCCL_DEBUG_LEVEL_MASK_RESET_TRIGGERED (~1u)

static inline bool ncclDebugShouldLog(int msgLevel, unsigned long flags, uint64_t mask) {
  uint32_t levelMask = COMPILER_ATOMIC_LOAD(&ncclDebugLevelMask, std::memory_order_acquire);
  // Let the first log call initialize the masks, then re-check them.
  if (levelMask == NCCL_DEBUG_LEVEL_MASK_UNINITIALIZED || levelMask == NCCL_DEBUG_LEVEL_MASK_RESET_TRIGGERED)
    return true;
  if ((flags & mask) == 0) return false;
  return levelMask & (1u << msgLevel);
}

#ifdef NCCL_OS_LINUX
void ncclDebugLog(ncclDebugLogLevel level, unsigned long flags, const char* filefunc, int line, const char* fmt, ...)
  __attribute__((format(printf, 5, 6)));
#elif defined(NCCL_OS_WINDOWS)
void ncclDebugLog(ncclDebugLogLevel level, unsigned long flags, const char* filefunc, int line, const char* fmt, ...);
#else
/* Fallback so headers (e.g. alloc.h via checks.h) compile when OS is not set (e.g. unit tests with MPI). */
void ncclDebugLog(ncclDebugLogLevel level, unsigned long flags, const char* filefunc, int line, const char* fmt, ...);
#endif

// Let code temporarily downgrade WARN into INFO
extern thread_local int ncclDebugNoWarn;
extern char ncclLastError[];

#define ERROR(...) ncclDebugLog(NCCL_LOG_ERROR, NCCL_ALL, __FILE__, __LINE__, __VA_ARGS__)
#define VERSION(...) ncclDebugLog(NCCL_LOG_VERSION, NCCL_ALL, __FILE__, __LINE__, __VA_ARGS__)
#define WARN(...) ncclDebugLog(NCCL_LOG_WARN, NCCL_ALL, __FILE__, __LINE__, __VA_ARGS__)
#define ATTN(...) ncclDebugLog(NCCL_LOG_ATTN, NCCL_ALL, __FILE__, __LINE__, __VA_ARGS__)

#define NOWARN(EXPR, FLAGS) \
  do { \
    int oldNoWarn = ncclDebugNoWarn; \
    ncclDebugNoWarn = FLAGS; \
    (EXPR); \
    ncclDebugNoWarn = oldNoWarn; \
  } while (0)

#define INFO(FLAGS, ...) \
  do { \
    if (ncclDebugShouldLog(NCCL_LOG_INFO, (unsigned long)(FLAGS), ncclDebugMask)) \
      ncclDebugLog(NCCL_LOG_INFO, (unsigned long)(FLAGS), __func__, __LINE__, __VA_ARGS__); \
  } while (0)

#define INFO_LOC_FN(FLAGS, file, line, fn, fmt, ...) \
  INFO((FLAGS), "%s:%d (%s) " fmt, (file), (line), (fn), ##__VA_ARGS__)
#define INFO_LOC(FLAGS, fmt, ...) INFO_LOC_FN((FLAGS), __FILE__, __LINE__, __func__, fmt, ##__VA_ARGS__)

#define TRACE_CALL(...) \
  do { \
    if (ncclDebugShouldLog(NCCL_LOG_TRACE, NCCL_CALL, ncclDebugMask)) { \
      ncclDebugLog(NCCL_LOG_TRACE, NCCL_CALL, __func__, __LINE__, __VA_ARGS__); \
    } \
  } while (0)

#ifdef ENABLE_TRACE
#define TRACE(FLAGS, ...) \
  do { \
    if (ncclDebugShouldLog(NCCL_LOG_TRACE, (unsigned long)(FLAGS), ncclDebugMask)) { \
      ncclDebugLog(NCCL_LOG_TRACE, (unsigned long)(FLAGS), __func__, __LINE__, __VA_ARGS__); \
    } \
  } while (0)
#define TRACE_LOC_FN(FLAGS, file, line, fn, fmt, ...) \
  TRACE((FLAGS), "%s:%d (%s) " fmt, (file), (line), (fn), ##__VA_ARGS__)
#define TRACE_LOC(FLAGS, fmt, ...) TRACE_LOC_FN((FLAGS), __FILE__, __LINE__, __func__, fmt, ##__VA_ARGS__)
#else
#define TRACE(...)
#define TRACE_LOC_FN(FLAGS, file, line, fn, fmt, ...)
#define TRACE_LOC(FLAGS, fmt, ...)
#endif

void ncclSetThreadName(std::thread& thread, const char* fmt, ...);
// [RCCL] Overload for legacy pthread_t-managed threads (e.g. net_ib*).
void ncclSetThreadName(pthread_t thread, const char* fmt, ...);

#ifdef __cplusplus
extern "C" {
#endif
#ifdef ncclResetDebugInit
#undef ncclResetDebugInit
#endif
void ncclResetDebugInit();
#ifdef __cplusplus
}
#endif

// RCCL custom error message handling.
static inline ncclResult_t rcclCudaErrorHandler(cudaError_t err) {
    // Print the cuda error
  ERROR("HIP failure: '%s'", cudaGetErrorString(err));

    // Special error message here:
  switch (err) {
  case cudaErrorStreamCaptureInvalidated:
    ERROR("Application is trying to use an invalidated stream to launch RCCL kernel. "
          "This operation is invalid. RCCL is exiting.");
    break;
  default:
    break;
  }
  return ncclUnhandledCudaError;
}

#endif
