/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
 *                         All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 * Modifications Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for more license information
 *
 * Minimal profiler-API types for the accl-profiler plugin only. Values must stay in sync with
 * src/include/plugin/nccl_profiler.h and src/include/nccl_common.h.
 *************************************************************************/

#ifndef ACCL_SHIM_H_
#define ACCL_SHIM_H_

#include <stdint.h>
#include <sys/types.h>

typedef int ncclResult_t;
#define ncclSuccess 0

typedef void (*ncclDebugLogger_t)(int level, unsigned long flags,
                                  const char* file, int line,
                                  const char* fmt, ...);

typedef pid_t ncclPid_t;

typedef enum {
  ncclProfilerProxyStepSendGPUWait     = 8,
  ncclProfilerProxyStepSendWait        = 9,
  ncclProfilerProxyStepRecvWait        = 10,
  ncclProfilerProxyStepRecvFlushWait   = 11,
  ncclProfilerProxyStepRecvGPUWait     = 12,
  ncclProfilerProxyStepSendPeerWait_v4 = 20,
  ncclProfilerKernelChStop             = 22,
} ncclProfilerEventState_v5_t;

enum {
  ncclProfileGroup          = (1 << 0),
  ncclProfileColl           = (1 << 1),
  ncclProfileP2p            = (1 << 2),
  ncclProfileProxyOp        = (1 << 3),
  ncclProfileProxyStep      = (1 << 4),
  ncclProfileProxyCtrl      = (1 << 5),
  ncclProfileKernelCh       = (1 << 6),
  ncclProfileNetPlugin      = (1 << 7),
};

#include "profiler/profiler_v5.h"

#endif
