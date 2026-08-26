/*
 * Copyright (c) 2016-2024, NVIDIA CORPORATION. All rights reserved.
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

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
