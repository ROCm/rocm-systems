/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Aggregation header for rccl-UnitTestsMicroEnqueue: defines no seams, just the per-TU set and their chained reset.
#ifndef RCCL_TEST_HOST_ENQUEUE_TEST_DEPS_H_
#define RCCL_TEST_HOST_ENQUEUE_TEST_DEPS_H_

#include "ce_fakes.h"            // src/ce_coll.cc
#include "comm_fakes.h"          // src/init.cc comm lifecycle
#include "dev_runtime_fakes.h"   // src/dev_runtime.cc
#include "env_fakes.h"           // src/misc/param.cc + getenv interposition
#include "group_fakes.h"         // src/group.cc
#include "hip_fakes.h"           // HIP runtime seams
#include "nccl_fakes.h"          // reusable nccl* seams
#include "nccl_stubs.h"          // core/lifecycle functors; nccl_stubs.cc is linked into this binary too
#include "proxy_fakes.h"         // src/proxy.cc
#include "rccl_wrap_fakes.h"     // src/rccl_wrap.cc
#include "recorder_fakes.h"      // src/recorder.cc
#include "register_stubs.h"      // src/register/coll_reg.cc (hookable entry only)
#include "sym_kernels_fakes.h"   // src/sym_kernels.cc
#include "transport_stubs.h"     // src/transport/net.cc (rcclUseAinic)
#include "tuning_fakes.h"        // src/graph/tuning.cc

// TRAP: NCCL_PROTO is latched in a static at updateCollCostTable:2527 and read via raw getenv at topoGetAlgoInfo:2687.
// TRAP: a test that fills the work FIFO must fail the poll or set abortFlag, or waitWorkFifoAvailable spins and hangs.

void ResetEnqueueTestDeps();

#endif  // RCCL_TEST_HOST_ENQUEUE_TEST_DEPS_H_
