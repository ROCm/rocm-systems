/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Aggregation header for the `rccl-UnitTestsMicroEnqueue` binary.
//
// It defines no seams of its own. Every symbol enqueue.cc needs from outside
// itself belongs to some OTHER production TU, so each seam lives in a fakes file
// named after that TU and is shared with the other microtest binaries. This
// header just pulls in the set enqueue-test.cc uses and declares the reset that
// chains their per-TU resets. See test/host/MICROTEST_README.md for the
// production-TU-to-fakes-file map.
//
// LINK FLOOR ONLY: a seam marked `// UNDRIVEN` in one of those headers is
// declared so the binary links and so an accidental call is visible, NOT because
// its path is covered. The marker travels with the declaration.

#ifndef RCCL_TEST_HOST_ENQUEUE_FAKES_H_
#define RCCL_TEST_HOST_ENQUEUE_FAKES_H_

#include "ce_fakes.h"            // src/ce_coll.cc
#include "comm_fakes.h"          // src/init.cc comm lifecycle
#include "dev_runtime_fakes.h"   // src/dev_runtime.cc
#include "env_fakes.h"           // src/misc/param.cc + getenv interposition
#include "hip_fakes.h"           // HIP runtime seams
#include "nccl_fakes.h"          // reusable nccl* seams
#include "proxy_fakes.h"         // src/proxy.cc
#include "rccl_wrap_fakes.h"     // src/rccl_wrap.cc
#include "recorder_fakes.h"      // src/recorder.cc
#include "sym_kernels_fakes.h"   // src/sym_kernels.cc
#include "transport_stubs.h"     // src/transport/net.cc (rcclUseAinic)
#include "tuning_fakes.h"        // src/graph/tuning.cc

// TRAP: updateCollCostTable:2527 caches NCCL_PROTO in a function-local static, so
// only the FIRST call in the PROCESS observes it. Scripting it later via
// SetMicroEnv is a no-op, and --gtest_shuffle passing reflects a constant latch
// rather than proven order-independence. The first test to script NCCL_PROTO
// inherits a seed-dependent flake and should add process isolation.
//
// topoGetAlgoInfo:2687-2688 read NCCL_PROTO/NCCL_ALGO through raw libc getenv()
// and not ncclGetEnv. env_fakes.cc interposes getenv itself, and
// ResetEnqueueFakes() maps both names ABSENT, so both routes are covered. The
// mapping is the load-bearing half: the interposer falls through to the real
// environment for any name the map does not know.

// waitWorkFifoAvailable escapes its spin loop via ncclCommPollEventCallbacks,
// which is INLINE in comm.h and cannot be faked. Its only observable action on an
// empty callback queue is hipThreadExchangeStreamCaptureMode, already routed
// through g_hipAsyncOpsResult (hip_fakes.h). Note that seam is a plain RESULT,
// not a callable -- it cannot count iterations or advance workFifoConsumed. A
// test that fills the FIFO must therefore make the poll FAIL, or set abortFlag;
// anything else spins forever and HANGS the suite rather than failing it.

void ResetEnqueueFakes();

#endif  // RCCL_TEST_HOST_ENQUEUE_FAKES_H_
