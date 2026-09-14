/*************************************************************************
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/
//
// Fail-loud (::abort()) link-satisfiers for the collective launch/registration
// pipeline plus the transport connect entry points a group/enqueue TU
// references but that a host-only control-flow test never executes. Reaching
// one at run time is a real escape and the abort surfaces it immediately.
//
// Self-contained by design: unlike the init binary's transport_stubs.cc (whose
// NVLS/P2P-level stubs now route through test-driven seam globals defined in
// the init test's own TUs), this floor has no external seam globals, so it
// links into any micro-test binary on its own.
//
// It also carries the local floor for a few symbols whose owning stubs file this
// target cannot link: nccl_stubs.cc and sched_stubs.cc own ncclMemAlloc /
// ncclMemFree / ncclShadowPoolToHost, but both also define symbols the real
// group.cc already supplies here (ncclGroupDepth, ncclGroupError,
// ncclAsyncLaunch). Deduplicating that properly is AICOMRCCL-1688.

#include <cstdlib>

#include "nccl.h"
#include "comm.h"
#include "mem_manager.h"
#include "enqueue.h"
#include "ce_coll.h"
#include "rma/rma.h"
#include "rma/rma_ce.h"
#include "argcheck.h"
#include "dev_runtime.h"
#include "transport.h"
#include "os.h"
#include "allocator.h"

// enqueue.h
ncclResult_t ncclPrepareTasks(struct ncclComm*, bool*, bool*, ncclSimInfo_t*) { ::abort(); }
ncclResult_t ncclTasksRegAndEnqueue(struct ncclComm*) { ::abort(); }
ncclResult_t ncclLaunchPrepare(struct ncclComm*) { ::abort(); }
ncclResult_t ncclLaunchKernelBefore_NoUncapturedCuda(struct ncclComm*, struct ncclKernelPlan*) { ::abort(); }
ncclResult_t ncclLaunchKernel(struct ncclComm*, struct ncclKernelPlan*) { ::abort(); }
ncclResult_t ncclLaunchKernelAfter_NoCuda(struct ncclComm*, struct ncclKernelPlan*) { ::abort(); }
ncclResult_t ncclLaunchFinish(struct ncclComm*) { ::abort(); }

// ce_coll.h
ncclResult_t ncclCeInit(struct ncclComm*) { ::abort(); }
ncclResult_t ncclLaunchCeColl(struct ncclComm*, struct ncclKernelPlan*) { ::abort(); }

// rma/rma.h, rma/rma_ce.h
// ncclLaunchRma and ncclRmaCeInit are absent on purpose: rma-test.cc and
// rma-ce-test.cc compile the real rma.cc / rma_ce.cc into this binary, so a
// stub for either would be a duplicate symbol.

// dev_runtime.h
ncclResult_t ncclDevrCommCreateInternal(struct ncclComm*, struct ncclDevCommRequirements*,
                                        struct ncclDevComm*, bool, struct ncclDevCommCompat*) { ::abort(); }
ncclResult_t ncclDevrWindowRegisterInGroup(struct ncclComm*, void*, size_t, int,
                                           struct ncclWindow_vidmem**) { ::abort(); }
void freeDevCommRequirements(struct ncclDevCommRequirements*) { ::abort(); }

// mem_manager.h
ncclResult_t ncclCommMemSuspend(struct ncclComm*) { ::abort(); }
ncclResult_t ncclCommMemResume(struct ncclComm*) { ::abort(); }

// argcheck.h
ncclResult_t ncclArgsGlobalCheck(struct ncclArgsInfo*) { ::abort(); }

// os.h
int ncclOsCpuCount(const ncclAffinity&) { ::abort(); }
ncclResult_t ncclOsSetAffinity(const ncclAffinity&) { ::abort(); }

// transport.h -- only the connect/setup entry points group.cc references.
ncclResult_t ncclTransportP2pSetup(struct ncclComm*, struct ncclTopoGraph*, int, bool*) { ::abort(); }
ncclResult_t ncclTransportRingConnect(struct ncclComm*) { ::abort(); }
ncclResult_t ncclTransportTreeConnect(struct ncclComm*) { ::abort(); }
ncclResult_t ncclTransportPatConnect(struct ncclComm*) { ::abort(); }
ncclResult_t ncclNvlsBufferSetup(struct ncclComm*) { ::abort(); }
ncclResult_t ncclNvlsTreeConnect(struct ncclComm*) { ::abort(); }
ncclResult_t ncclCollNetChainBufferSetup(ncclComm_t) { ::abort(); }
ncclResult_t ncclCollNetDirectBufferSetup(ncclComm_t) { ::abort(); }

// allocator.h / nccl.h -- see the note at the top of this file for why these are
// here rather than in nccl_stubs.cc / sched_stubs.cc. Host memory is enough; the
// RMA copy-engine path only needs its signal window to be addressable.
ncclResult_t ncclMemAlloc(void** ptr, size_t size) {
  if (ptr == nullptr) return ncclInvalidArgument;
  *ptr = ::calloc(1, size);
  return *ptr ? ncclSuccess : ncclSystemError;
}

ncclResult_t ncclMemFree(void* ptr) {
  ::free(ptr);
  return ncclSuccess;
}

// No device object in a host-only build, so the identity mapping is the honest
// answer: the pointer handed in is the one that comes back.
ncclResult_t ncclShadowPoolToHost(struct ncclShadowPool*, void* devObj, void** outHostObj) {
  if (outHostObj) *outHostObj = devObj;
  return ncclSuccess;
}

// api_trace.h
ncclResult_t ncclCommWindowDeregister(ncclComm_t, ncclWindow_t) { return ncclSuccess; }
