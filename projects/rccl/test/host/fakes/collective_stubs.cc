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

// enqueue.h
ncclResult_t ncclPrepareTasks(struct ncclComm*, bool*, bool*, ncclSimInfo_t*) { ::abort(); }
ncclResult_t ncclTasksRegAndEnqueue(struct ncclComm*) { ::abort(); }
ncclResult_t ncclLaunchPrepare(struct ncclComm*) { ::abort(); }
ncclResult_t ncclLaunchKernelBefore_NoUncapturedCuda(struct ncclComm*, struct ncclKernelPlan*) { ::abort(); }
ncclResult_t ncclLaunchKernel(struct ncclComm*, struct ncclKernelPlan*) { ::abort(); }
ncclResult_t ncclLaunchKernelAfter_NoCuda(struct ncclComm*, struct ncclKernelPlan*) { ::abort(); }
ncclResult_t ncclLaunchFinish(struct ncclComm*) { ::abort(); }
// The 2.31 task-prep split: group.cc calls ncclTaskPrepare where it used to
// call ncclPrepareTasks, and both it and dev_runtime.cc gate the rearch job
// path on this param. Pinned to 0 so those call sites take the in-group task
// branch, which is the one the suites here drive.
ncclResult_t ncclTaskPrepare(struct ncclComm*, ncclSimInfo_t*) { ::abort(); }
int64_t ncclParamEnqueueRearchEnable() { return 0; }

// ce_coll.h
ncclResult_t ncclCeInit(struct ncclComm*) { ::abort(); }
ncclResult_t ncclLaunchCeColl(struct ncclComm*, struct ncclKernelPlan*) { ::abort(); }

// rma/rma.h, rma/rma_ce.h
// ncclLaunchRma is absent on purpose: rma-test.cc compiles the real rma.cc in,
// so a stub here would be a duplicate symbol.
ncclResult_t ncclRmaCeInit(struct ncclComm*) { ::abort(); }

// dev_runtime.h
// ncclDevrCommCreateInternal, ncclDevrWindowRegisterInGroup and
// freeDevCommRequirements used to be ::abort() stubs here. dev_runtime.cc is
// now compiled into this binary (dev-runtime-test.cc) and defines all three for
// real, so the stubs would be duplicate symbols. Nothing regressed by dropping
// them: an ::abort() stub is only ever reached by a test that should not have
// called it, and no suite in this binary did.

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
