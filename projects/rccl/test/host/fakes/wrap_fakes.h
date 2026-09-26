/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Remaining dependency seams used by wrap-test.cc that do not yet have a
// dedicated shared header. DDA seams live in dda_fakes.h; shared subsystem
// seams are included from their production-owned headers below.

#ifndef RCCL_TEST_HOST_WRAP_FAKES_H_
#define RCCL_TEST_HOST_WRAP_FAKES_H_

#include <cstdint>
#include <functional>

// Include canonical fake-owner headers here so wrap-test.cc consumes their
// seams without redeclaring them. comm.h/enqueue.h provide the remaining
// ncclComm, task, datatype, and algorithm types used below.
#include "ce_fakes.h"
#include "comm.h"
#include "dda_fakes.h"
#include "enqueue.h"
#include "nccl.h"
#include "strongstream_stubs.h"
#include "sym_kernels_fakes.h"
#include "transport_stubs.h"
#include "tuning_fakes.h"

// The env-var test-control API (SetMicroEnv/SetMicroEnvAbsent/ClearMicroEnv)
// is declared by fakes/env_fakes.h, the shared owner of getenv interposition
// for every microtest binary -- include that directly rather than this file
// for those. See MICROTEST_README.md's "Where a fake belongs".

// getFirmwareVersion()'s sole dependency, made settable so a test can script
// a canned firmware response or a failure.
extern std::function<ncclResult_t(uint32_t, uint64_t*)> g_amdSmiGetFirmwareVersion;

// External parameter accessors made settable so tests can drive their
// respective guards independently.
extern std::function<int64_t()> g_paramForceCe;
extern std::function<int64_t()> g_paramP2pDisable;

// Deliberate reset opt-out: every mutable hook in this file is installed via
// ScopedHook (or exercised in a forked RUN_ISOLATED_TEST child), so state is
// restored at scope exit or discarded with the child process. This suite has
// no persistent hook state for a ResetWrapFakes() entry point to clean up.

// ---------------------------------------------------------------------------
// Dispatcher seams: drive rcclSelectAllReduce/AllGather/ReduceScatter,
// rcclHierarchicalAlgoInfo, rcclGetAlgoInfo, rcclGetCollImplInfo, and
// rcclSymkQuery/rcclSymKGetInfo's deep path. See wrap_fakes.cc for each
// default's rationale.
// ---------------------------------------------------------------------------

extern std::function<bool(struct ncclComm*, ncclFunc_t, int, ncclDataType_t, size_t, const void*, void*, bool)>
    g_isSymmetricKernelRequested;
extern std::function<bool(const struct ncclComm*, size_t, ncclDataType_t, bool, bool, bool)> g_allReduceShouldTakeDdaPath;

extern std::function<ncclResult_t(struct ncclComm*, struct ncclTaskColl*, int, int, int, ncclSimInfo_t*)>
    g_getAlgoInfo;
extern std::function<int(struct ncclComm*, ncclFunc_t, size_t, ncclDataType_t, int, int)> g_kernelPackedChannels;
extern std::function<ncclResult_t(const ncclComm_t, int*)> g_commCount;

#endif  // RCCL_TEST_HOST_WRAP_FAKES_H_
