/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Fakes for src/graph/tuning.cc (and the two channel-clamp params that
// src/graph/connect.cc emits), shared by the host-only microtest binaries.

#ifndef RCCL_TEST_HOST_TUNING_FAKES_H_
#define RCCL_TEST_HOST_TUNING_FAKES_H_

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>

#include "nccl.h"

struct ncclComm;

// -------------------------------------------------------------------------
// ncclTopoGetAlgoTime (tuning.cc:1599) fills the cost table that a caller's
// algorithm selection reads, so a test drives selection entirely through this
// rather than by constructing a real topology.
//
// TRAP: the cost table is float[NCCL_NUM_ALGORITHMS][NCCL_NUM_PROTOCOLS] and
// NCCL_ALGO_PROTO_IGNORE marks a cell as unusable. A fake that returns a time
// for EVERY (algo, proto) pair makes every cell selectable, which is not what
// production sees -- prefer scripting only the pairs a test cares about.
// -------------------------------------------------------------------------
extern std::function<ncclResult_t(struct ncclComm*, int coll, int algorithm, int protocol,
                                  size_t nBytes, int numPipeOps, float* time)>
    g_topoGetAlgoTime;
extern int g_topoGetAlgoTimeCalls;

// Min/max channel clamps. These do NOT mirror production: NCCL_PARAM defaults
// both to -2 (graph/connect.cc:832-833), the sentinel meaning "unset". The fakes
// default to 0/MAXCHANNELS instead, i.e. the already-resolved clamps, so a test
// sees an inert clamp without going through ncclMinNchannels/ncclMaxNchannels.
extern int64_t g_paramMinNchannels;  // UNDRIVEN
extern int64_t g_paramMaxNchannels;  // UNDRIVEN

// rcclGetTuningIndexForArch (tuning.cc:1637). Records the arch it was handed:
// a caller forwarding "" instead of comm->archName is invisible without it.
extern int g_tuningIndexValue;
extern std::string g_tuningIndexLastArch;

void ResetTuningFakes();

#endif  // RCCL_TEST_HOST_TUNING_FAKES_H_
