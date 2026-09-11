/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// See tuning_fakes.h.

#include "tuning_fakes.h"

#include "comm.h"
#include "device.h"

// Defaults to ncclSystemError, NOT success. Production wraps this in NCCLCHECK,
// so the effect is that the FIRST eligible lookup aborts the caller with an
// error, not that the table is left fully IGNOREd. That is deliberate: an
// unscripted topology query must fail loudly rather than quietly succeed,
// because a success default would make every (algo, proto) pair look selectable
// and any test that forgot to script a time would silently exercise the wrong path.
static ncclResult_t DefaultTopoGetAlgoTime(struct ncclComm*, int, int, int, size_t, int, float*) {
  return ncclSystemError;
}
std::function<ncclResult_t(struct ncclComm*, int, int, int, size_t, int, float*)>
    g_topoGetAlgoTime = DefaultTopoGetAlgoTime;
int g_topoGetAlgoTimeCalls = 0;

ncclResult_t ncclTopoGetAlgoTime(struct ncclComm* comm, int coll, int algorithm, int protocol,
                                 size_t nBytes, int numPipeOps, float* time) {
  ++g_topoGetAlgoTimeCalls;
  return g_topoGetAlgoTime(comm, coll, algorithm, protocol, nBytes, numPipeOps, time);
}

int64_t g_paramMinNchannels = 0;
int64_t g_paramMaxNchannels = MAXCHANNELS;
int64_t ncclParamMinNchannels() { return g_paramMinNchannels; }
int64_t ncclParamMaxNchannels() { return g_paramMaxNchannels; }

int g_tuningIndexValue = 0;
std::string g_tuningIndexLastArch;
int rcclGetTuningIndexForArch(const char* gfxarch) {
  g_tuningIndexLastArch = gfxarch ? gfxarch : "<null>";
  return g_tuningIndexValue;
}

void ResetTuningFakes() {
  g_topoGetAlgoTime = DefaultTopoGetAlgoTime;
  g_topoGetAlgoTimeCalls = 0;
  g_paramMinNchannels = 0;
  g_paramMaxNchannels = MAXCHANNELS;
  g_tuningIndexValue = 0;
  g_tuningIndexLastArch.clear();
}
