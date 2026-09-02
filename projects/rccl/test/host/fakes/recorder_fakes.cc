/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// See recorder_fakes.h.
//
// NOT "recorder.h": comm.h already pulls in one copy and that header has no
// include guard, so including it again re-declares the whole rcclCall_t
// enumerator list ("redefinition of enumerator rrBroadcast").

#include "recorder_fakes.h"

#include "comm.h"

ncclResult_t g_recorderResult = ncclSuccess;

void ResetRecorderFakes() { g_recorderResult = ncclSuccess; }

// The overloads the microtest units reach today, not all 12 declared in
// recorder.h. Add one here when a unit needs it; do not re-add it locally.
namespace rccl {
Recorder::Recorder() {}
Recorder::~Recorder() {}
Recorder& Recorder::instance() {
  static Recorder inst;
  return inst;
}

void Recorder::record(const char*) {}                                  // non-replayable
void Recorder::record(ncclComm_t*, int, const int*) {}                 // CommInitAll
void Recorder::record(int, ncclSimInfo_t*) {}                          // SimulatedGroupEnd
void Recorder::record(rcclCall_t, int, int, ncclUniqueId*, ncclConfig_t*, ncclComm_t) {}

ncclResult_t Recorder::record(rcclCall_t, int) { return g_recorderResult; }            // group op
ncclResult_t Recorder::record(rcclCall_t, ncclComm_t) { return g_recorderResult; }     // comm destroy
ncclResult_t Recorder::record(rcclCall_t, int, int, ncclUniqueId*, ncclComm_t, int) {  // init
  return g_recorderResult;
}
// recorder.h:128 declares ONE overload with defaulted trailing args, which both
// RedOpCreate (6 args) and RedOpDestroy (3 args) call. Defining a second 3-arg
// version here would be a redefinition, not an overload.
ncclResult_t Recorder::record(rcclCall_t, ncclRedOp_t, ncclComm_t, ncclDataType_t,
                              ncclScalarResidence_t, void*) {
  return g_recorderResult;
}
}  // namespace rccl
