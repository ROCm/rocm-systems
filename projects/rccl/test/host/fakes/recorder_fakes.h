/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// rccl::Recorder (src/recorder.cc) faked once for every host-only microtest
// binary. Previously the ctor/dtor/instance() triple was copied verbatim into
// comm_fakes.cc, init_fakes.cc and enqueue_test_deps.cc, differing only in which
// record() overloads each target happened to reference.

#ifndef RCCL_TEST_HOST_RECORDER_FAKES_H_
#define RCCL_TEST_HOST_RECORDER_FAKES_H_

#include <cstddef>
#include <functional>
#include <string>
#include <vector>

#include "nccl.h"

// Every label passed to the non-replayable record(const char*) overload, in call order.
extern std::vector<std::string> g_recorderLabels;

// Recorder is pure instrumentation, but production NCCLCHECKs the result of some
// record() calls (e.g. the RedOpCreate path), so a failing recorder is an
// observable arm. Every ncclResult_t-returning overload returns this.
extern ncclResult_t g_recorderResult;

// The init overload's arguments, recorded: ncclGetUniqueId_impl's record() call is a line of the
// unit under test, and a fake that drops what it was handed cannot see a swapped rank/nranks/id.
extern int g_recorderIdCalls;
extern int g_recorderLastIdCall;  // rcclCall_t as int, so an includer needs no recorder.h
extern ncclUniqueId* g_recorderLastId;
extern int g_recorderLastRank;
extern int g_recorderLastNranks;

// What the collective and send/recv-based overloads were handed (type is rcclCall_t as int).
struct RecorderCall {
  int type;
  const void* sendbuff;
  void* recvbuff;
  size_t count;
  int root;
  const size_t* sendcounts;
  const size_t* rdispls;
};
// Both overloads route here; the default returns g_recorderResult.
extern std::function<ncclResult_t(const RecorderCall&)> g_recorderCall;
// Every Recorder::skip argument, in call order.
extern std::vector<bool> g_recorderSkips;

void ResetRecorderFakes();

#endif  // RCCL_TEST_HOST_RECORDER_FAKES_H_
