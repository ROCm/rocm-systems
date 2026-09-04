/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// rccl::Recorder (src/recorder.cc) faked once for every host-only microtest
// binary. Previously the ctor/dtor/instance() triple was copied verbatim into
// comm_fakes.cc, init_fakes.cc and enqueue_fakes.cc, differing only in which
// record() overloads each target happened to reference.

#ifndef RCCL_TEST_HOST_RECORDER_FAKES_H_
#define RCCL_TEST_HOST_RECORDER_FAKES_H_

#include "nccl.h"

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

void ResetRecorderFakes();

#endif  // RCCL_TEST_HOST_RECORDER_FAKES_H_
