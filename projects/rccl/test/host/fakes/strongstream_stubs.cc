/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Fail-loud stub floor for src/misc/strongstream.cc (CUDA-graph capture and
// strong-stream ordering), shared by the host-only microtest binaries.
//
// ncclStrongStreamAcquire / ncclStrongStreamRelease stay in nccl_fakes.cc: they
// are controllable seams carrying ASSERT_HOOK_MATCHES_PROD drift assertions, and
// moving those is a larger change than this one. Everything else this TU owns is
// here, so the subsystem now lives in two files rather than four.

#include "nccl.h"
#include "strongstream.h"

#include "fail_loud.h"

ncclResult_t ncclCudaGetCapturingGraph(struct ncclCudaGraph*, hipStream_t, int) {
  FailLoudUnfaked("strongstream_stubs", "ncclCudaGetCapturingGraph");
}
ncclResult_t ncclCudaGraphAddDestructor(struct ncclCudaGraph, hipHostFn_t, void*) {
  FailLoudUnfaked("strongstream_stubs", "ncclCudaGraphAddDestructor");
}
ncclResult_t ncclStreamAdvanceToEvent(struct ncclCudaGraph, hipStream_t, hipEvent_t) {
  FailLoudUnfaked("strongstream_stubs", "ncclStreamAdvanceToEvent");
}
ncclResult_t ncclStrongStreamAcquiredWorkStream(struct ncclCudaGraph, struct ncclStrongStream*,
                                                bool, hipStream_t*) {
  FailLoudUnfaked("strongstream_stubs", "ncclStrongStreamAcquiredWorkStream");
}
// Benign teardown: commFree reaches this on a happy-path destroy.
ncclResult_t ncclStrongStreamDestruct(struct ncclStrongStream*) { return ncclSuccess; }
