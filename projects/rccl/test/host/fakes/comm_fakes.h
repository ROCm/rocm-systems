/*************************************************************************
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Reusable fakes for communicator-lifecycle symbols that are real in the
// init unit under test (init.cc / utils.cc) but must be faked for any other
// micro-test unit that instantiates a comm or group. Kept out of the shared
// nccl_fakes.cc precisely because the init binary compiles the real
// definitions and would otherwise get a duplicate symbol.
//
// The one symbol worth controlling from a test -- ncclCommSetAsyncError -- is
// exposed as a std::function seam; tests install per-test behaviour from a
// fixture and call ResetCommFakes() in TearDown.

#ifndef RCCL_TEST_HOST_FAKES_COMM_FAKES_H_
#define RCCL_TEST_HOST_FAKES_COMM_FAKES_H_

#include <functional>

#include "nccl.h"

struct ncclComm;

// Controllable seam: the async-error state a communicator is moved to. Default
// returns ncclSuccess; a test observes/injects by overwriting the hook.
extern std::function<ncclResult_t(struct ncclComm*, ncclResult_t)> g_commSetAsyncError;

// Restore every comm_fakes seam to its default. Call from a fixture TearDown.
void ResetCommFakes();

#endif  // RCCL_TEST_HOST_FAKES_COMM_FAKES_H_
