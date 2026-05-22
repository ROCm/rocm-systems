/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Per-test controllable seams for the fakes layer.
//
// See README.md, "Adding more controllable seams". Tests install per-test
// behaviour by overwriting one of these std::function hooks in a fixture's
// SetUp(), and ResetP2pFakes() in TearDown() restores defaults so tests
// don't contaminate each other.

#pragma once

#include <functional>

#include "nccl.h"
#include "strongstream.h"

// ncclStrongStreamAcquire: by default returns ncclSuccess with *stream=nullptr
// (matching the stub's prior behaviour). Tests that need to exercise the
// strong-stream block's failure paths can install a hook that returns an
// error code; tests that want to count entries can install a counting hook.
extern std::function<ncclResult_t(struct ncclCudaGraph,
                                  struct ncclStrongStream*,
                                  bool,
                                  hipStream_t*)>
    g_strongStreamAcquire;

// Restore every hook in this header to its default. Call from fixture
// TearDown().
void ResetP2pFakes();
