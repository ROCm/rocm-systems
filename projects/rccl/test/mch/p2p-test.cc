/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Unit tests for src/transport/p2p-scratch.cc.
//
// Unlike rccl-UnitTests, this binary does NOT link librccl.so. The goal is to
// eventually compile p2p-scratch.cc directly into the test binary so that
// internal symbols (e.g. the static ipcRegisterBuffer) can be exercised with
// link-time-substituted fakes for the HIP driver API and the NCCL proxy layer.
//
// For now this file just stands up the test target with a trivial smoke test;
// real coverage of ipcRegisterBuffer will be added once the stub layer for
// ncclProxyCallBlocking / ncclProxyConnect / the HIP driver seam is in place.

#include <gtest/gtest.h>

// Pull in the hipified copy of p2p.cc (cudaXxx -> hipXxx rewrites already
// applied by the hipify pass that runs as part of the main RCCL build).
// P2P_CC_PATH is defined by this target's CMakeLists.txt as a string
// literal pointing at ${PROJECT_BINARY_DIR}/hipify/src/transport/p2p.cc.
#include P2P_CC_PATH

TEST(P2pScratchSmoke, BinaryLinksAndRuns)
{
    EXPECT_EQ(1 + 1, 2);
}
