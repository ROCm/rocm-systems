// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#ifndef __KFD_MULTINODE_P2P_TEST__H__
#define __KFD_MULTINODE_P2P_TEST__H__

#include <gtest/gtest.h>
#include "KFDBaseComponentTest.hpp"
#include "KFDRemoteTestInfra.hpp"
#include <sys/types.h>
#include <string>


class KFDMultiNodeP2PTest : public KFDBaseComponentTest {
 public:
    KFDMultiNodeP2PTest() {}
    ~KFDMultiNodeP2PTest() {}

 protected:
    virtual void SetUp() override;
    virtual void TearDown() override;

 protected:  // Members
    KFDRemoteTestInfra::RemoteSession m_RemoteSession;
    // Remote binary path for SSH child launch, initialized in SetUp.
    std::string m_RemoteBinaryPath;
    // Timeout used by handshake protocol (milliseconds).
    static constexpr int HANDSHAKE_TIMEOUT_MS = 15000;
};

#endif  // __KFD_MULTINODE_P2P_TEST__H__
