// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "KFDMultiNodeP2PTest.hpp"
#include "KFDTestUtil.hpp"
#include "KFDRemoteTestInfra.hpp"
#include <errno.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <limits.h>

#include <algorithm>
#include <vector>

extern bool g_IsRemoteChildProcess;
extern std::string g_RemoteHost;

void KFDMultiNodeP2PTest::SetUp() {
    ROUTINE_START
    KFDBaseComponentTest::SetUp();
    KFDRemoteTestInfra::InitSession(m_RemoteSession);

    /* HsaDefaultGPUNode() silently falls back to the first GPU for an unknown
     * --node, which on the remote side would run the test on a GPU the user
     * did not ask for.
     */
    if (g_IsRemoteChildProcess && g_TestNodeId >= 0) {
        const std::vector<int> gpuNodes = m_NodeInfo.GetNodesWithGPU();
        ASSERT_NE(gpuNodes.end(), std::find(gpuNodes.begin(), gpuNodes.end(), g_TestNodeId))
            << "GPU node " << g_TestNodeId << " requested with --node does not exist on this host";
    }

    /* An explicit override always wins, otherwise run the remote test from the
     * same path we are executing the local kfdtest binary from. This assumes
     * the local and remote machines have kfdtest installed at the same path.
     */
    const char *remoteBin = getenv("KFDTEST_REMOTE_BIN");
    if (remoteBin && remoteBin[0]) {
        m_RemoteBinaryPath = remoteBin;
    } else {
        char exePath[PATH_MAX];
        ssize_t len = readlink("/proc/self/exe", exePath, sizeof(exePath) - 1);
        if (len > 0) {
            exePath[len] = '\0';
            m_RemoteBinaryPath = exePath;
        } else if (!g_RemoteHost.empty()) {
            FAIL() << "Cannot resolve the local kfdtest path via /proc/self/exe ("
                   << strerror(errno) << "); set KFDTEST_REMOTE_BIN to the kfdtest "
                   << "path on " << g_RemoteHost;
        }
    }
    ROUTINE_END
}

void KFDMultiNodeP2PTest::TearDown() {
    ROUTINE_START
    KFDRemoteTestInfra::CleanupSession(m_RemoteSession);
    KFDBaseComponentTest::TearDown();
    ROUTINE_END
}
