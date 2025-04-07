/*
 * Copyright (C) 2019 Advanced Micro Devices, Inc. All Rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
 * OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 *
 */

#include "KFDMultiProcessTest.hpp"

void KFDMultiProcessTest::ForkChildProcesses(unsigned int nodeId, int nprocesses) {
    int i;
    int gpuIndex = m_NodeInfo.HsaGPUindexFromGpuNode(nodeId);

    for (i = 0; i < nprocesses - 1; ++i) {
        pid_t pid = fork();
        ASSERT_GE(pid, 0);

        if (pid == 0) {
            /* Child process */
            /* Cleanup file descriptors copied from parent process
             * then call SetUp->hsaKmtOpenKFD to create new process
             */
            m_psName[gpuIndex] = "Child Test process " + std::to_string(i) +
                          " on gpuNode: " + std::to_string(gpuIndex) + " ";
            TearDown();
            SetUp();
            m_ChildPids[gpuIndex].clear();
            m_IsParent[gpuIndex] = false;
            m_ProcessIndex[gpuIndex] = i;
            return;
        }

        /* Parent process */
        m_ChildPids[gpuIndex].push_back(pid);
    }

    m_psName[gpuIndex] = "Parent Test process " + std::to_string(i) +
                        " on gpuNode: " + std::to_string(gpuIndex) + " ";
    m_ProcessIndex[gpuIndex] = i;
}

void KFDMultiProcessTest::WaitChildProcesses(unsigned int nodeId) {

    int gpuIndex = m_NodeInfo.HsaGPUindexFromGpuNode(nodeId);

    if (m_IsParent[gpuIndex]) {
        /* Only run by parent process */
        int childStatus;
        int childExitOkNum = 0;
        int size = m_ChildPids[gpuIndex].size();

        for (HSAuint32 i = 0; i < size; i++) {
            pid_t pid = m_ChildPids[gpuIndex].front();

            waitpid(pid, &childStatus, 0);
            if (WIFEXITED(childStatus) == 1 && WEXITSTATUS(childStatus) == 0)
                childExitOkNum++;

            m_ChildPids[gpuIndex].erase(m_ChildPids[gpuIndex].begin());
        }

        EXPECT_EQ(childExitOkNum, size);
    }

    /* Child process or parent process finished successfully */
    m_ChildStatus[gpuIndex] = HSAKMT_STATUS_SUCCESS;
}
