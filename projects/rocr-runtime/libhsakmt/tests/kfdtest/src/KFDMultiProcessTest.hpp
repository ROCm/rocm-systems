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

#ifndef __KFD_MULTI_PROCESS_TEST__H__
#define __KFD_MULTI_PROCESS_TEST__H__

#include <string>
#include <vector>
#include "KFDBaseComponentTest.hpp"

// @class KFDMultiProcessTest
// Base class for tests forking multiple child processes
class KFDMultiProcessTest :  public KFDBaseComponentTest {
 public:
    KFDMultiProcessTest(void) {
        for ( int i = 0; i < MAX_GPU; i++) {
            m_ChildStatus[i] = HSAKMT_STATUS_ERROR;
            m_IsParent[i] = true;
        }
    }

    ~KFDMultiProcessTest(void) {
        for (int i = 0; i < MAX_GPU; i++) {
            if (!m_IsParent[i]) {
                /* Child process has to exit
                 * otherwise gtest will continue other tests
                */
                exit(m_ChildStatus[i]);
            }
        }

        try {
            const std::vector<int> gpuNodes = m_NodeInfo.GetNodesWithGPU();
            int gpu_node;
            /* parent porcess waits all its child processes on each gpu */
            for (int i = 0; i < std::min((int)gpuNodes.size(), MAX_GPU); i++) {
                gpu_node = gpuNodes.at(i);
                WaitChildProcesses(gpu_node);
            }
       } catch (...) {}

    }

 protected:
    void ForkChildProcesses(unsigned int nodeId, int nprocesses);
    void WaitChildProcesses(unsigned int nodeId);

 protected:  // Members
    std::string     m_psName[MAX_GPU];
    int             m_ProcessIndex[MAX_GPU];
    std::vector<pid_t> m_ChildPids[MAX_GPU];
    HSAKMT_STATUS   m_ChildStatus[MAX_GPU];
    bool            m_IsParent[MAX_GPU];
};

#endif  // __KFD_MULTI_PROCESS_TEST__H__
