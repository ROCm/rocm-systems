/*
 * Copyright (C) 2020 Advanced Micro Devices, Inc. All Rights Reserved.
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

#ifndef __KFD_SVM_EVICT_TEST__H__
#define __KFD_SVM_EVICT_TEST__H__

#include <string>
#include <vector>
#include "KFDLocalMemoryTest.hpp"
#include "KFDBaseComponentTest.hpp"

// @class KFDEvictTest
// Test eviction and restore procedure using two processes
class KFDSVMEvictTest : public KFDLocalMemoryTest,
                        public ::testing::WithParamInterface<int> {
 public:
    KFDSVMEvictTest(void): m_ChildStatus(HSAKMT_STATUS_ERROR), m_IsParent(true) {}

    ~KFDSVMEvictTest(void) {
        if (!m_IsParent) {
            /* child process has to exit
             * otherwise gtest will continue other tests
             */
            exit(m_ChildStatus);
        }

        try {
            WaitChildProcesses();
        } catch (...) {}
    }

 protected:
    virtual void SetUp();
    virtual void TearDown();

 protected:
    std::string CreateShader();
    void AllocBuffers(HSAuint32 defaultGPUNode, HSAuint32 count, HSAuint64 vramBufSize,
                    std::vector<void *> &pBuffers, HSAuint32 Granularity);
    void FreeBuffers(std::vector<void *> &pBuffers, HSAuint64 vramBufSize);
    void ForkChildProcesses(int nprocesses);
    void WaitChildProcesses();
    HSAint32 GetBufferCounter(HSAuint64 vramSize, HSAuint64 vramBufSize);
    HSAint64 GetBufferSize(HSAuint64 vramSize, HSAuint32 count,
                           HSAint32 xnack_enable);

 protected:  // members
    std::string     m_psName;
    std::vector<pid_t> m_ChildPids;
    HSA_SVM_FLAGS   m_Flags;
    void*           m_pBuf;
    HSAKMT_STATUS   m_ChildStatus;
    bool            m_IsParent;
};

#endif  // __KFD_SVM_EVICT_TEST__H__
