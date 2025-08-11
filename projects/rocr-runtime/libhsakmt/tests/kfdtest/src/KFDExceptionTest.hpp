/*
 * Copyright (C) 2014-2018 Advanced Micro Devices, Inc. All Rights Reserved.
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

#ifndef __KFD_EXCEPTION_TEST__H__
#define __KFD_EXCEPTION_TEST__H__

#include <gtest/gtest.h>

#include "KFDBaseComponentTest.hpp"

class KFDExceptionTest : public KFDBaseComponentTest {
 public:
    KFDExceptionTest() : m_ChildPid(-1) {
        /* Because there could be early return before m_ChildPid is set
         * by fork(), we should initialize m_ChildPid to a non-zero value
         * to avoid possible exit of the main process.
         */
    }

    ~KFDExceptionTest() {
        /* exit() is necessary for the child process. Otherwise when the
         * child process finishes, gtest assumes the test has finished and
         * starts the next test while the parent is still active.
         */
        if (m_ChildPid == 0) {
            if (!m_ChildStatus && HasFatalFailure())
                m_ChildStatus = HSAKMT_STATUS_ERROR;
            exit(m_ChildStatus);
        }
    }

    friend void AddressFault(KFDTEST_PARAMETERS* pTestParamters);
    friend void PermissionFault(KFDTEST_PARAMETERS* pTestParamters);
    friend void PermissionFaultUserPointer(KFDTEST_PARAMETERS* pTestParamters);
    friend void FaultStorm(KFDTEST_PARAMETERS* pTestParamters);
    friend void SdmaQueueException(KFDTEST_PARAMETERS* pTestParamters);

 protected:
    virtual void SetUp();
    virtual void TearDown();

    void TestMemoryException(int gpuNode, HSAuint64 pSrc, HSAuint64 pDst,
                             unsigned int dimX = 1, unsigned int dimY = 1,
                             unsigned int dimZ = 1);
    void TestSdmaException(int gpuNode, void *pDst);

 protected:  // Members
    pid_t m_ChildPid;
    HSAKMT_STATUS m_ChildStatus;
};

#endif  // __KFD_EXCEPTION_TEST__H__
