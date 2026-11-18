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
    KFDExceptionTest() : m_ChildStatus(HSAKMT_STATUS_ERROR) {
        /* m_ChildStatus is only assigned once TestMemoryException() or
         * TestSdmaException() gets as far as queue.Create(). Seed it to
         * an error so a child that bails out before then fails closed
         * rather than exiting on stack garbage, matching KFDSVMEvictTest
         * and KFDMultiProcessTest.
         */
    }

    /* exit() is necessary for the child process. Otherwise when the
     * child process finishes, gtest assumes the test has finished and
     * starts the next test while the parent is still active.
     *
     * exit() does not unwind, so the check has to happen here rather
     * than in the destructor.
     */
    void ExitChild() {
        if (!m_ChildStatus && HasFatalFailure())
            m_ChildStatus = HSAKMT_STATUS_ERROR;
        exit(m_ChildStatus);
    }

    void AddressFault(int gpuNode);
    void PermissionFault(int gpuNode);
    void PermissionFaultUserPointer(int gpuNode);
    void FaultStorm(int gpuNode);
    void SdmaQueueException(int gpuNode);

 protected:
    virtual void SetUp();
    virtual void TearDown();

    void TestMemoryException(int gpuNode, HSAuint64 pSrc, HSAuint64 pDst,
                             unsigned int dimX = 1, unsigned int dimY = 1,
                             unsigned int dimZ = 1);
    void TestSdmaException(int gpuNode, void *pDst);

 protected:  // Members
    HSAKMT_STATUS m_ChildStatus;
};

#endif  // __KFD_EXCEPTION_TEST__H__
