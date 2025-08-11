/*
 * Copyright (C) 2017-2018 Advanced Micro Devices, Inc. All Rights Reserved.
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

#ifndef __KFD_EVICT_TEST__H__
#define __KFD_EVICT_TEST__H__

#include <string>
#include <vector>
#include "KFDMultiProcessTest.hpp"
#include "PM4Queue.hpp"

// @class KFDEvictTest
// Test eviction and restore procedure using two processes
class KFDEvictTest :  public KFDMultiProcessTest {
 public:
    KFDEvictTest(void) {}
    ~KFDEvictTest(void) {}

 protected:
    virtual void SetUp();
    virtual void TearDown();

    void AllocBuffers(bool m_IsParent, HSAuint32 defaultGPUNode, HSAuint32 count, HSAuint64 vramBufSize,
                      std::vector<void *> &pBuffers);
    void FreeBuffers(std::vector<void *> &pBuffers, HSAuint64 vramBufSize);
    void AllocAmdgpuBo(bool m_IsParent, int rn, HSAuint64 vramBufSize, amdgpu_bo_handle &handle);
    void FreeAmdgpuBo(amdgpu_bo_handle handle);
    void AmdgpuCommandSubmissionSdmaNop(int rn, amdgpu_bo_handle handle,
                                           PM4Queue *computeQueue);

 protected:  // Members
    HsaMemFlags     m_Flags;
    void*           m_pBuf;
};

#endif  // __KFD_EVICT_TEST__H__
