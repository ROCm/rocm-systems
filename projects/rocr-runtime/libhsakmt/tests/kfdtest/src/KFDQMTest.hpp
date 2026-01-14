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

#ifndef __KFD_QCM_TEST__H__
#define __KFD_QCM_TEST__H__

#include <gtest/gtest.h>

#include "PM4Queue.hpp"
#include "KFDBaseComponentTest.hpp"
#include "Dispatch.hpp"

/*
 * Used by ExtendedCuMasking test case to pass GPU configuration information to helper functions.
 */
typedef struct {
    uint32_t numDwords;
    uint32_t numBits;
    uint32_t numSEs;
    uint32_t numSAperSE;
    uint32_t numWGPperSA;
    uint32_t *pInactiveMask;
} mask_config_t;

/*
 * Used by ExtendedCuMasking test case.
 *
 * Struct is hardware-dependent and fields are layed out same way as hardware register.
 *
 */
typedef union {
    uint32_t data;
    // Fields needed from HW_ID1 (format same for GFX11 and GFX12)
    struct {
        unsigned     :10;
        unsigned wgp : 4;
        unsigned     : 2;
        unsigned  sa : 1;
        unsigned     : 1;
        unsigned  se : 3;
        unsigned     :11;
    };
} out_data_t;


class KFDQMTest : public KFDBaseComponentTest {
 public:
    KFDQMTest() {}

    ~KFDQMTest() {}

    void CreateDestroyCpQueue(int gpuNode);
    void SubmitNopCpQueue(int gpuNode);
    void SubmitPacketCpQueue(int gpuNode);
    void AllCpQueues(int gpuNode);
    void CreateDestroySdmaQueue(int gpuNode);
    void SubmitNopSdmaQueue(int gpuNode);
    void SubmitPacketSdmaQueue(int gpuNode);
    void AllSdmaQueues(int gpuNode);
    void AllXgmiSdmaQueues(int gpuNode);
    void AllQueues(int gpuNode);
    void SdmaConcurrentCopies(int gpuNode);
    void DisableCpQueueByUpdateWithNullAddress(int gpuNode);
    void DisableSdmaQueueByUpdateWithNullAddress(int gpuNode);
    void DisableCpQueueByUpdateWithZeroPercentage(int gpuNode);
    void CreateQueueStressSingleThreaded(int gpuNode);
    void OverSubscribeCpQueues(int gpuNode);
    void BasicCuMaskingLinear(int gpuNode);
    void extendedCuMasking(int gpuNode);
    void BasicCuMaskingEven(int gpuNode);
    void QueuePriorityOnDifferentPipe(int gpuNode);
    void QueuePriorityOnSamePipe(int gpuNode);
    void EmptyDispatch(int gpuNode);
    void SimpleWriteDispatch(int gpuNode);
    void MultipleCpQueuesStressDispatch(int gpuNode);
    void CpuWriteCoherence(int gpuNode);
    void CreateAqlCpQueue(int gpuNode);
    void QueueLatency(int gpuNode);
    void CpQueueWraparound(int gpuNode);
    void SdmaQueueWraparound(int gpuNode);
    void Atomics(int gpuNode);
    void PM4EventInterrupt(int gpuNode);
    void SdmaEventInterrupt(int gpuNode);
    void GPUDoorbellWrite(int gpuNode);
    void GpuMemCopyTest(int gpuNode);

 protected:
    virtual void SetUp();
    virtual void TearDown();
    void SyncDispatch(const HsaMemoryBuffer& isaBuffer, void* arg0, void* arg1, int node = -1);
    HSAint64 TimeConsumedwithCUMask(int node, uint32_t *mask, uint32_t mask_count);
    HSAint64 GetAverageTimeConsumedwithCUMask(int node, uint32_t *mask, uint32_t mask_count, int iterations);
    void testQueuePriority(int gpuNode, bool isSamePipe);

 protected:  // Members
    /* Acceptable performance for CU Masking should be within 5% of linearly-predicted performance */
    const double CuVariance = 0.15;
    const double CuNegVariance = 1.0 - CuVariance;
    const double CuPosVariance = 1.0 + CuVariance;
};

#endif  // __KFD_QCM_TEST__H__
