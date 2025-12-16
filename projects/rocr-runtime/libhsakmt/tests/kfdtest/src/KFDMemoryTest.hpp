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

#include "KFDBaseComponentTest.hpp"

#ifndef __KFD_MEMORY_TEST__H__
#define __KFD_MEMORY_TEST__H__

/* @class KFDTopologyTest
 * This class has no additional features to KFDBaseComponentTest
 * The separation was made so we are able to group all memory tests together
 */
class KFDMemoryTest :  public KFDBaseComponentTest {
 public:
    KFDMemoryTest(void) {}
    ~KFDMemoryTest(void) {}

    void MMapLarge(int gpuNode);
    void MapUnmapToNodes(int gpuNode);
    void MapMemoryToGPU(int gpuNode);
    void MemoryAllocAll(int gpuNode);
    void AccessPPRMem(int gpuNode);
    void MemoryRegister(int gpuNode);
    void MemoryRegisterSamePtr(int gpuNode);
    void FlatScratchAccess(int gpuNode);
    void GetTileConfigTest(int gpuNode);
    void LargestSysBufferTest(int gpuNode);
    void LargestVramBufferTest(int gpuNode);
    void BigSysBufferStressTest(int gpuNode);
    void MMBench(int gpuNode);
    void QueryPointerInfo(int gpuNode);
    void PtraceAccess(int gpuNode);
    void PtraceAccessInvisibleVram(int gpuNode);
    void SignalHandling(int gpuNode);
    void CheckZeroInitializationSysMem(int gpuNode);
    void MMBandWidth(int gpuNode);
    void HostHdpFlush(int gpuNode);
    void DeviceHdpFlush(int gpuNode);
    void CacheInvalidateOnSdmaWrite(int gpuNode);
    void CacheInvalidateOnCPUWrite(int gpuNode);
    void CacheInvalidateOnRemoteWrite(int gpuNode);
    void VramCacheCoherenceWithRemoteGPU(int gpuNode);
    void VramCacheCoherenceWithCPU(int gpuNode);
    void SramCacheCoherenceWithGPU(int gpuNode);
    void ExportDMABufTest(int gpuNode);
    void VA_VRAM_Only_AllocTest(int gpuNode);
      

 protected:
    virtual void SetUp();
    virtual void TearDown();

 protected:
    void SearchLargestBuffer(int allocNode, const HsaMemFlags &memFlags,
                                            HSAuint64 highMB, int nodeToMap,
                                            HSAuint64 *lastSizeMB);
    void AcquireReleaseTestRunCPU(HSAuint32 acquireNode, bool scalar);
    void AcquireReleaseTestRun(HSAuint32 acquireNode, HSAuint32 releaseNode,
                                          bool localToRemote, bool scalar);
    void AcquireReleaseTest(bool withinGPU, bool localToRemote, bool scalar);
};

#endif  // __KFD_MEMORY_TEST__H__
