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

#include <vector>
#include <string>
#include "KFDEvictTest.hpp"
#include "PM4Queue.hpp"
#include "PM4Packet.hpp"
#include "SDMAPacket.hpp"
#include "SDMAQueue.hpp"
#include "Dispatch.hpp"

#define N_PROCESSES             (2)     /* Number of processes running in parallel, must be at least 2 */
#define ALLOCATE_BUF_SIZE_MB    (64)
#define ALLOCATE_RETRY_TIMES    (3)
#define MAX_WAVEFRONTS          (512)

#define SDMA_NOP  0x0

void KFDEvictTest::SetUp() {
    ROUTINE_START

    KFDBaseComponentTest::SetUp();

    ROUTINE_END
}

void KFDEvictTest::TearDown() {
    ROUTINE_START

    KFDBaseComponentTest::TearDown();

    ROUTINE_END
}

void KFDEvictTest::AllocBuffers(bool m_IsParent, HSAuint32 defaultGPUNode, HSAuint32 count, HSAuint64 vramBufSize,
                                std::vector<void *> &pBuffers) {
    HSAuint64   totalMB;

    totalMB = N_PROCESSES*count*(vramBufSize>>20);
    if (m_IsParent) {
        LOG() << "Allocating " << N_PROCESSES << "*" << count << "*" << (vramBufSize>>20) << "(="
              << totalMB << ")MB VRAM in KFD" << std::endl;
    }

    HsaMemMapFlags mapFlags = {0};
    HSAKMT_STATUS ret;
    HSAuint32 retry = 0;

    m_Flags.Value = 0;
    m_Flags.ui32.PageSize = HSA_PAGE_SIZE_4KB;
    m_Flags.ui32.HostAccess = 0;
    m_Flags.ui32.NonPaged = 1;

    for (HSAuint32 i = 0; i < count; ) {
        ret = hsaKmtAllocMemory(defaultGPUNode, vramBufSize, m_Flags, &m_pBuf);
        if (ret == HSAKMT_STATUS_SUCCESS) {
            if (hsakmt_is_dgpu()) {
                if (hsaKmtMapMemoryToGPUNodes(m_pBuf, vramBufSize, NULL,
                       mapFlags, 1, reinterpret_cast<HSAuint32 *>(&defaultGPUNode)) == HSAKMT_STATUS_ERROR) {
                    EXPECT_SUCCESS(hsaKmtFreeMemory(m_pBuf, vramBufSize));
                    LOG() << "Map failed for " << i << "/" << count << " buffer. Retrying allocation" << std::endl;
                    goto retry;
                }
            }
            pBuffers.push_back(m_pBuf);

            i++;
            retry = 0;
            continue;
        }
retry:
        if (retry++ > ALLOCATE_RETRY_TIMES) {
            break;
        }

        /* Wait for 1 second to try allocate again */
        sleep(1);
    }
}

void KFDEvictTest::FreeBuffers(std::vector<void *> &pBuffers, HSAuint64 vramBufSize) {
    for (HSAuint32 i = 0; i < pBuffers.size(); i++) {
        m_pBuf = pBuffers[i];
        if (m_pBuf != NULL) {
            if (hsakmt_is_dgpu())
                EXPECT_SUCCESS(hsaKmtUnmapMemoryToGPU(m_pBuf));
            EXPECT_SUCCESS(hsaKmtFreeMemory(m_pBuf, vramBufSize));
        }
    }
}

void KFDEvictTest::AllocAmdgpuBo(bool m_IsParent, int rn, HSAuint64 vramBufSize, amdgpu_bo_handle &handle) {
    struct amdgpu_bo_alloc_request alloc;

    alloc.alloc_size = vramBufSize / N_PROCESSES;
    alloc.phys_alignment = PAGE_SIZE;
    alloc.preferred_heap = AMDGPU_GEM_DOMAIN_VRAM;
    alloc.flags = AMDGPU_GEM_CREATE_VRAM_CLEARED;

    if (m_IsParent) {
        LOG() << "Allocating " << N_PROCESSES << "*" << (vramBufSize >> 20) / N_PROCESSES << "(="
              << (vramBufSize >> 20)  << ")MB VRAM in GFX" << std::endl;
    }
    ASSERT_EQ(0, amdgpu_bo_alloc(m_RenderNodes[rn].device_handle, &alloc, &handle));
}

void KFDEvictTest::FreeAmdgpuBo(amdgpu_bo_handle handle) {
    ASSERT_EQ(0, amdgpu_bo_free(handle));
}

static int amdgpu_bo_alloc_and_map(amdgpu_device_handle dev, unsigned size,
                                   unsigned alignment, unsigned heap, uint64_t flags,
                                   amdgpu_bo_handle *bo, void **cpu, uint64_t *mc_address,
                                   amdgpu_va_handle *va_handle) {
    struct amdgpu_bo_alloc_request request = {};
    amdgpu_bo_handle buf_handle;
    amdgpu_va_handle handle;
    uint64_t vmc_addr;
    int r;

    request.alloc_size = size;
    request.phys_alignment = alignment;
    request.preferred_heap = heap;
    request.flags = flags;

    r = amdgpu_bo_alloc(dev, &request, &buf_handle);
    if (r)
        return r;

    r = amdgpu_va_range_alloc(dev,
                  amdgpu_gpu_va_range_general,
                  size, alignment, 0, &vmc_addr,
                  &handle, 0);
    if (r)
        goto error_va_alloc;

    r = amdgpu_bo_va_op(buf_handle, 0, size, vmc_addr, 0, AMDGPU_VA_OP_MAP);
    if (r)
        goto error_va_map;

    r = amdgpu_bo_cpu_map(buf_handle, cpu);
    if (r)
        goto error_cpu_map;

    *bo = buf_handle;
    *mc_address = vmc_addr;
    *va_handle = handle;

    return 0;

error_cpu_map:
    amdgpu_bo_cpu_unmap(buf_handle);

error_va_map:
    amdgpu_bo_va_op(buf_handle, 0, size, vmc_addr, 0, AMDGPU_VA_OP_UNMAP);

error_va_alloc:
    amdgpu_bo_free(buf_handle);
    return r;
}

static inline int amdgpu_bo_unmap_and_free(amdgpu_bo_handle bo, amdgpu_va_handle va_handle,
                                           uint64_t mc_addr, uint64_t size) {
    amdgpu_bo_cpu_unmap(bo);
    amdgpu_bo_va_op(bo, 0, size, mc_addr, 0, AMDGPU_VA_OP_UNMAP);
    amdgpu_va_range_free(va_handle);
    amdgpu_bo_free(bo);

    return 0;
}

static inline int amdgpu_get_bo_list(amdgpu_device_handle dev, amdgpu_bo_handle bo1,
                                     amdgpu_bo_handle bo2, amdgpu_bo_list_handle *list) {
    amdgpu_bo_handle resources[] = {bo1, bo2};

    return amdgpu_bo_list_create(dev, bo2 ? 2 : 1, resources, NULL, list);
}

void KFDEvictTest::AmdgpuCommandSubmissionSdmaNop(int rn, amdgpu_bo_handle handle,
                                                     PM4Queue *computeQueue = NULL) {
    amdgpu_context_handle contextHandle;
    amdgpu_bo_handle ibResultHandle;
    void *ibResultCpu;
    uint64_t ibResultMcAddress;
    struct amdgpu_cs_request ibsRequest;
    struct amdgpu_cs_ib_info ibInfo;
    struct amdgpu_cs_fence fenceStatus;
    amdgpu_bo_list_handle boList;
    amdgpu_va_handle vaHandle;
    uint32_t *ptr;
    uint32_t expired;
    unsigned failCount = 0;

    ASSERT_EQ(0, amdgpu_cs_ctx_create(m_RenderNodes[rn].device_handle, &contextHandle));

    ASSERT_EQ(0, amdgpu_bo_alloc_and_map(m_RenderNodes[rn].device_handle,
        PAGE_SIZE, PAGE_SIZE,
        AMDGPU_GEM_DOMAIN_GTT, 0,
        &ibResultHandle, &ibResultCpu,
        &ibResultMcAddress, &vaHandle));

    ASSERT_EQ(0, amdgpu_get_bo_list(m_RenderNodes[rn].device_handle, ibResultHandle, handle,
        &boList));

    /* Fill Nop cammands in IB */
    ptr = reinterpret_cast<uint32_t *>(ibResultCpu);
    for (int i = 0; i < 16; i++)
        ptr[i] = SDMA_NOP;

    memset(&ibInfo, 0, sizeof(struct amdgpu_cs_ib_info));
    ibInfo.ib_mc_address = ibResultMcAddress;
    ibInfo.size = 16;

    memset(&ibsRequest, 0, sizeof(struct amdgpu_cs_request));
    ibsRequest.ip_type = AMDGPU_HW_IP_DMA;
    ibsRequest.ring = 0;
    ibsRequest.number_of_ibs = 1;
    ibsRequest.ibs = &ibInfo;
    ibsRequest.resources = boList;
    ibsRequest.fence_info.handle = NULL;

    memset(&fenceStatus, 0, sizeof(struct amdgpu_cs_fence));
    for (int i = 0; i < 100; i++) {
        int r = amdgpu_cs_submit(contextHandle, 0, &ibsRequest, 1);

        Delay(50);
        if (r) {
            failCount++;
            ASSERT_LE(failCount, 2);
            continue;
        }

        fenceStatus.context = contextHandle;
        fenceStatus.ip_type = AMDGPU_HW_IP_DMA;
        fenceStatus.ip_instance = 0;
        fenceStatus.ring = 0;
        fenceStatus.fence = ibsRequest.seq_no;

        EXPECT_EQ(0, amdgpu_cs_query_fence_status(&fenceStatus,
                                                  g_TestTimeOut*1000000,
                                                  0, &expired));
        if (!expired)
            WARN() << "CS did not signal completion" << std::endl;

        /* If a compute queue is given, submit a short compute job
         * every 16 loops (about once a second). If the process was
         * evicted, restore can take quite long.
         */
        if (computeQueue && (i & 0xf) == 0) {
            computeQueue->PlaceAndSubmitPacket(PM4NopPacket());
            computeQueue->Wait4PacketConsumption(NULL, 10000);
        }
    }

    EXPECT_EQ(0, amdgpu_bo_list_destroy(boList));

    EXPECT_EQ(0, amdgpu_bo_unmap_and_free(ibResultHandle, vaHandle,
        ibResultMcAddress, PAGE_SIZE));

    EXPECT_EQ(0, amdgpu_cs_ctx_free(contextHandle));
}

/* Evict and restore procedure basic test
 *
 * Use N_PROCESSES processes to allocate vram buf size larger than total vram size
 *
 * ALLOCATE_BUF_SIZE_MB buf allocation size
 *
 * buf is equal to (vramSizeMB / (vramBufSizeMB * N_PROCESSES) ) + 8
 * Total vram all processes allocated: 8GB for 4GB Fiji, and 20GB for 16GB Vega10
 *
 * Eviction and restore will happen many times:
 * ttm will evict buffers of another process if there is not enough free vram
 * process restore will evict buffers of another process
 *
 * Sometimes the allocation may fail (maybe that is normal)
 * ALLOCATE_RETRY_TIMES max retry times to allocate
 *
 * This is basic test with no queue, so vram is not used by the GPU during test
 *
 * TODO:
 *    - Synchronization between the processes, so they know for sure when
 *        they are done allocating memory
 */
TEST_F(KFDEvictTest, BasicTest) {
    TEST_REQUIRE_ENV_CAPABILITIES(ENVCAPS_64BITLINUX);
    TEST_START(TESTPROFILE_RUNALL);

    HSAuint32 defaultGPUNode = m_NodeInfo.HsaDefaultGPUNode();
    ASSERT_GE(defaultGPUNode, 0) << "failed to get default GPU Node";
    HSAuint64 vramBufSize = ALLOCATE_BUF_SIZE_MB * 1024 * 1024;

    HSAuint64 vramSize = GetVramSize(defaultGPUNode);
    HSAuint64 sysMemSize = GetSysMemSize();

    int gpuIndex = m_NodeInfo.HsaGPUindexFromGpuNode(defaultGPUNode);
    const HsaNodeProperties *pNodeProperties = m_NodeInfo.HsaDefaultGPUNodeProperties();

    if (!vramSize) {
        LOG() << "Skipping test: No VRAM found." << std::endl;
        return;
    }

    if (pNodeProperties->Integrated) {
        LOG() << "Skipping test on APU." << std::endl;
        return;
    }

    LOG() << "Found VRAM of " << std::dec << (vramSize >> 20) << "MB" << std::endl;
    LOG() << "Found System RAM of " << std::dec << (sysMemSize >> 20) << "MB" << std::endl;

    // Use 7/8 of VRAM between all processes
    HSAuint64 testSize = vramSize * 7 / 8;
    HSAuint32 count = testSize / (vramBufSize * N_PROCESSES);

    if (count == 0) {
        LOG() << "Skipping test: Not enough system memory available." << std::endl;
        return;
    }

    /* Fork the child processes */
    ForkChildProcesses(defaultGPUNode, N_PROCESSES);

    int rn = FindDRMRenderNode(defaultGPUNode);
    if (rn < 0) {
        LOG() << "Skipping test: Could not find render node for default GPU." << std::endl;
        WaitChildProcesses(defaultGPUNode);
        return;
    }

    std::vector<void *> pBuffers;
    AllocBuffers(m_IsParent[gpuIndex], defaultGPUNode, count, vramBufSize, pBuffers);

    /* Allocate gfx vram size of at most one-fourth system memory */
    HSAuint64 size = sysMemSize / 4 < testSize / 3 ? sysMemSize / 4 : testSize / 3;
    amdgpu_bo_handle handle;
    AllocAmdgpuBo(m_IsParent[gpuIndex], rn, size, handle);

    AmdgpuCommandSubmissionSdmaNop(rn, handle);

    FreeAmdgpuBo(handle);
    LOG() << m_psName[gpuIndex] << "free buffer" << std::endl;
    FreeBuffers(pBuffers, vramBufSize);

    WaitChildProcesses(defaultGPUNode);

    TEST_END
}

/* Evict and restore queue test
 *
 * N_PROCESSES processes read all local buffers in parallel while buffers are evicted and restored
 * If GPU vm page fault happens, then test shader will stop and failed to write specific value
 * at dest buffer. Test will report failed.
 *
 * Steps:
 *    - fork N_PROCESSES processes, each process does the same below
 *    - allocate local buffers, each buffer size is 64MB
 *    - allocate zero initialized host access address buffer and result buffer
 *        address buffer to pass address of local buffers to shader
 *        result buffer to store shader output result
 *    - submit queue to run ReadMemory shader
 *    - shader start m_DimX wavefronts, each wavefront keep reading one local buffer
 *    - notify shader to quit
 *    - check result buffer with specific value to confirm all wavefronts quit normally
 */
TEST_F(KFDEvictTest, QueueTest) {
    TEST_REQUIRE_ENV_CAPABILITIES(ENVCAPS_64BITLINUX);
    TEST_START(TESTPROFILE_RUNALL)

    HSAuint32 defaultGPUNode = m_NodeInfo.HsaDefaultGPUNode();
    ASSERT_GE(defaultGPUNode, 0) << "failed to get default GPU Node";
    unsigned int count = MAX_WAVEFRONTS;

    int gpuIndex = m_NodeInfo.HsaGPUindexFromGpuNode(defaultGPUNode);
    const HsaNodeProperties *pNodeProperties = m_NodeInfo.HsaDefaultGPUNodeProperties();

    /* Skip test for chip if it doesn't have CWSR, which the test depends on */
    if (m_FamilyId < FAMILY_VI || isTonga(pNodeProperties)) {
        LOG() << std::hex << "Skipping test: No CWSR present for family ID 0x" << m_FamilyId << "." << std::endl;
        return;
    }

    if (pNodeProperties->Integrated) {
        LOG() << "Skipping test on APU." << std::endl;
        return;
    }

    HSAuint32 i;
    HSAuint64 vramSize = GetVramSize(defaultGPUNode);
    HSAuint64 sysMemSize = GetSysMemSize();

    if (!vramSize) {
        LOG() << "Skipping test: No VRAM found." << std::endl;
        return;
    }

    LOG() << "Found VRAM of " << std::dec << (vramSize >> 20) << "MB" << std::endl;
    LOG() << "Found System RAM of " << std::dec << (sysMemSize >> 20) << "MB" << std::endl;

    // Use 7/8 of VRAM between all processes
    HSAuint64 testSize = vramSize * 7 / 8;
    HSAuint32 vramBufSize = testSize / (count * N_PROCESSES);
    vramBufSize = (vramBufSize / (1024 * 1024)) * (1024 * 1024);

    if (vramBufSize == 0) {
        LOG() << "Skipping test: Not enough system memory available." << std::endl;
        return;
    }
    /* Assert all buffer address can be stored within one page
     * because only one page host memory srcBuf is allocated
     */
    ASSERT_LE(count, PAGE_SIZE/sizeof(unsigned int *));

    /* Fork the child processes */
    ForkChildProcesses(defaultGPUNode, N_PROCESSES);

    int rn = FindDRMRenderNode(defaultGPUNode);
    if (rn < 0) {
        LOG() << "Skipping test: Could not find render node for default GPU." << std::endl;
        WaitChildProcesses(defaultGPUNode);
        return;
    }

    HsaMemoryBuffer isaBuffer(PAGE_SIZE, defaultGPUNode, true/*zero*/, false/*local*/, true/*exec*/);
    HsaMemoryBuffer addrBuffer(PAGE_SIZE, defaultGPUNode);
    HsaMemoryBuffer resultBuffer(PAGE_SIZE, defaultGPUNode);

    ASSERT_SUCCESS(m_pAsm->RunAssembleBuf(ReadMemoryIsa, isaBuffer.As<char*>()));

    PM4Queue pm4Queue;
    ASSERT_SUCCESS(pm4Queue.Create(defaultGPUNode));

    Dispatch dispatch0(isaBuffer);

    std::vector<void *> pBuffers;
    AllocBuffers(m_IsParent[gpuIndex], defaultGPUNode, count, vramBufSize, pBuffers);

    /* Allocate gfx vram size of at most one-fourth system memory */
    HSAuint64 size = sysMemSize / 4 < testSize / 3 ? sysMemSize / 4 : testSize / 3;
    amdgpu_bo_handle handle;
    AllocAmdgpuBo(m_IsParent[gpuIndex], rn, size, handle);

    unsigned int wavefront_num = pBuffers.size();
    LOG() << m_psName[gpuIndex] << "wavefront number " << wavefront_num << std::endl;

    void **localBufAddr = addrBuffer.As<void **>();
    unsigned int *result = resultBuffer.As<uint32_t *>();

    for (i = 0; i < wavefront_num; i++)
        *(localBufAddr + i) = pBuffers[i];

    for (i = 0; i < wavefront_num; i++)
        *(result + i) = vramBufSize;

    dispatch0.SetArgs(localBufAddr, result);
    dispatch0.SetDim(wavefront_num, 1, 1);
    /* Submit the packet and start shader */
    dispatch0.Submit(pm4Queue);

    AmdgpuCommandSubmissionSdmaNop(rn, handle);

    /* Uncomment this line for debugging */
    // LOG() << m_psName << "notify shader to quit" << std::endl;

    /* Fill address buffer so shader quits */
    addrBuffer.Fill(0x5678);

    /* Wait for shader to finish or timeout if shader has vm page fault */
    EXPECT_EQ(0, dispatch0.SyncWithStatus(g_TestTimeOut * 5));

    EXPECT_SUCCESS(pm4Queue.Destroy());

    FreeAmdgpuBo(handle);

    /* Uncomment this line for debugging */
    // LOG() << m_psName << "free buffer" << std::endl;

    /* Cleanup */
    FreeBuffers(pBuffers, vramBufSize);

    /* Check if all wavefronts finished successfully */
    for (i = 0; i < wavefront_num; i++)
        EXPECT_EQ(0x5678, *(result + i));

    WaitChildProcesses(defaultGPUNode);

    TEST_END
}

/* Evict a queue running in bursts, so that the process has a chance
 * to be idle when restored but the queue needs to resume to perform
 * more work later. This test is designed to stress the idle process
 * eviction optimization in KFD that leaves idle processes evicted
 * until the next time the doorbell page is accessed.
 */
TEST_F(KFDEvictTest, BurstyTest) {
    TEST_REQUIRE_ENV_CAPABILITIES(ENVCAPS_64BITLINUX);
    TEST_START(TESTPROFILE_RUNALL);

    HSAuint32 defaultGPUNode = m_NodeInfo.HsaDefaultGPUNode();
    ASSERT_GE(defaultGPUNode, 0) << "failed to get default GPU Node";
    HSAuint64 vramBufSize = ALLOCATE_BUF_SIZE_MB * 1024 * 1024;

    int gpuIndex = m_NodeInfo.HsaGPUindexFromGpuNode(defaultGPUNode);
    const HsaNodeProperties *pNodeProperties = m_NodeInfo.HsaDefaultGPUNodeProperties();

    if (pNodeProperties->Integrated) {
        LOG() << "Skipping test on APU." << std::endl;
        return;
    }

    HSAuint64 vramSize = GetVramSize(defaultGPUNode);
    HSAuint64 sysMemSize = GetSysMemSize();

    if (!vramSize) {
        LOG() << "Skipping test: No VRAM found." << std::endl;
        return;
    }

    LOG() << "Found VRAM of " << std::dec << (vramSize >> 20) << "MB" << std::endl;
    LOG() << "Found System RAM of " << std::dec << (sysMemSize >> 20) << "MB" << std::endl;

    // Use 7/8 of VRAM between all processes
    HSAuint64 testSize = vramSize * 7 / 8;
    HSAuint32 count = testSize / (vramBufSize * N_PROCESSES);

    if (count == 0) {
        LOG() << "Skipping test: Not enough system memory available." << std::endl;
        return;
    }

    /* Fork the child processes */
    ForkChildProcesses(defaultGPUNode, N_PROCESSES);

    int rn = FindDRMRenderNode(defaultGPUNode);
    if (rn < 0) {
        LOG() << "Skipping test: Could not find render node for default GPU." << std::endl;
        WaitChildProcesses(defaultGPUNode);
        return;
    }

    PM4Queue pm4Queue;
    ASSERT_SUCCESS(pm4Queue.Create(defaultGPUNode));

    std::vector<void *> pBuffers;
    AllocBuffers(m_IsParent[gpuIndex], defaultGPUNode, count, vramBufSize, pBuffers);

    /* Allocate gfx vram size of at most one third system memory */
    HSAuint64 size = sysMemSize / 3 < testSize / 2 ? sysMemSize / 3 : testSize / 2;
    amdgpu_bo_handle handle;
    AllocAmdgpuBo(m_IsParent[gpuIndex], rn, size, handle);

    AmdgpuCommandSubmissionSdmaNop(rn, handle, &pm4Queue);

    FreeAmdgpuBo(handle);
    LOG() << m_psName[gpuIndex] << "free buffer" << std::endl;
    FreeBuffers(pBuffers, vramBufSize);

    EXPECT_SUCCESS(pm4Queue.Destroy());

    WaitChildProcesses(defaultGPUNode);

    TEST_END
}
