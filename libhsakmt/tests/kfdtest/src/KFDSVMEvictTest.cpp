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

#include "KFDSVMEvictTest.hpp"
#include <sys/mman.h>
#include <vector>
#include <string>
#include "PM4Queue.hpp"
#include "PM4Packet.hpp"
#include "SDMAPacket.hpp"
#include "SDMAQueue.hpp"
#include "Dispatch.hpp"

#define N_PROCESSES             (2)     /* number of processes running in parallel, at least 2 */
#define ALLOCATE_BUF_SIZE_MB    (64)
#define ALLOCATE_RETRY_TIMES    (3)
#define MAX_WAVEFRONTS          (512)

void KFDSVMEvictTest::SetUp() {
    ROUTINE_START

    KFDLocalMemoryTest::SetUp();

    SVMSetXNACKMode(GetParam());

    ROUTINE_END
}

void KFDSVMEvictTest::TearDown() {
    ROUTINE_START

    SVMRestoreXNACKMode();

    KFDLocalMemoryTest::TearDown();

    ROUTINE_END
}

HSAint32 KFDSVMEvictTest::GetBufferCounter(HSAuint64 vramSize, HSAuint64 vramBufSize) {
    HSAuint64 vramBufSizeInPages = vramBufSize >> PAGE_SHIFT;
    HSAuint64 sysMemSize = GetSysMemSize();
    HSAuint64 size, sizeInPages;
    HSAuint32 count;

    LOG() << "Found System RAM of " << std::dec << (sysMemSize >> 20) << "MB" << std::endl;

    /* use one third of total system memory for eviction buffer to test
     * limit max allocate size to double of vramSize
     * count is zero if not enough memory for XNACK off case
     */
    size = MIN(sysMemSize / 3, vramSize / 2);
    size += vramSize;

    /* Check if there is enough system memory to pass test for XNACK off
     * KFD system memory limit is 15/16.
     */
    HSAint32 xnack_enable = 0;
    EXPECT_SUCCESS(hsaKmtGetXNACKMode(&xnack_enable));
    if (!xnack_enable && size > (sysMemSize - (sysMemSize >> 4)))
        return 0;

    sizeInPages = size >> PAGE_SHIFT;
    count = sizeInPages / (vramBufSizeInPages * N_PROCESSES);

    return count;
}

HSAint64 KFDSVMEvictTest::GetBufferSize(HSAuint64 vramSize, HSAuint32 count,
                                        HSAint32 xnack_enable) {
    HSAuint64 sysMemSize = GetSysMemSize();
    HSAuint64 size, sizeInPages;
    HSAuint64 vramBufSizeInPages;

    LOG() << "Found System RAM of " << std::dec << (sysMemSize >> 20) << "MB" << std::endl;

    /* use up to one third of total system memory for eviction buffer to test
     * limit max eviction size to 1/2 of vramSize.
     */
    size = MIN(sysMemSize / 3, vramSize / 2);
    size += vramSize;

    /* Check if there is enough system memory to pass test for XNACK off
     * KFD system memory limit is 15/16.
     */
    if (!xnack_enable && size > (sysMemSize - (sysMemSize >> 4)))
        return 0;

    sizeInPages = size >> PAGE_SHIFT;
    vramBufSizeInPages = sizeInPages / (count * N_PROCESSES);

    return vramBufSizeInPages << PAGE_SHIFT;
}

void KFDSVMEvictTest::AllocBuffers(HSAuint32 defaultGPUNode, HSAuint32 count, HSAuint64 vramBufSize,
        std::vector<void *> &pBuffers, HSAuint32 Granularity) {
    HSAuint64   totalMB;

    totalMB = N_PROCESSES * count * (vramBufSize >> 20);
    if (m_IsParent) {
        LOG() << "Testing " << N_PROCESSES << "*" << count << "*" << (vramBufSize>>20) << "(="<< totalMB << ")MB" << std::endl;
    }
    HSAKMT_STATUS ret;
    HSAuint32 retry = 0;

    for (HSAuint32 i = 0; i < count; i++) {
        m_pBuf = mmap(0, vramBufSize, PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
        ASSERT_NE(MAP_FAILED, m_pBuf);

        m_Flags = (HSA_SVM_FLAGS)0;
retry:
        ret = RegisterSVMRange(defaultGPUNode, m_pBuf, vramBufSize, defaultGPUNode, m_Flags);
        if (ret == HSAKMT_STATUS_SUCCESS) {
            pBuffers.push_back(m_pBuf);
            if (Granularity)
                EXPECT_SUCCESS(SVMRangSetGranularity(m_pBuf, vramBufSize, Granularity));
            retry = 0;
        } else {
            if (retry++ > ALLOCATE_RETRY_TIMES) {
                munmap(m_pBuf, vramBufSize);
                break;
            }
            printf("retry %d allocate vram\n", retry);

            /* wait for 1 second to try allocate again */
            sleep(1);
            goto retry;
        }
    }
}

void KFDSVMEvictTest::FreeBuffers(std::vector<void *> &pBuffers, HSAuint64 vramBufSize) {
    for (HSAuint32 i = 0; i < pBuffers.size(); i++) {
        m_pBuf = pBuffers[i];
        if (m_pBuf != NULL)
            munmap(m_pBuf, vramBufSize);
    }
}

void KFDSVMEvictTest::ForkChildProcesses(int nprocesses) {
    int i;

    for (i = 0; i < nprocesses - 1; ++i) {
        pid_t pid = fork();
        ASSERT_GE(pid, 0);

        if (pid == 0) {
            /* Child process */
            /* Cleanup file descriptors copied from parent process
             * then call SetUp->hsaKmtOpenKFD to create new process
             */
            m_psName = "Test process " + std::to_string(i) + " ";
            TearDown();
            SetUp();
            m_ChildPids.clear();
            m_IsParent = false;
            return;
        }

        /* Parent process */
        m_ChildPids.push_back(pid);
    }

    m_psName = "Test process " + std::to_string(i) + " ";
}

void KFDSVMEvictTest::WaitChildProcesses() {
    if (m_IsParent) {
        /* only run by parent process */
        int childStatus;
        int childExitOkNum = 0;
        int size = m_ChildPids.size();

        for (HSAuint32 i = 0; i < size; i++) {
            pid_t pid = m_ChildPids.front();

            waitpid(pid, &childStatus, 0);
            if (WIFEXITED(childStatus) == 1 && WEXITSTATUS(childStatus) == 0)
                childExitOkNum++;

            m_ChildPids.erase(m_ChildPids.begin());
        }

        ASSERT_EQ(childExitOkNum, size);
    }

    /* child process or parent process finished successfullly */
    m_ChildStatus = HSAKMT_STATUS_SUCCESS;
}

/* Evict and restore procedure basic test
 *
 * Use N_PROCESSES processes to allocate vram buf size larger than total vram size
 *
 * ALLOCATE_BUF_SIZE_MB buf allocation size
 *
 * number of buf is equal to (vramSizeMB / (vramBufSizeMB * N_PROCESSES) ) + 8
 * Total vram all processes allocated: 8GB for 4GB Fiji, and 20GB for 16GB Vega10
 *
 * many times of eviction and restore will happen:
 * ttm will evict buffers of another process if not enough free vram
 * process restore will evict buffers of another process
 *
 * Sometimes the allocate may fail (maybe that is normal)
 * ALLOCATE_RETRY_TIMES max retry times to allocate
 *
 * This is basic test, no queue so vram are not used by GPU during test
 *
 * Todo:
 *    - Synchronization between the processes, so they know for sure when
 *        they are done allocating memory
 */
TEST_P(KFDSVMEvictTest, BasicTest) {
    TEST_REQUIRE_ENV_CAPABILITIES(ENVCAPS_64BITLINUX);
    TEST_START(TESTPROFILE_RUNALL);

    if (!SVMAPISupported())
        return;

    HSAint32 xnack_enable = 0;
    EXPECT_SUCCESS(hsaKmtGetXNACKMode(&xnack_enable));
    if (!xnack_enable) {
	    LOG() << std::hex << "Test is skipped with xnack off" << std::endl;
            return;
    }

    HSAuint32 defaultGPUNode = m_NodeInfo.HsaDefaultGPUNode();
    ASSERT_GE(defaultGPUNode, 0) << "failed to get default GPU Node";
    HSAuint64 vramBufSize = ALLOCATE_BUF_SIZE_MB * 1024 * 1024;

    const HsaNodeProperties *pNodeProperties = m_NodeInfo.HsaDefaultGPUNodeProperties();

    if (pNodeProperties->Integrated) {
        LOG() << "Skipping test on APU." << std::endl;
        return;
    }

    HSAuint64 vramSize = GetVramSize(defaultGPUNode);

    if (!vramSize) {
        LOG() << "No VRAM found, skipping the test" << std::endl;
        return;
    } else {
        LOG() << "Found VRAM of " << std::dec << (vramSize >> 20) << "MB" << std::endl;
    }

    HSAuint32 count = GetBufferCounter(vramSize, vramBufSize);
    if (count == 0) {
        LOG() << "Not enough system memory, skipping the test" << std::endl;
        return;
    }

    /* Fork the child processes */
    ForkChildProcesses(N_PROCESSES);

    std::vector<void *> pBuffers;
    AllocBuffers(defaultGPUNode, count, vramBufSize, pBuffers, 0);

    /* wait for other processes to finish allocation, then free buffer */
    sleep(ALLOCATE_RETRY_TIMES);

    LOG() << m_psName << "free buffer" << std::endl;
    FreeBuffers(pBuffers, vramBufSize);

    WaitChildProcesses();

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
TEST_P(KFDSVMEvictTest, QueueTest) {
    TEST_REQUIRE_ENV_CAPABILITIES(ENVCAPS_64BITLINUX);
    TEST_START(TESTPROFILE_RUNALL)

    if (!SVMAPISupported())
        return;

    HSAint32 xnack_enable = 0;
    EXPECT_SUCCESS(hsaKmtGetXNACKMode(&xnack_enable));
    if (!xnack_enable) {
	LOG() << std::hex << "Test is skipped with xnack off" << std::endl;
        return;
    }

    HSAuint32 defaultGPUNode = m_NodeInfo.HsaDefaultGPUNode();
    ASSERT_GE(defaultGPUNode, 0) << "failed to get default GPU Node";
    unsigned int count = MAX_WAVEFRONTS;

    const HsaNodeProperties *pNodeProperties = m_NodeInfo.HsaDefaultGPUNodeProperties();

    /* Skip test for chip it doesn't have CWSR, which the test depends on */
    if (m_FamilyId < FAMILY_VI || isTonga(pNodeProperties) || m_FamilyId >= FAMILY_NV) {
        LOG() << std::hex << "Test is skipped for family ID 0x" << m_FamilyId << std::endl;
        return;
    }

    if (pNodeProperties->Integrated) {
        LOG() << "Skipping test on APU." << std::endl;
        return;
    }

    uint32_t cu_num = pNodeProperties->NumFComputeCores / pNodeProperties->NumSIMDPerCU;
    uint32_t wave_num = MIN(cu_num * 40,
                        (pNodeProperties->NumShaderBanks / pNodeProperties->NumArrays) * 512);
    if (wave_num < count * N_PROCESSES) {
        LOG() << std::hex << "Test is skipped, wave_num " << wave_num << " not enough" << std::endl;
        return;
    }

    HSAuint32 i;
    HSAuint64 vramSize = GetVramSize(defaultGPUNode);

    if (!vramSize) {
        LOG() << "No VRAM found, skipping the test" << std::endl;
        return;
    } else {
        LOG() << "Found VRAM of " << std::dec << (vramSize >> 20) << "MB." << std::endl;
    }

    HSAuint64 vramBufSize = GetBufferSize(vramSize, count, xnack_enable);
    if (vramBufSize == 0) {
        LOG() << "Not enough system memory, skipping the test" << std::endl;
        return;
    }
    /* assert all buffer address can be stored within one page
     * because only one page host memory srcBuf is allocated
     */
    ASSERT_LE(count, PAGE_SIZE/sizeof(unsigned int *));

    /* Fork the child processes */
    ForkChildProcesses(N_PROCESSES);

    HsaMemoryBuffer isaBuffer(PAGE_SIZE, defaultGPUNode, true/*zero*/, false/*local*/, true/*exec*/);
    HsaMemoryBuffer addrBuffer(PAGE_SIZE, defaultGPUNode);
    HsaMemoryBuffer resultBuffer(PAGE_SIZE, defaultGPUNode);

    std::vector<void *> pBuffers;
    HSAuint32 granularity = 0;
    /* xnack is on, shadder code will trigger gpu page fault that bring data
     * to vram. use granularity to move all data from system buffer to vram
     * to reduce system ram pressure in order to avoid system ram oom in system
     * that has less system ram.
     */
    if (xnack_enable)
       granularity = 0xff;
    AllocBuffers(defaultGPUNode, count, vramBufSize, pBuffers, granularity);

    unsigned int wavefront_num = pBuffers.size();
    LOG() << m_psName << "wavefront number " << wavefront_num << std::endl;

    void **localBufAddr = addrBuffer.As<void **>();
    unsigned int *result = resultBuffer.As<uint32_t *>();

    for (i = 0; i < wavefront_num; i++)
        *(localBufAddr + i) = pBuffers[i];

    for (i = 0; i < wavefront_num; i++)
        *(result + i) = vramBufSize;

    ASSERT_SUCCESS(m_pAsm->RunAssembleBuf(ReadMemoryIsa, isaBuffer.As<char*>()));

    PM4Queue pm4Queue;
    ASSERT_SUCCESS(pm4Queue.Create(defaultGPUNode));

    Dispatch dispatch0(isaBuffer);
    dispatch0.SetArgs(localBufAddr, result);
    dispatch0.SetDim(wavefront_num, 1, 1);
    /* submit the packet and start shader */
    dispatch0.Submit(pm4Queue);

    /* doing evict/restore queue test for 5 seconds while queue is running */
    sleep(5);

    /* LOG() << m_psName << "notify shader to quit" << std::endl; */
    /* fill address buffer so shader quits */
    addrBuffer.Fill(0x5678);

    /* wait for shader to finish or timeout if shade has vm page fault */
    dispatch0.SyncWithStatus(g_TestTimeOut * 5);

    ASSERT_SUCCESS(pm4Queue.Destroy());
    /* LOG() << m_psName << "free buffer" << std::endl; */
    /* cleanup */
    FreeBuffers(pBuffers, vramBufSize);

    /* check if all wavefronts finish successfully */
    for (i = 0; i < wavefront_num; i++)
        ASSERT_EQ(0x5678, *(result + i));

    WaitChildProcesses();

    TEST_END
}

INSTANTIATE_TEST_CASE_P(, KFDSVMEvictTest,::testing::Values(0, 1));
