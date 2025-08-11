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

#include "KFDMemoryTest.hpp"
#include <sys/prctl.h>
#include <sys/ptrace.h>
#include <errno.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/types.h>
#include <signal.h>
#include <numa.h>
#include <vector>
#include "Dispatch.hpp"
#include "PM4Queue.hpp"
#include "PM4Packet.hpp"
#include "SDMAQueue.hpp"
#include "SDMAPacket.hpp"
#include "hsakmt/linux/kfd_ioctl.h"

/* Captures user specified time (seconds) to sleep */
extern unsigned int g_SleepTime;

static pthread_mutex_t ptrace_mtx;

void KFDMemoryTest::SetUp() {
    ROUTINE_START

    KFDBaseComponentTest::SetUp();

    ROUTINE_END
}

void KFDMemoryTest::TearDown() {
    ROUTINE_START

    KFDBaseComponentTest::TearDown();

    ROUTINE_END
}

#include <sys/mman.h>
#define GB(x) ((x) << 30)

/*
 * Try to map as much as possible system memory to gpu
 * to see if KFD supports 1TB memory correctly or not.
 * After this test case, we can observe if there are any side effects.
 * NOTICE: There are memory usage limit checks in hsa/kfd according to the total
 * physical system memory.
 */
static void MMapLarge(KFDTEST_PARAMETERS* pTestParamters) {

    int gpuNode = pTestParamters->gpuNode;
    KFDMemoryTest* pKFDMemoryTest = (KFDMemoryTest*)pTestParamters->pTestObject;

    if (!hsakmt_is_dgpu()) {
        LOG() << "Skipping test: Test not supported on APU." << std::endl;
        return;
    }

	const HSAuint64 nObjects = 1<<14;
    HSAuint64 *AlternateVAGPU = new HSAuint64[nObjects];
    ASSERT_NE_GPU((HSAuint64)AlternateVAGPU, 0, gpuNode);
    HsaMemMapFlags mapFlags = {0};
    HSAuint64 s;
    char *addr;
    HSAuint64 flags = MAP_ANONYMOUS | MAP_PRIVATE;

    /* Test up to 1TB memory*/
    s = GB(1024ULL) / nObjects;
    addr = reinterpret_cast<char*>(mmap(0, s, PROT_READ | PROT_WRITE, flags, -1, 0));
    ASSERT_NE_GPU(addr, MAP_FAILED, gpuNode);
    memset(addr, 0, s);

    int i = 0;
    /* Allocate 1024GB, aka 1TB*/
    for (; i < nObjects; i++) {

        /* Code snippet to allow CRIU checkpointing */
        if (i == (1 << 6)) {
            if (g_SleepTime > 0) {
                LOG() << "Pause for: " << g_SleepTime << " seconds" <<  std::endl;
                sleep(g_SleepTime);
            }
        }

        if (hsaKmtRegisterMemory(addr + i, s - i))
            break;
        if (hsaKmtMapMemoryToGPUNodes(addr + i, s - i,
                    &AlternateVAGPU[i], mapFlags, 1, reinterpret_cast<HSAuint32 *>(&gpuNode))) {
            hsaKmtDeregisterMemory(addr + i);
            break;
        }
    }

    LOG() << "Successfully registered and mapped " << (i * s >> 30)
            << "GB system memory to gpu" << std::endl;

    RECORD(i * s >> 30) << "Mmap-SysMem-Size";

    while (i--) {
        EXPECT_SUCCESS_GPU(hsaKmtUnmapMemoryToGPU(reinterpret_cast<void*>(AlternateVAGPU[i])), gpuNode);
        EXPECT_SUCCESS_GPU(hsaKmtDeregisterMemory(reinterpret_cast<void*>(AlternateVAGPU[i])), gpuNode);
    }

    munmap(addr, s);
    delete []AlternateVAGPU;

}

TEST_F(KFDMemoryTest, MMapLarge) {
    TEST_REQUIRE_ENV_CAPABILITIES(ENVCAPS_64BITLINUX);
    TEST_START(TESTPROFILE_RUNALL)

    ASSERT_SUCCESS(KFDTest_Launch(MMapLarge));

    TEST_END
}

/* Keep memory mapped to default node
 * Keep mapping/unmapping memory to/from non-default node
 * A shader running on default node consistantly accesses
 * memory - make sure memory is always accessible by default,
 * i.e. there is no gpu vm fault.
 * Synchronization b/t host program and shader:
 * 1. Host initializes src and dst buffer to 0
 * 2. Shader keeps reading src buffer and check value
 * 3. Host writes src buffer to 0x5678 to indicate quit, polling dst until it becomes 0x5678
 * 4. Shader write dst buffer to 0x5678 after src changes to 0x5678, then quits
 * 5. Host program quits after dst becomes 0x5678
 * Need at least two gpu nodes to run the test. The default node has to be a gfx9 node,
 * otherwise, test is skipped. Use kfdtest --node=$$ to specify the default node
 * This test case is introduced as a side-result of investigation of SWDEV-134798, which
 * is a gpu vm fault while running rocr conformance test. Here we try to simulate the
 * same test behaviour.
 */
static void MapUnmapToNodes(KFDTEST_PARAMETERS* pTestParamters) {

    int gpuNode = pTestParamters->gpuNode;
    KFDMemoryTest* pKFDMemoryTest = (KFDMemoryTest*)pTestParamters->pTestObject;
	HSAuint32 m_FamilyId = pKFDMemoryTest->GetFamilyIdFromNodeId(gpuNode);

    if (m_FamilyId < FAMILY_AI) {
        LOG() << "Skipping test: Test requires gfx9 and later asics." << std::endl;
        return;
    }

    Assembler* m_pAsm;
    m_pAsm = pKFDMemoryTest->GetAssemblerFromNodeId(gpuNode);
    ASSERT_NOTNULL_GPU(m_pAsm, gpuNode);

    const std::vector<int> gpuNodes = pKFDMemoryTest->Get_NodeInfo()->GetNodesWithGPU();
    if (gpuNodes.size() < 2) {
        LOG() << "Skipping test: At least two GPUs are required." << std::endl;
        return;
    }

    HSAuint32 nondefaultNode;
    for (unsigned i = 0; i < gpuNodes.size(); i++) {
        if (gpuNodes.at(i) != gpuNode) {
            nondefaultNode = gpuNodes.at(i);
            break;
        }
    }
    HSAuint32 mapNodes[2] = {HSAuint32(gpuNode), nondefaultNode};

    HsaMemoryBuffer isaBuffer(PAGE_SIZE, gpuNode, true/*zero*/, false/*local*/, true/*exec*/);
    HsaMemoryBuffer srcBuffer(PAGE_SIZE, gpuNode);
    HsaMemoryBuffer dstBuffer(PAGE_SIZE, gpuNode);

    ASSERT_SUCCESS_GPU(m_pAsm->RunAssembleBuf(PollMemoryIsa, isaBuffer.As<char*>()), gpuNode);

    PM4Queue pm4Queue;
    ASSERT_SUCCESS_GPU(pm4Queue.Create(gpuNode), gpuNode);

    Dispatch dispatch0(isaBuffer);
    dispatch0.SetArgs(srcBuffer.As<void*>(), dstBuffer.As<void*>());
    dispatch0.Submit(pm4Queue);

    HsaMemMapFlags memFlags = {0};
    memFlags.ui32.PageSize = HSA_PAGE_SIZE_4KB;
    memFlags.ui32.HostAccess = 1;

    for (unsigned i = 0; i < 1<<14; i ++) {
        hsaKmtMapMemoryToGPUNodes(srcBuffer.As<void*>(), PAGE_SIZE, NULL, memFlags, (i>>5)&1+1, mapNodes);
    }

    /* Fill src buffer so shader quits */
    srcBuffer.Fill(0x5678);
    WaitOnValue(dstBuffer.As<uint32_t *>(), 0x5678);
    EXPECT_EQ_GPU(*dstBuffer.As<uint32_t *>(), 0x5678, gpuNode);
    EXPECT_SUCCESS_GPU(pm4Queue.Destroy(), gpuNode);
}

TEST_F(KFDMemoryTest, MapUnmapToNodes) {
    TEST_START(TESTPROFILE_RUNALL)

    ASSERT_SUCCESS(KFDTest_Launch(MapUnmapToNodes));

    TEST_END
}

// Basic test of hsaKmtMapMemoryToGPU and hsaKmtUnmapMemoryToGPU
static void MapMemoryToGPU(KFDTEST_PARAMETERS* pTestParamters) {

    int gpuNode = pTestParamters->gpuNode;
    KFDMemoryTest* pKFDMemoryTest = (KFDMemoryTest*)pTestParamters->pTestObject;

    unsigned int *nullPtr = NULL;
    unsigned int* pDb = NULL;

    ASSERT_SUCCESS_GPU(hsaKmtAllocMemory(gpuNode /* system */, PAGE_SIZE, pKFDMemoryTest->GetHsaMemFlags(),
                   reinterpret_cast<void**>(&pDb)), gpuNode);
    // verify that pDb is not null before it's being used
    ASSERT_NE_GPU(nullPtr, pDb, gpuNode) << "hsaKmtAllocMemory returned a null pointer";
    ASSERT_SUCCESS_GPU(hsaKmtMapMemoryToGPU(pDb, PAGE_SIZE, NULL), gpuNode);
    EXPECT_SUCCESS_GPU(hsaKmtUnmapMemoryToGPU(pDb), gpuNode);
    // Release the buffers
    EXPECT_SUCCESS_GPU(hsaKmtFreeMemory(pDb, PAGE_SIZE), gpuNode);
}

TEST_F(KFDMemoryTest, MapMemoryToGPU) {
    TEST_START(TESTPROFILE_RUNALL)

    ASSERT_SUCCESS(KFDTest_Launch(MapMemoryToGPU));

    TEST_END
}


// Following tests are for hsaKmtAllocMemory with invalid params
TEST_F(KFDMemoryTest, InvalidMemoryPointerAlloc) {
    TEST_START(TESTPROFILE_RUNALL)

    m_MemoryFlags.ui32.NoNUMABind = 1;
    EXPECT_EQ(HSAKMT_STATUS_INVALID_PARAMETER, hsaKmtAllocMemory(0 /* system */, PAGE_SIZE, m_MemoryFlags, NULL));

    TEST_END
}

TEST_F(KFDMemoryTest, ZeroMemorySizeAlloc) {
    TEST_START(TESTPROFILE_RUNALL)

    unsigned int* pDb = NULL;
    EXPECT_EQ(HSAKMT_STATUS_INVALID_PARAMETER, hsaKmtAllocMemory(0 /* system */, 0, m_MemoryFlags,
              reinterpret_cast<void**>(&pDb)));

    TEST_END
}

// Basic test for hsaKmtAllocMemory
TEST_F(KFDMemoryTest, MemoryAlloc) {
    TEST_START(TESTPROFILE_RUNALL)

    unsigned int* pDb = NULL;
    m_MemoryFlags.ui32.NoNUMABind = 1;
    EXPECT_SUCCESS(hsaKmtAllocMemory(0 /* system */, PAGE_SIZE, m_MemoryFlags, reinterpret_cast<void**>(&pDb)));

    TEST_END
}

// Basic test for hsaKmtAllocMemory
static void MemoryAllocAll(KFDTEST_PARAMETERS* pTestParamters) {

    int gpuNode = pTestParamters->gpuNode;
    KFDMemoryTest* pKFDMemoryTest = (KFDMemoryTest*)pTestParamters->pTestObject;

    HsaMemFlags memFlags = {0};
    memFlags.ui32.NonPaged = 1; // sys mem vs vram
    HSAuint64 available;

    if (pKFDMemoryTest->Get_Version()->KernelInterfaceMinorVersion < 9) {
        LOG() << "Available memory IOCTL not present in KFD. Exiting." << std::endl;
        return;
    }

    void *object = NULL;
    int shrink = 21, success = HSAKMT_STATUS_NO_MEMORY;
    EXPECT_SUCCESS_GPU(hsaKmtAvailableMemory(gpuNode, &available), gpuNode);
    LOG() << "Available: " << available << " bytes" << std::endl;
    HSAuint64 leeway = (10 << shrink), size = available + leeway;
    for (int i = 0; i < available >> shrink; i++) {
        if (hsaKmtAllocMemory(gpuNode, size, memFlags, &object) == HSAKMT_STATUS_SUCCESS) {
            success = hsaKmtFreeMemory(object, available);
            break;
        }
        size -= (1 << shrink);
    }
    if (success == HSAKMT_STATUS_SUCCESS) {
        LOG() << "Allocated: " << size << " bytes" << std::endl;
        if (size > available + leeway) {
            LOG() << "Under-reported available memory!" << std::endl;
        }
        if (size < available - leeway) {
            LOG() << "Over-reported available memory!" << std::endl;
        }
    }
    EXPECT_SUCCESS_GPU(success, gpuNode);
}

TEST_F(KFDMemoryTest, MemoryAllocAll) {
    TEST_START(TESTPROFILE_RUNALL)

    ASSERT_SUCCESS(KFDTest_Launch(MemoryAllocAll));

    TEST_END
}

static void AccessPPRMem(KFDTEST_PARAMETERS* pTestParamters) {

    int gpuNode = pTestParamters->gpuNode;
    KFDMemoryTest* pKFDMemoryTest = (KFDMemoryTest*)pTestParamters->pTestObject;

    if (hsakmt_is_dgpu()) {
        LOG() << "Skipping test: Test requires APU." << std::endl;
        return;
    }

    unsigned int *destBuf = (unsigned int *)VirtualAllocMemory(NULL, PAGE_SIZE,
                                            MEM_READ | MEM_WRITE);

    PM4Queue queue;

    ASSERT_SUCCESS_GPU(queue.Create(gpuNode), gpuNode);

    HsaEvent *event;
    ASSERT_SUCCESS_GPU(CreateQueueTypeEvent(false, false, gpuNode, &event), gpuNode);

    queue.PlaceAndSubmitPacket(PM4WriteDataPacket(destBuf,
                                0xABCDEF09, 0x12345678));

    queue.Wait4PacketConsumption(event);

    WaitOnValue(destBuf, 0xABCDEF09);
    WaitOnValue(destBuf + 1, 0x12345678);

    hsaKmtDestroyEvent(event);
    EXPECT_SUCCESS_GPU(queue.Destroy(), gpuNode);

    /* This sleep hides the dmesg PPR message storm on Raven, which happens
     * when the CPU buffer is freed before the excessive PPRs are all
     * consumed by IOMMU HW. Because of that, a kernel driver workaround
     * is put in place to address that, so we don't need to wait here.
     */
    // sleep(5);

    VirtualFreeMemory(destBuf, PAGE_SIZE);
}

TEST_F(KFDMemoryTest, AccessPPRMem) {
    TEST_START(TESTPROFILE_RUNALL)

    ASSERT_SUCCESS(KFDTest_Launch(AccessPPRMem));

    TEST_END
}

// Linux OS-specific Test for registering OS allocated memory
static void MemoryRegister(KFDTEST_PARAMETERS* pTestParamters) {

    int gpuNode = pTestParamters->gpuNode;
    KFDMemoryTest* pKFDMemoryTest = (KFDMemoryTest*)pTestParamters->pTestObject;

    Assembler* m_pAsm;
    m_pAsm = pKFDMemoryTest->GetAssemblerFromNodeId(gpuNode);
    ASSERT_NOTNULL_GPU(m_pAsm, gpuNode);

	const HsaNodeProperties *pNodeProperties = pKFDMemoryTest->Get_NodeInfo()->GetNodeProperties(gpuNode);

    /* Different unaligned memory locations to be mapped for GPU
     * access:
     *
     * - initialized data segment (file backed)
     * - stack (anonymous memory)
     *
     * Separate them enough so they are in different cache lines
     * (64-byte = 16-dword).
     */
    static volatile HSAuint32 globalData = 0xdeadbeef;
    volatile HSAuint32 stackData[17] = {0};
    const unsigned dstOffset = 0;
    const unsigned sdmaOffset = 16;

    HsaMemoryBuffer srcBuffer((void *)&globalData, sizeof(HSAuint32));
    HsaMemoryBuffer dstBuffer((void *)&stackData[dstOffset], sizeof(HSAuint32));
    HsaMemoryBuffer sdmaBuffer((void *)&stackData[sdmaOffset], sizeof(HSAuint32));

    /* Create PM4 and SDMA queues before fork+COW to test queue
     * eviction and restore
     */
    PM4Queue pm4Queue;
    SDMAQueue sdmaQueue;
    ASSERT_SUCCESS_GPU(pm4Queue.Create(gpuNode), gpuNode);
    ASSERT_SUCCESS_GPU(sdmaQueue.Create(gpuNode), gpuNode);

    HsaMemoryBuffer isaBuffer(PAGE_SIZE, gpuNode, true/*zero*/, false/*local*/, true/*exec*/);

    ASSERT_SUCCESS_GPU(m_pAsm->RunAssembleBuf(CopyDwordIsa, isaBuffer.As<char*>()), gpuNode);

    /* First submit just so the queues are not empty, and to get the
     * TLB populated (in case we need to flush TLBs somewhere after
     * updating the page tables)
     */
    Dispatch dispatch0(isaBuffer);
    dispatch0.SetArgs(srcBuffer.As<void*>(), dstBuffer.As<void*>());
    dispatch0.Submit(pm4Queue);
    dispatch0.Sync(g_TestTimeOut);

    sdmaQueue.PlaceAndSubmitPacket(SDMAWriteDataPacket(sdmaQueue.GetFamilyId(), sdmaBuffer.As<HSAuint32 *>(), 0x12345678));
    sdmaQueue.Wait4PacketConsumption();
    EXPECT_TRUE_GPU(WaitOnValue(&stackData[sdmaOffset], 0x12345678), gpuNode);

    /* Fork a child process to mark pages as COW */
    pid_t pid = fork();
    ASSERT_GE_GPU(pid, 0, gpuNode);
    if (pid == 0) {
        /* Child process waits for a SIGTERM from the parent. It can't
         * make any write access to the stack because we want the
         * parent to make the first write access and get a new copy. A
         * busy loop is the safest way to do that, since any function
         * call (e.g. sleep) would write to the stack.
         */
        while (1)
        {}
        WARN() << "Shouldn't get here!" << std::endl;
        exit(0);
    }

    /* Parent process writes to COW page(s) and gets a new copy. MMU
     * notifier needs to update the GPU mapping(s) for the test to
     * pass.
     */
    globalData = 0xD00BED00;
    stackData[dstOffset] = 0xdeadbeef;
    stackData[sdmaOffset] = 0xdeadbeef;

    /* Terminate the child process before a possible test failure that
     * would leave it spinning in the background indefinitely.
     */
    int status;
    EXPECT_EQ_GPU(0, kill(pid, SIGTERM), gpuNode);
    EXPECT_EQ_GPU(pid, waitpid(pid, &status, 0), gpuNode);
    EXPECT_NE_GPU(0, WIFSIGNALED(status), gpuNode);
    EXPECT_EQ_GPU(SIGTERM, WTERMSIG(status), gpuNode);

    /* Now check that the GPU is accessing the correct page */
    Dispatch dispatch1(isaBuffer);
    dispatch1.SetArgs(srcBuffer.As<void*>(), dstBuffer.As<void*>());
    dispatch1.Submit(pm4Queue);
    dispatch1.Sync(g_TestTimeOut);

    sdmaQueue.PlaceAndSubmitPacket(SDMAWriteDataPacket(sdmaQueue.GetFamilyId(), sdmaBuffer.As<HSAuint32 *>(), 0xD0BED0BE));
    sdmaQueue.Wait4PacketConsumption();

    EXPECT_SUCCESS_GPU(pm4Queue.Destroy(), gpuNode);
    EXPECT_SUCCESS_GPU(sdmaQueue.Destroy(), gpuNode);

    EXPECT_EQ_GPU(0xD00BED00, globalData, gpuNode);
    EXPECT_EQ_GPU(0xD00BED00, stackData[dstOffset], gpuNode);
    EXPECT_EQ_GPU(0xD0BED0BE, stackData[sdmaOffset], gpuNode);
}

TEST_F(KFDMemoryTest, MemoryRegister) {
    TEST_START(TESTPROFILE_RUNALL)

    ASSERT_SUCCESS(KFDTest_Launch(MemoryRegister));

    TEST_END
}

static void MemoryRegisterSamePtr(KFDTEST_PARAMETERS* pTestParamters) {

    int gpuNode = pTestParamters->gpuNode;
    KFDMemoryTest* pKFDMemoryTest = (KFDMemoryTest*)pTestParamters->pTestObject;

    Assembler* m_pAsm;
    m_pAsm = pKFDMemoryTest->GetAssemblerFromNodeId(gpuNode);
    ASSERT_NOTNULL_GPU(m_pAsm, gpuNode);

	HSAuint32 m_FamilyId = pKFDMemoryTest->GetFamilyIdFromNodeId(gpuNode);

    if (!hsakmt_is_dgpu()) {
        LOG() << "Skipping test: Will run on APU once APU+dGPU supported." << std::endl;
        return;
    }

    const std::vector<int> gpuNodes = pKFDMemoryTest->Get_NodeInfo()->GetNodesWithGPU();
    HSAuint64 nGPU = gpuNodes.size();  // number of gpu nodes
    static volatile HSAuint32 mem[4];
    HSAuint64 gpuva1, gpuva2;

    /* Same address, different size */
    EXPECT_SUCCESS(hsaKmtRegisterMemory((void *)&mem[0], sizeof(HSAuint32)*2));
    EXPECT_SUCCESS(hsaKmtMapMemoryToGPU((void *)&mem[0], sizeof(HSAuint32)*2,
                                        &gpuva1));
    EXPECT_SUCCESS(hsaKmtRegisterMemory((void *)&mem[0], sizeof(HSAuint32)));
    EXPECT_SUCCESS(hsaKmtMapMemoryToGPU((void *)&mem[0], sizeof(HSAuint32),
                                        &gpuva2));
    EXPECT_SUCCESS(hsaKmtUnmapMemoryToGPU(reinterpret_cast<void *>(gpuva1)));
    EXPECT_SUCCESS(hsaKmtDeregisterMemory(reinterpret_cast<void *>(gpuva1)));
    EXPECT_SUCCESS(hsaKmtUnmapMemoryToGPU(reinterpret_cast<void *>(gpuva2)));
    EXPECT_SUCCESS(hsaKmtDeregisterMemory(reinterpret_cast<void *>(gpuva2)));

    /* Same address, same size */
    HsaMemMapFlags memFlags = {0};
    memFlags.ui32.PageSize = HSA_PAGE_SIZE_4KB;
    memFlags.ui32.HostAccess = 1;

    HSAuint32 nodes[nGPU];
    for (unsigned int i = 0; i < nGPU; i++)
        nodes[i] = gpuNodes.at(i);
    EXPECT_SUCCESS(hsaKmtRegisterMemoryToNodes((void *)&mem[2],
                            sizeof(HSAuint32)*2, nGPU, nodes));
    EXPECT_SUCCESS(hsaKmtMapMemoryToGPUNodes((void *)&mem[2],
                                        sizeof(HSAuint32) * 2,
                                        &gpuva1, memFlags, nGPU, nodes));
    EXPECT_SUCCESS(hsaKmtRegisterMemoryToNodes((void *)&mem[2],
                                        sizeof(HSAuint32) * 2, nGPU, nodes));
    EXPECT_SUCCESS(hsaKmtMapMemoryToGPUNodes((void *)&mem[2],
                                        sizeof(HSAuint32) * 2,
                                        &gpuva2, memFlags, nGPU, nodes));
    EXPECT_EQ(gpuva1, gpuva2);
    EXPECT_SUCCESS(hsaKmtUnmapMemoryToGPU(reinterpret_cast<void *>(gpuva1)));
    EXPECT_SUCCESS(hsaKmtDeregisterMemory(reinterpret_cast<void *>(gpuva1)));
    /* Confirm that we still have access to the memory, mem[2] */
    PM4Queue queue;
    ASSERT_SUCCESS(queue.Create(gpuNode));
    mem[2] = 0x0;
    queue.PlaceAndSubmitPacket(PM4WriteDataPacket(reinterpret_cast<unsigned int *>(gpuva2),
                                                  0xdeadbeef));
    queue.PlaceAndSubmitPacket(PM4ReleaseMemoryPacket(m_FamilyId, true, 0, 0));
    queue.Wait4PacketConsumption();
    EXPECT_EQ(true, WaitOnValue((unsigned int *)(&mem[2]), 0xdeadbeef));
    EXPECT_SUCCESS(queue.Destroy());
    EXPECT_SUCCESS(hsaKmtUnmapMemoryToGPU(reinterpret_cast<void *>(gpuva2)));
    EXPECT_SUCCESS(hsaKmtDeregisterMemory(reinterpret_cast<void *>(gpuva2)));
}

TEST_F(KFDMemoryTest, MemoryRegisterSamePtr) {
    TEST_START(TESTPROFILE_RUNALL)

    ASSERT_SUCCESS(KFDTest_Launch(MemoryRegisterSamePtr));

    TEST_END
}

/* FlatScratchAccess
 * Since HsaMemoryBuffer has to be associated with a specific GPU node, this function in the current form
 * will not work for multiple GPU nodes. For now test only one default GPU node.
 * TODO: Generalize it to support multiple nodes
 */

#define SCRATCH_SLICE_SIZE 0x10000
#define SCRATCH_SLICE_NUM 3
#define SCRATCH_SIZE (SCRATCH_SLICE_NUM * SCRATCH_SLICE_SIZE)
#define SCRATCH_SLICE_OFFSET(i) ((i) * SCRATCH_SLICE_SIZE)

static void FlatScratchAccess(KFDTEST_PARAMETERS* pTestParamters) {

    int gpuNode = pTestParamters->gpuNode;
    KFDMemoryTest* pKFDMemoryTest = (KFDMemoryTest*)pTestParamters->pTestObject;

	HSAuint32 m_FamilyId = pKFDMemoryTest->GetFamilyIdFromNodeId(gpuNode);
    if (m_FamilyId == FAMILY_CI || m_FamilyId == FAMILY_KV) {
        LOG() << "Skipping test: VI-based shader not supported on other ASICs." << std::endl;
        return;
    }

    Assembler* m_pAsm;
    m_pAsm = pKFDMemoryTest->GetAssemblerFromNodeId(gpuNode);
    ASSERT_NOTNULL_GPU(m_pAsm, gpuNode);

    HsaMemoryBuffer isaBuffer(PAGE_SIZE, gpuNode, true/*zero*/, false/*local*/, true/*exec*/);
    HsaMemoryBuffer scratchBuffer(SCRATCH_SIZE, gpuNode, false/*zero*/, false/*local*/,
                                  false/*exec*/, true /*scratch*/);

    // Unmap scratch for sub-allocation mapping tests
    ASSERT_SUCCESS_GPU(hsaKmtUnmapMemoryToGPU(scratchBuffer.As<void*>()), gpuNode);

    // Map and unmap a few slices in different order: 2-0-1, 0-2-1
    ASSERT_SUCCESS_GPU(hsaKmtMapMemoryToGPU(scratchBuffer.As<char*>() + SCRATCH_SLICE_OFFSET(2),
                                        SCRATCH_SLICE_SIZE, NULL), gpuNode);
    ASSERT_SUCCESS_GPU(hsaKmtMapMemoryToGPU(scratchBuffer.As<char*>() + SCRATCH_SLICE_OFFSET(0),
                                        SCRATCH_SLICE_SIZE, NULL), gpuNode);
    ASSERT_SUCCESS_GPU(hsaKmtMapMemoryToGPU(scratchBuffer.As<char*>() + SCRATCH_SLICE_OFFSET(1),
                                        SCRATCH_SLICE_SIZE, NULL), gpuNode);

    EXPECT_SUCCESS_GPU(hsaKmtUnmapMemoryToGPU(scratchBuffer.As<char*>() + SCRATCH_SLICE_OFFSET(1)), gpuNode);
    EXPECT_SUCCESS_GPU(hsaKmtUnmapMemoryToGPU(scratchBuffer.As<char*>() + SCRATCH_SLICE_OFFSET(2)), gpuNode);
    EXPECT_SUCCESS_GPU(hsaKmtUnmapMemoryToGPU(scratchBuffer.As<char*>() + SCRATCH_SLICE_OFFSET(0)), gpuNode);

    // Map everything for test below
    ASSERT_SUCCESS_GPU(hsaKmtMapMemoryToGPU(scratchBuffer.As<char*>(), SCRATCH_SIZE, NULL), gpuNode);

    HsaMemoryBuffer srcMemBuffer(PAGE_SIZE, gpuNode);
    HsaMemoryBuffer dstMemBuffer(PAGE_SIZE, gpuNode);

    // Initialize the srcBuffer to some fixed value
    srcMemBuffer.Fill(0x01010101);

    ASSERT_SUCCESS_GPU(m_pAsm->RunAssembleBuf(ScratchCopyDwordIsa, isaBuffer.As<char*>()), gpuNode);

    const HsaNodeProperties *pNodeProperties = pKFDMemoryTest->Get_NodeInfo()->GetNodeProperties(gpuNode);

    /* TODO: Add support to all GPU Nodes.
     * The loop over the system nodes is removed as the test can be executed only on GPU nodes. This
     * also requires changes to be made to all the HsaMemoryBuffer variables defined above, as
     * HsaMemoryBuffer is now associated with a Node.
     */
    if (pNodeProperties != NULL) {
        // Get the aperture of the scratch buffer
        HsaMemoryProperties *memoryProperties = new HsaMemoryProperties[pNodeProperties->NumMemoryBanks];
        EXPECT_SUCCESS_GPU(hsaKmtGetNodeMemoryProperties(gpuNode, pNodeProperties->NumMemoryBanks,
                       memoryProperties), gpuNode);

        for (unsigned int bank = 0; bank < pNodeProperties->NumMemoryBanks; bank++) {
            if (memoryProperties[bank].HeapType == HSA_HEAPTYPE_GPU_SCRATCH) {
                int numWaves = pNodeProperties->NumShaderBanks;  // WAVES must be >= # SE
                int waveSize = 1;  // Amount of space used by each wave in units of 256 dwords

                PM4Queue queue;
                ASSERT_SUCCESS_GPU(queue.Create(gpuNode), gpuNode);

                HSAuint64 scratchApertureAddr = memoryProperties[bank].VirtualBaseAddress;

                // Create a dispatch packet to copy
                Dispatch dispatchSrcToScratch(isaBuffer);

                // Setup the dispatch packet
                // Copying from the source Memory Buffer to the scratch buffer
                dispatchSrcToScratch.SetArgs(srcMemBuffer.As<void*>(), reinterpret_cast<void*>(scratchApertureAddr));
                dispatchSrcToScratch.SetDim(1, 1, 1);
                dispatchSrcToScratch.SetScratch(numWaves, waveSize, scratchBuffer.As<uint64_t>());
                // Submit the packet
                dispatchSrcToScratch.Submit(queue);
                dispatchSrcToScratch.Sync();

                // Create another dispatch packet to copy scratch buffer contents to destination buffer.
                Dispatch dispatchScratchToDst(isaBuffer);

                // Set the arguments to copy from the scratch buffer to the destination buffer
                dispatchScratchToDst.SetArgs(reinterpret_cast<void*>(scratchApertureAddr), dstMemBuffer.As<void*>());
                dispatchScratchToDst.SetDim(1, 1, 1);
                dispatchScratchToDst.SetScratch(numWaves, waveSize, scratchBuffer.As<uint64_t>());

                // Submit the packet
                dispatchScratchToDst.Submit(queue);
                dispatchScratchToDst.Sync();

                // Check that the scratch buffer contents were correctly copied over to the system memory buffer
                EXPECT_EQ_GPU(dstMemBuffer.As<unsigned int*>()[0], 0x01010101, gpuNode);
            }
        }

        delete [] memoryProperties;
    }
}

TEST_F(KFDMemoryTest, FlatScratchAccess) {
    TEST_START(TESTPROFILE_RUNALL)

    ASSERT_SUCCESS(KFDTest_Launch(FlatScratchAccess));

    TEST_END
}

static void GetTileConfigTest(KFDTEST_PARAMETERS* pTestParamters) {

    int gpuNode = pTestParamters->gpuNode;
    KFDMemoryTest* pKFDMemoryTest = (KFDMemoryTest*)pTestParamters->pTestObject;

    HSAuint32 tile_config[32] = {0};
    HSAuint32 macro_tile_config[16] = {0};
    unsigned int i;
    HsaGpuTileConfig config = {0};

    config.TileConfig = tile_config;
    config.MacroTileConfig = macro_tile_config;
    config.NumTileConfigs = 32;
    config.NumMacroTileConfigs = 16;

    ASSERT_SUCCESS(hsaKmtGetTileConfig(gpuNode, &config));

    LOG() << "tile_config:" << std::endl;
    for (i = 0; i < config.NumTileConfigs; i++)
        LOG() << "\t" << std::dec << i << ": 0x" << std::hex
                << tile_config[i] << std::endl;

    LOG() << "macro_tile_config:" << std::endl;
    for (i = 0; i < config.NumMacroTileConfigs; i++)
        LOG() << "\t" << std::dec << i << ": 0x" << std::hex
                << macro_tile_config[i] << std::endl;

    LOG() << "gb_addr_config: 0x" << std::hex << config.GbAddrConfig
            << std::endl;
    LOG() << "num_banks: 0x" << std::hex << config.NumBanks << std::endl;
    LOG() << "num_ranks: 0x" << std::hex << config.NumRanks << std::endl;
}

TEST_F(KFDMemoryTest, GetTileConfigTest) {
    TEST_START(TESTPROFILE_RUNALL)

    ASSERT_SUCCESS(KFDTest_Launch(GetTileConfigTest));

    TEST_END
}

void SearchLargestBuffer(int allocNode, const HsaMemFlags &memFlags,
                                        HSAuint64 highMB, int nodeToMap,
                                        HSAuint64 *lastSizeMB) {
    int ret;

    HsaMemMapFlags mapFlags = {0};
    HSAuint64 granularityMB = 8;

    /* Testing big buffers in VRAM */
    unsigned int * pDb = NULL;

    highMB = (highMB + granularityMB - 1) & ~(granularityMB - 1);

    HSAuint64 sizeMB;
    HSAuint64 size = 0;

    while (highMB > granularityMB) {
        sizeMB = highMB - granularityMB;
        size = sizeMB * 1024 * 1024;
        ret = hsaKmtAllocMemory(allocNode, size, memFlags,
                                reinterpret_cast<void**>(&pDb));
        if (ret) {
            highMB = sizeMB;
            continue;
        }

        /* Code snippet to allow CRIU checkpointing */
        if (g_SleepTime > 0) {
            LOG() << "Pause for: " << g_SleepTime << " seconds" <<  std::endl;
            sleep(g_SleepTime);
        }

        ret = hsaKmtMapMemoryToGPUNodes(pDb, size, NULL,
                        mapFlags, 1, reinterpret_cast<HSAuint32 *>(&nodeToMap));
        if (ret) {
            EXPECT_SUCCESS_GPU(hsaKmtFreeMemory(pDb, size), nodeToMap);
            highMB = sizeMB;
            continue;
        }
        EXPECT_SUCCESS_GPU(hsaKmtUnmapMemoryToGPU(pDb), nodeToMap);
        EXPECT_SUCCESS_GPU(hsaKmtFreeMemory(pDb, size), nodeToMap);

        if (lastSizeMB)
           *lastSizeMB = sizeMB;
        break;
    }
}

/*
 * Largest*BufferTest allocates, maps/unmaps, and frees the largest possible
 * buffers. Its size is found using binary search in the range
 * (0, RAM SIZE) with a granularity of 8M. Also, the similar logic is
 * repeated on local buffers (VRAM).
 * Please note we limit the largest possible system buffer to be smaller than
 * the RAM size. The reason is that the system buffer can make use of virtual
 * memory so that a system buffer could be very large even though the RAM size
 * is small. For example, on a typical Carrizo platform, the largest allocated
 * system buffer could be more than 14G even though it only has 4G memory.
 * In that situation, it will take too much time to finish the test because of
 * the onerous memory swap operation. So we limit the buffer size that way.
 */
static void LargestSysBufferTest(KFDTEST_PARAMETERS* pTestParamters) {

    int gpuNode = pTestParamters->gpuNode;
    KFDMemoryTest* pKFDMemoryTest = (KFDMemoryTest*)pTestParamters->pTestObject;

    if (!hsakmt_is_dgpu()) {
        LOG() << "Skipping test: Running on APU fails and locks the system." << std::endl;
        return;
    }

    int gpuNum = pKFDMemoryTest->Get_NodeInfo()->GetNodesWithGPU().size();

	/* if no gpu node */
	if (gpuNum <= 0)
		return;

    HSAuint64 lastTestedSizeMB = 0;

    HSAuint64 sysMemSizeMB;
    sysMemSizeMB = pKFDMemoryTest->GetSysMemSize() >> 20;

    sysMemSizeMB/=gpuNum;

    LOG() << "Found System Memory of " << std::dec << sysMemSizeMB
                    << "MB. Using 95% of that for the test" << std::endl;

    SearchLargestBuffer(0, pKFDMemoryTest->GetHsaMemFlags(), sysMemSizeMB*0.95, gpuNode,
                    &lastTestedSizeMB);

    LOG() << "The largest allocated system buffer is " << std::dec
            << lastTestedSizeMB << "MB" << std::endl;
}

TEST_F(KFDMemoryTest, LargestSysBufferTest) {
     TEST_REQUIRE_ENV_CAPABILITIES(ENVCAPS_64BITLINUX);
	 TEST_START(TESTPROFILE_RUNALL);

    ASSERT_SUCCESS(KFDTest_Launch(LargestSysBufferTest));

    TEST_END
}

static void LargestVramBufferTest(KFDTEST_PARAMETERS* pTestParamters) {

   int gpuNode = pTestParamters->gpuNode;
   KFDMemoryTest* pKFDMemoryTest = (KFDMemoryTest*)pTestParamters->pTestObject;

    if (!hsakmt_is_dgpu()) {
        LOG() << "Skipping test: Running on APU fails and locks the system." << std::endl;
        return;
    }

    HSAuint64 lastTestedSizeMB = 0;

    HsaMemFlags memFlags = {0};
    memFlags.ui32.HostAccess = 0;
    memFlags.ui32.NonPaged = 1;

    HSAuint64 vramSizeMB;
    vramSizeMB = pKFDMemoryTest->GetVramSize(gpuNode) >> 20;

    LOG() << "Found VRAM of " << std::dec << vramSizeMB << "MB." << std::endl;

    SearchLargestBuffer(gpuNode, memFlags, vramSizeMB, gpuNode,
                    &lastTestedSizeMB);

    LOG() << "The largest allocated VRAM buffer is " << std::dec
            << lastTestedSizeMB << "MB" << std::endl;

    /* Make sure 3/5 vram can be allocated.*/
    if (vramSizeMB <= 512)
        EXPECT_GE_GPU(lastTestedSizeMB * 5, vramSizeMB * 3, gpuNode);
    else
        EXPECT_GE_GPU(lastTestedSizeMB * 4, vramSizeMB * 3, gpuNode);

    if (lastTestedSizeMB * 16 < vramSizeMB * 15)
        WARN() << "The largest allocated VRAM buffer size is smaller than the expected "
            << vramSizeMB * 15 / 16 << "MB" << std::endl;
}

TEST_F(KFDMemoryTest, LargestVramBufferTest) {
     TEST_REQUIRE_ENV_CAPABILITIES(ENVCAPS_64BITLINUX);
	 TEST_START(TESTPROFILE_RUNALL);

    ASSERT_SUCCESS(KFDTest_Launch(LargestVramBufferTest));

    TEST_END
}
/*
 * BigSysBufferStressTest allocates and maps 128M system buffers in a loop until it
 * fails, then unmaps and frees them afterwards. Meanwhile, a queue task is
 * performed on each buffer.
 */
static void BigSysBufferStressTest(KFDTEST_PARAMETERS* pTestParamters) {

   int gpuNode = pTestParamters->gpuNode;
   KFDMemoryTest* pKFDMemoryTest = (KFDMemoryTest*)pTestParamters->pTestObject;

    if (!hsakmt_is_dgpu()) {
        LOG() << "Skipping test: Running on APU fails and locks the system." << std::endl;
        return;
    }

    HSAuint64 AlternateVAGPU;
    HsaMemMapFlags mapFlags = {0};
    int ret;

    /* Repeatedly allocate and map big buffers in system memory until it fails,
     * then unmap and free them.
     */
#define ARRAY_ENTRIES 2048

    int i = 0, allocationCount = 0;
    unsigned int* pDb_array[ARRAY_ENTRIES];
    HSAuint64 block_size_mb = 128;
    HSAuint64 block_size = block_size_mb * 1024 * 1024;

    /* Test 4 times to see if there is any memory leak.*/
    for (int repeat = 1; repeat < 5; repeat++) {

        for (i = 0; i < ARRAY_ENTRIES; i++) {
            ret = hsaKmtAllocMemory(0 /* system */, block_size, pKFDMemoryTest->GetHsaMemFlags(),
                    reinterpret_cast<void**>(&pDb_array[i]));
            if (ret)
                break;

            ret = hsaKmtMapMemoryToGPUNodes(pDb_array[i], block_size,
                    &AlternateVAGPU, mapFlags, 1, reinterpret_cast<HSAuint32 *>(&gpuNode));
            if (ret) {
                EXPECT_SUCCESS_GPU(hsaKmtFreeMemory(pDb_array[i], block_size), gpuNode);
                break;
            }
        }

        LOG() << "Allocated system buffers time " << std::dec << repeat << ": "
            << i << " * " << block_size_mb << "MB" << std::endl;

        if (allocationCount == 0)
            allocationCount = i;
        EXPECT_GE_GPU(i, allocationCount, gpuNode) << "There might be memory leak!" << std::endl;

        for (int j = 0; j < i; j++) {
            EXPECT_SUCCESS_GPU(hsaKmtUnmapMemoryToGPU(pDb_array[j]), gpuNode);
            EXPECT_SUCCESS_GPU(hsaKmtFreeMemory(pDb_array[j], block_size), gpuNode);
        }
    }
}

TEST_F(KFDMemoryTest, BigSysBufferStressTest) {
     TEST_REQUIRE_ENV_CAPABILITIES(ENVCAPS_64BITLINUX);
	 TEST_START(TESTPROFILE_RUNALL);

    ASSERT_SUCCESS(KFDTest_Launch(LargestVramBufferTest));

    TEST_END
}

#define VRAM_ALLOCATION_ALIGN (1 << 21)  //Align VRAM allocations to 2MB
static void MMBench(KFDTEST_PARAMETERS* pTestParamters) {

   int gpuNode = pTestParamters->gpuNode;
   KFDMemoryTest* pKFDMemoryTest = (KFDMemoryTest*)pTestParamters->pTestObject;

    unsigned testIndex, sizeIndex, memType, nMemTypes;
    const char *memTypeStrings[2] = {"SysMem", "VRAM"};
    const struct {
        unsigned size;
        unsigned num;
    } bufParams[] = {
        /* Buffer sizes in x16 increments. Limit memory usage to about
         * 1GB. For small sizes we use 1000 buffers, which means we
         * conveniently measure microseconds and report nanoseconds.
         */
        {PAGE_SIZE      , 1000},  /*  4KB */
        {PAGE_SIZE <<  4, 1000},  /* 64KB */
        {PAGE_SIZE <<  9,  500},  /*  2MB */
        {PAGE_SIZE << 13,   32},  /* 32MB */
        {PAGE_SIZE << 18,    1},  /*  1GB */
    };
    const unsigned nSizes = sizeof(bufParams) / sizeof(bufParams[0]);
    const unsigned nTests = nSizes << 2;
#define TEST_BUFSIZE(index) (bufParams[(index) % nSizes].size)
#define TEST_NBUFS(index)  (bufParams[(index) % nSizes].num)
#define TEST_MEMTYPE(index) ((index / nSizes) & 0x1)
#define TEST_SDMA(index)    (((index / nSizes) >> 1) & 0x1)

    void *bufs[1000];
    HSAuint64 start, end;
    unsigned i;
    HSAKMT_STATUS ret;
    HsaMemFlags memFlags = {0};
    HsaMemMapFlags mapFlags = {0};
    HSAuint64 altVa;

    HSAuint64 vramSizeMB = pKFDMemoryTest->GetVramSize(gpuNode) >> 20;

    const std::vector<int> gpuNodes = pKFDMemoryTest->Get_NodeInfo()->GetNodesWithGPU();
    bool is_all_large_bar = true;

    for (unsigned i = 0; i < gpuNodes.size(); i++) {
        if (!pKFDMemoryTest->Get_NodeInfo()->IsGPUNodeLargeBar(gpuNodes.at(i))) {
                is_all_large_bar = false;
                break;
        }
    }

    LOG() << "Found VRAM of " << std::dec << vramSizeMB << "MB." << std::endl;

    if (vramSizeMB == 0)
        nMemTypes = 1;
    else
        nMemTypes = 2;

    /* Two SDMA queues to interleave user mode SDMA with memory
     * management on either SDMA engine. Make the queues long enough
     * to buffer at least nBufs x WriteData packets (7 dwords per
     * packet).
     */
    SDMAQueue sdmaQueue[2];
    ASSERT_SUCCESS_GPU(sdmaQueue[0].Create(gpuNode, PAGE_SIZE*8), gpuNode);
    ASSERT_SUCCESS_GPU(sdmaQueue[1].Create(gpuNode, PAGE_SIZE*8), gpuNode);
    HsaMemoryBuffer sdmaBuffer(PAGE_SIZE, 0); /* system memory */
#define INTERLEAVE_SDMA() do {                                          \
        if (interleaveSDMA) {                                           \
            sdmaQueue[0].PlaceAndSubmitPacket(                          \
                SDMAWriteDataPacket(sdmaQueue[0].GetFamilyId(), sdmaBuffer.As<HSAuint32 *>(),       \
                                    0x12345678));                       \
            sdmaQueue[1].PlaceAndSubmitPacket(                          \
                SDMAWriteDataPacket(sdmaQueue[1].GetFamilyId(), sdmaBuffer.As<HSAuint32 *>()+16,    \
                                    0x12345678));                       \
        }                                                               \
    } while (0)
#define IDLE_SDMA() do {                                                \
        if (interleaveSDMA) {                                           \
            sdmaQueue[0].Wait4PacketConsumption();                      \
            sdmaQueue[1].Wait4PacketConsumption();                      \
        }                                                               \
    } while (0)

    LOG() << "Test (avg. ns)\t    alloc   mapOne  umapOne   mapAll  umapAll     free" << std::endl;
    for (testIndex = 0; testIndex < nTests; testIndex++) {
        unsigned bufSize = TEST_BUFSIZE(testIndex);
        unsigned nBufs = TEST_NBUFS(testIndex);
        unsigned memType = TEST_MEMTYPE(testIndex);
        bool interleaveSDMA = TEST_SDMA(testIndex);
        unsigned bufLimit;
        HSAuint64 allocTime, map1Time, unmap1Time, mapAllTime, unmapAllTime, freeTime;
        HSAuint32 allocNode;

        /* Code snippet to allow CRIU checkpointing */
        if (testIndex == 3) {
            if (g_SleepTime > 0) {
                LOG() << "Pause for: " << g_SleepTime << " seconds" <<  std::endl;
                sleep(g_SleepTime);
            }
        }

        if ((testIndex % nSizes) == 0)
            LOG() << "--------------------------------------------------------------------------" << std::endl;

        if (memType >= nMemTypes)
            continue;  // skip unsupported mem types

        if (memType == 0) {
            allocNode = 0;
            memFlags.ui32.PageSize = HSA_PAGE_SIZE_4KB;
            memFlags.ui32.HostAccess = 1;
            memFlags.ui32.NonPaged = 0;
            memFlags.ui32.NoNUMABind = 1;
        } else {
            allocNode = gpuNode;
            memFlags.ui32.PageSize = HSA_PAGE_SIZE_4KB;
            memFlags.ui32.HostAccess = 0;
            memFlags.ui32.NonPaged = 1;

            /* Buffer sizes are 2MB aligned to match new allocation policy.
             * Upper limit of buffer number to fit 80% vram size. APUs w/
			 * smaller VRAM needs different criteria.
             */
            if (vramSizeMB <= 512)
                bufLimit = ((vramSizeMB << 20) * 6 / 10) / ALIGN_UP(bufSize, VRAM_ALLOCATION_ALIGN);
            else
                bufLimit = ((vramSizeMB << 20) * 8 / 10) / ALIGN_UP(bufSize, VRAM_ALLOCATION_ALIGN);

            if (bufLimit == 0)
                continue; // skip when bufSize > vram

            /* When vram is too small to fit all the buffers, fill 90% vram size*/
            nBufs = (nBufs < bufLimit) ? nBufs : bufLimit;
        }

        /* Allocation */
        start = GetSystemTickCountInMicroSec();
        for (i = 0; i < nBufs; i++) {
            ASSERT_SUCCESS_GPU(hsaKmtAllocMemory(allocNode, bufSize, memFlags,
                                             &bufs[i]), gpuNode);
            INTERLEAVE_SDMA();
        }
        allocTime = GetSystemTickCountInMicroSec() - start;
        IDLE_SDMA();

        /* Map to one GPU */
        start = GetSystemTickCountInMicroSec();
        for (i = 0; i < nBufs; i++) {
            ASSERT_SUCCESS_GPU(hsaKmtMapMemoryToGPUNodes(bufs[i], bufSize,
                                                     &altVa, mapFlags, 1,
                                                     (HSAuint32*)&gpuNode),  gpuNode);
            INTERLEAVE_SDMA();
        }
        map1Time = GetSystemTickCountInMicroSec() - start;
        IDLE_SDMA();

        /* Unmap from GPU */
        start = GetSystemTickCountInMicroSec();
        for (i = 0; i < nBufs; i++) {
            EXPECT_SUCCESS_GPU(hsaKmtUnmapMemoryToGPU(bufs[i]), gpuNode);
            INTERLEAVE_SDMA();
        }
        unmap1Time = GetSystemTickCountInMicroSec() - start;
        IDLE_SDMA();

        /* Map to all GPUs */
        if (is_all_large_bar) {
            start = GetSystemTickCountInMicroSec();
            for (i = 0; i < nBufs; i++) {
                ASSERT_SUCCESS_GPU(hsaKmtMapMemoryToGPU(bufs[i], bufSize, &altVa), gpuNode);
                INTERLEAVE_SDMA();
            }
            mapAllTime = GetSystemTickCountInMicroSec() - start;
            IDLE_SDMA();

            /* Unmap from all GPUs */
            start = GetSystemTickCountInMicroSec();
            for (i = 0; i < nBufs; i++) {
                EXPECT_SUCCESS(hsaKmtUnmapMemoryToGPU(bufs[i]));
                INTERLEAVE_SDMA();
            }
            unmapAllTime = GetSystemTickCountInMicroSec() - start;
            IDLE_SDMA();
        }

        /* Free */
        start = GetSystemTickCountInMicroSec();
        for (i = 0; i < nBufs; i++) {
            EXPECT_SUCCESS_GPU(hsaKmtFreeMemory(bufs[i], bufSize), gpuNode);
            INTERLEAVE_SDMA();
        }
        freeTime = GetSystemTickCountInMicroSec() - start;
        IDLE_SDMA();

        allocTime = allocTime * 1000 / nBufs;
        map1Time = map1Time * 1000 / nBufs;
        unmap1Time = unmap1Time * 1000 / nBufs;
        mapAllTime = mapAllTime * 1000 / nBufs;
        unmapAllTime = unmapAllTime * 1000 / nBufs;
        freeTime = freeTime * 1000 / nBufs;

        unsigned bufSizeLog;
        char bufSizeUnit;
        if (bufSize < (1 << 20)) {
            bufSizeLog = bufSize >> 10;
            bufSizeUnit = 'K';
        } else if (bufSize < (1 << 30)) {
            bufSizeLog = bufSize >> 20;
            bufSizeUnit = 'M';
        } else {
            bufSizeLog = bufSize >> 30;
            bufSizeUnit = 'G';
        }

        LOG() << std::dec << std::setiosflags(std::ios::right)
              << std::setw(3) << bufSizeLog << bufSizeUnit << "-"
              << memTypeStrings[memType] << "-"
              << (interleaveSDMA ? "SDMA\t" : "noSDMA\t")
              << std::setw(9) << allocTime
              << std::setw(9) << map1Time
              << std::setw(9) << unmap1Time
              << std::setw(9) << mapAllTime
              << std::setw(9) << unmapAllTime
              << std::setw(9) << freeTime << std::endl;

#define MMBENCH_KEY_PREFIX memTypeStrings[memType] << "-" \
                           << (interleaveSDMA ? "SDMA" : "noSDMA") << "-" \
                           << (bufSize >> 10) << "K-"
        RECORD(allocTime) << MMBENCH_KEY_PREFIX << "alloc";
        RECORD(map1Time) << MMBENCH_KEY_PREFIX << "mapOne";
        RECORD(unmap1Time) << MMBENCH_KEY_PREFIX << "unmapOne";
        RECORD(mapAllTime) << MMBENCH_KEY_PREFIX << "mapAll";
        RECORD(unmapAllTime) << MMBENCH_KEY_PREFIX << "unmapAll";
        RECORD(freeTime) << MMBENCH_KEY_PREFIX << "free";
    }
}

TEST_F(KFDMemoryTest, MMBench) {
     TEST_REQUIRE_ENV_CAPABILITIES(ENVCAPS_64BITLINUX);
	 TEST_START(TESTPROFILE_RUNALL);

    ASSERT_SUCCESS(KFDTest_Launch(MMBench));

    TEST_END
}

static void QueryPointerInfo(KFDTEST_PARAMETERS* pTestParamters) {

    int gpuNode = pTestParamters->gpuNode;
    KFDMemoryTest* pKFDMemoryTest = (KFDMemoryTest*)pTestParamters->pTestObject;

    unsigned int bufSize = PAGE_SIZE * 8;  // CZ and Tonga need 8 pages
    HsaPointerInfo ptrInfo;
    const std::vector<int> gpuNodes = pKFDMemoryTest->Get_NodeInfo()->GetNodesWithGPU();
    HSAuint64 nGPU = gpuNodes.size();  // number of gpu nodes

    /* GraphicHandle is tested at KFDGraphicsInterop.RegisterGraphicsHandle */

    /*** Memory allocated on CPU node ***/
    HsaMemoryBuffer hostBuffer(bufSize, 0/*node*/, false, false/*local*/);
    EXPECT_SUCCESS_GPU(hsaKmtQueryPointerInfo(hostBuffer.As<void*>(), &ptrInfo), gpuNode);
    EXPECT_EQ_GPU(ptrInfo.Type, HSA_POINTER_ALLOCATED, gpuNode);
    EXPECT_EQ_GPU(ptrInfo.Node, 0, gpuNode);
    EXPECT_EQ_GPU(ptrInfo.MemFlags.Value, hostBuffer.Flags().Value, gpuNode);
    EXPECT_EQ_GPU(ptrInfo.CPUAddress, hostBuffer.As<void*>(), gpuNode);
    EXPECT_EQ_GPU(ptrInfo.GPUAddress, (HSAuint64)hostBuffer.As<void*>(), gpuNode);
    EXPECT_EQ_GPU(ptrInfo.SizeInBytes, (HSAuint64)hostBuffer.Size(), gpuNode);
    EXPECT_EQ_GPU(ptrInfo.MemFlags.ui32.CoarseGrain, 0, gpuNode);
    if (hsakmt_is_dgpu()) {
        EXPECT_EQ_GPU((HSAuint64)ptrInfo.NMappedNodes, nGPU, gpuNode);
        // Check NMappedNodes again after unmapping the memory
        hsaKmtUnmapMemoryToGPU(hostBuffer.As<void*>());
        hsaKmtQueryPointerInfo(hostBuffer.As<void*>(), &ptrInfo);
    }
    EXPECT_EQ_GPU((HSAuint64)ptrInfo.NMappedNodes, 0, gpuNode);

    /* Skip testing local memory if the platform does not have it */
    if (pKFDMemoryTest->GetVramSize(gpuNode)) {
        HsaMemoryBuffer localBuffer(bufSize, gpuNode, false, true);
        EXPECT_SUCCESS_GPU(hsaKmtQueryPointerInfo(localBuffer.As<void*>(), &ptrInfo), gpuNode);
        EXPECT_EQ_GPU(ptrInfo.Type, HSA_POINTER_ALLOCATED, gpuNode);
        EXPECT_EQ_GPU(ptrInfo.Node, gpuNode, gpuNode);
        EXPECT_EQ_GPU(ptrInfo.MemFlags.Value, localBuffer.Flags().Value, gpuNode);
        EXPECT_EQ_GPU(ptrInfo.CPUAddress, localBuffer.As<void*>(), gpuNode);
        EXPECT_EQ_GPU(ptrInfo.GPUAddress, (HSAuint64)localBuffer.As<void*>(), gpuNode);
        EXPECT_EQ_GPU(ptrInfo.SizeInBytes, (HSAuint64)localBuffer.Size(), gpuNode);
        EXPECT_EQ_GPU(ptrInfo.MemFlags.ui32.CoarseGrain, 1, gpuNode);

        HSAuint32 *addr = localBuffer.As<HSAuint32 *>() + 4;
        EXPECT_SUCCESS_GPU(hsaKmtQueryPointerInfo(reinterpret_cast<void *>(addr), &ptrInfo), gpuNode);
        EXPECT_EQ_GPU(ptrInfo.GPUAddress, (HSAuint64)localBuffer.As<void*>(), gpuNode);
    }

    /** Registered memory: user pointer */
    static volatile HSAuint32 mem[4];  // 8 bytes for register only and
                                       // 8 bytes for register to nodes
    HsaMemoryBuffer hsaBuffer((void *)(&mem[0]), sizeof(HSAuint32)*2);
    /*
     * APU doesn't use userptr.
     * User pointers registered with SVM API, does not create vm_object_t.
     * Therefore, pointer info can not be queried.
     */
    if (hsakmt_is_dgpu() && mem != hsaBuffer.As<void*>()) {
        EXPECT_SUCCESS_GPU(hsaKmtQueryPointerInfo((void *)(&mem[0]), &ptrInfo), gpuNode);
        EXPECT_EQ_GPU(ptrInfo.Type, HSA_POINTER_REGISTERED_USER, gpuNode);
        EXPECT_EQ_GPU(ptrInfo.CPUAddress, &mem[0], gpuNode);
        EXPECT_EQ_GPU(ptrInfo.GPUAddress, (HSAuint64)hsaBuffer.As<void*>(), gpuNode);
        EXPECT_EQ_GPU(ptrInfo.SizeInBytes, sizeof(HSAuint32)*2, gpuNode);
        EXPECT_EQ_GPU(ptrInfo.NRegisteredNodes, 0, gpuNode);
        EXPECT_EQ_GPU(ptrInfo.NMappedNodes, nGPU, gpuNode);
        EXPECT_EQ_GPU(ptrInfo.MemFlags.ui32.CoarseGrain, 1, gpuNode);
        // Register to nodes
        HSAuint32 nodes[nGPU];
        for (unsigned int i = 0; i < nGPU; i++)
            nodes[i] = gpuNodes.at(i);
        EXPECT_SUCCESS_GPU(hsaKmtRegisterMemoryToNodes((void *)(&mem[2]),
                                sizeof(HSAuint32)*2, nGPU, nodes), gpuNode);
        EXPECT_SUCCESS_GPU(hsaKmtQueryPointerInfo((void *)(&mem[2]), &ptrInfo), gpuNode);
        EXPECT_EQ_GPU(ptrInfo.NRegisteredNodes, nGPU, gpuNode);
        EXPECT_SUCCESS_GPU(hsaKmtDeregisterMemory((void *)(&mem[2])), gpuNode);
    }

    /* Not a starting address, but an address inside the memory range
     * should also get the memory information
     */
    HSAuint32 *address = hostBuffer.As<HSAuint32 *>() + 1;
    EXPECT_SUCCESS_GPU(hsaKmtQueryPointerInfo(reinterpret_cast<void *>(address), &ptrInfo), gpuNode);
    EXPECT_EQ_GPU(ptrInfo.Type, HSA_POINTER_ALLOCATED, gpuNode);
    EXPECT_EQ_GPU(ptrInfo.CPUAddress, hostBuffer.As<void*>(), gpuNode);
    if (hsakmt_is_dgpu() && &mem[1] != hsaBuffer.As<HSAuint32 *>() + 1) {
        EXPECT_SUCCESS_GPU(hsaKmtQueryPointerInfo((void *)(&mem[1]), &ptrInfo), gpuNode);
        EXPECT_EQ_GPU(ptrInfo.Type, HSA_POINTER_REGISTERED_USER, gpuNode);
        EXPECT_EQ_GPU(ptrInfo.CPUAddress, &mem[0], gpuNode);
    }

    /*** Set user data ***/
    char userData[16] = "This is a test.";
    EXPECT_SUCCESS_GPU(hsaKmtSetMemoryUserData(hostBuffer.As<HSAuint32 *>(), reinterpret_cast<void *>(userData)), gpuNode);
    EXPECT_SUCCESS_GPU(hsaKmtQueryPointerInfo(hostBuffer.As<void*>(), &ptrInfo), gpuNode);
    EXPECT_EQ_GPU(ptrInfo.UserData, (void *)userData, gpuNode);
}

TEST_F(KFDMemoryTest, QueryPointerInfo) {

	 TEST_START(TESTPROFILE_RUNALL);

    ASSERT_SUCCESS(KFDTest_Launch(QueryPointerInfo));

    TEST_END
}

/* Linux OS-specific test for a debugger accessing HSA memory in a
 * debugged process.
 *
 * Allocates a system memory and a visible local memory buffer (if
 * possible). Forks a child process that PTRACE_ATTACHes to the parent
 * to access its memory like a debugger would. Child copies data in
 * the parent process using PTRACE_PEEKDATA and PTRACE_POKEDATA. After
 * the child terminates, the parent checks that the copy was
 * successful.
 */
static void PtraceAccess(KFDTEST_PARAMETERS* pTestParamters) {

    int gpuNode = pTestParamters->gpuNode;
    KFDMemoryTest* pKFDMemoryTest = (KFDMemoryTest*)pTestParamters->pTestObject;

    HsaMemFlags memFlags = {0};
    memFlags.ui32.PageSize = HSA_PAGE_SIZE_4KB;
    memFlags.ui32.HostAccess = 1;

    void *mem[2];
    unsigned i;

    /* Offset in the VRAM buffer to test crossing non-contiguous
     * buffer boundaries. The second access starting from offset
     * sizeof(HSAint64)+1 will cross a node boundary in a single access,
     * for node sizes of 4MB or smaller.
     */
    const HSAuint64 VRAM_OFFSET = (4 << 20) - 2 * sizeof(HSAint64);

    // Alloc system memory from node 0 and initialize it
    memFlags.ui32.NonPaged = 0;
    memFlags.ui32.NoNUMABind = 1;
    ASSERT_SUCCESS_GPU(hsaKmtAllocMemory(0, PAGE_SIZE*2, memFlags, &mem[0]), gpuNode);
    for (i = 0; i < 4*sizeof(HSAint64) + 4; i++) {
        (reinterpret_cast<HSAuint8 *>(mem[0]))[i] = i;            // source
        (reinterpret_cast<HSAuint8 *>(mem[0]))[PAGE_SIZE+i] = 0;  // destination
    }

    // Try to alloc local memory from GPU node
    memFlags.ui32.NonPaged = 1;
    if (pKFDMemoryTest->Get_NodeInfo()->IsGPUNodeLargeBar(gpuNode)) {
        EXPECT_SUCCESS_GPU(hsaKmtAllocMemory(gpuNode, PAGE_SIZE*2 + (4 << 20),
                                            memFlags, &mem[1]), gpuNode);
        mem[1] = reinterpret_cast<void *>(reinterpret_cast<HSAuint8 *>(mem[1]) + VRAM_OFFSET);
        for (i = 0; i < 4*sizeof(HSAint64) + 4; i++) {
            (reinterpret_cast<HSAuint8 *>(mem[1]))[i] = i;
            (reinterpret_cast<HSAuint8 *>(mem[1]))[PAGE_SIZE+i] = 0;
        }
    } else {
        LOG() << "Not testing local memory, it's invisible" << std::endl;
        mem[1] = NULL;
    }

    /* Allow any process to trace this one. If kernel is built without
     * Yama, this is not needed, and this call will fail.
     */
#ifdef PR_SET_PTRACER
    prctl(PR_SET_PTRACER, PR_SET_PTRACER_ANY, 0, 0, 0);
#endif

    pthread_mutex_lock(&ptrace_mtx);

    // Find current pid so the child can trace it
    pid_t tracePid = getpid();

    // Fork the child
    pid_t childPid = fork();
    ASSERT_GE_GPU(childPid, 0, gpuNode);
    if (childPid == 0) {
        int traceStatus;
        int err = 0, r;

        /* Child process: we catch any exceptions to make sure we detach
         * from the traced process, because terminating without detaching
         * leaves the traced process stopped.
         */
        r = ptrace(PTRACE_ATTACH, tracePid, NULL, NULL);
        if (r) {
            WARN() << "PTRACE_ATTACH failed: " << r << std::endl;
            exit(1);
        }
        try {
            do {
                waitpid(tracePid, &traceStatus, 0);
            } while (!WIFSTOPPED(traceStatus));

            for (i = 0; i < 4; i++) {
                // Test 4 different (mis-)alignments, leaving 1-byte gaps between longs
                HSAuint8 *addr = reinterpret_cast<HSAuint8 *>(reinterpret_cast<long *>(mem[0]) + i) + i;
                errno = 0;
                long data = ptrace(PTRACE_PEEKDATA, tracePid, addr, NULL);
                EXPECT_EQ_GPU(0, errno, gpuNode);
                EXPECT_EQ_GPU(0, ptrace(PTRACE_POKEDATA, tracePid, addr + PAGE_SIZE,
                                    reinterpret_cast<void *>(data)), gpuNode);

                if (mem[1] == NULL)
                    continue;

                addr = reinterpret_cast<HSAuint8 *>(reinterpret_cast<long *>(mem[1]) + i) + i;
                errno = 0;
                data = ptrace(PTRACE_PEEKDATA, tracePid, addr, NULL);
                EXPECT_EQ_GPU(0, errno, gpuNode);
                EXPECT_EQ_GPU(0, ptrace(PTRACE_POKEDATA, tracePid, addr + PAGE_SIZE,
                                reinterpret_cast<void *>(data)), gpuNode);
            }
        } catch (...) {
            err = 1;
        }
        r = ptrace(PTRACE_DETACH, tracePid, NULL, NULL);
        if (r) {
            WARN() << "PTRACE_DETACH failed: " << r << std::endl;
            exit(1);
        }
        exit(err);
    } else {
        int childStatus;

        // Parent process, just wait for the child to finish
        EXPECT_EQ_GPU(childPid, waitpid(childPid, &childStatus, 0), gpuNode);
        EXPECT_NE_GPU(0, WIFEXITED(childStatus), gpuNode);
        EXPECT_EQ_GPU(0, WEXITSTATUS(childStatus), gpuNode);
    }

    pthread_mutex_unlock(&ptrace_mtx);

    // Clear gaps in the source that should not have been copied
    (reinterpret_cast<uint8_t*>(mem[0]))[  sizeof(long)    ] = 0;
    (reinterpret_cast<uint8_t*>(mem[0]))[2*sizeof(long) + 1] = 0;
    (reinterpret_cast<uint8_t*>(mem[0]))[3*sizeof(long) + 2] = 0;
    (reinterpret_cast<uint8_t*>(mem[0]))[4*sizeof(long) + 3] = 0;
    // Check results
    EXPECT_EQ_GPU(0, memcmp(mem[0], reinterpret_cast<HSAuint8 *>(mem[0]) + PAGE_SIZE,
                        sizeof(long)*4 + 4), gpuNode);
    // Free memory
    EXPECT_SUCCESS_GPU(hsaKmtFreeMemory(mem[0], PAGE_SIZE*2), gpuNode);

    if (mem[1]) {
        (reinterpret_cast<uint8_t*>(mem[1]))[  sizeof(HSAint64)    ] = 0;
        (reinterpret_cast<uint8_t*>(mem[1]))[2*sizeof(HSAint64) + 1] = 0;
        (reinterpret_cast<uint8_t*>(mem[1]))[3*sizeof(HSAint64) + 2] = 0;
        (reinterpret_cast<uint8_t*>(mem[1]))[4*sizeof(HSAint64) + 3] = 0;
        EXPECT_EQ_GPU(0, memcmp(mem[1], reinterpret_cast<HSAuint8 *>(mem[1]) + PAGE_SIZE,
                            sizeof(HSAint64)*4 + 4), gpuNode);
        mem[1] = reinterpret_cast<void *>(reinterpret_cast<HSAuint8 *>(mem[1]) - VRAM_OFFSET);
        EXPECT_SUCCESS_GPU(hsaKmtFreeMemory(mem[1], PAGE_SIZE*2), gpuNode);
    }
}

TEST_F(KFDMemoryTest, PtraceAccess) {
	TEST_START(TESTPROFILE_RUNALL);

    ASSERT_SUCCESS(KFDTest_Launch(PtraceAccess));

    TEST_END
}

static void PtraceAccessInvisibleVram(KFDTEST_PARAMETERS* pTestParamters) {

    int gpuNode = pTestParamters->gpuNode;
    KFDMemoryTest* pKFDMemoryTest = (KFDMemoryTest*)pTestParamters->pTestObject;

	Assembler* m_pAsm;
	m_pAsm = pKFDMemoryTest->GetAssemblerFromNodeId(gpuNode);
	ASSERT_NOTNULL_GPU(m_pAsm, gpuNode);

    HSAuint32 m_FamilyId = pKFDMemoryTest->GetFamilyIdFromNodeId(gpuNode);

    char *hsaDebug = getenv("HSA_DEBUG");

    if (!hsakmt_is_dgpu()) {
        LOG() << "Skipping test: There is no VRAM on APU." << std::endl;
        return;
    }

    if (!hsaDebug || !strcmp(hsaDebug, "0")) {
        LOG() << "Skipping test: HSA_DEBUG environment variable not set." << std::endl;
        return;
    }

    HsaMemMapFlags mapFlags = {0};
    HsaMemFlags memFlags = {0};
    memFlags.ui32.PageSize = HSA_PAGE_SIZE_4KB;
    /* Allocate host not accessible vram */
    memFlags.ui32.HostAccess = 0;
    memFlags.ui32.NonPaged = 1;

    void *mem, *mem0, *mem1;
    unsigned size = PAGE_SIZE*2 + (4 << 20);
    HSAuint64 data[2] = {0xdeadbeefdeadbeef, 0xcafebabecafebabe};
    unsigned int data0[2] = {0xdeadbeef, 0xdeadbeef};
    unsigned int data1[2] = {0xcafebabe, 0xcafebabe};

    const HSAuint64 VRAM_OFFSET = (4 << 20) - sizeof(HSAuint64);

    ASSERT_SUCCESS_GPU(hsaKmtAllocMemory(gpuNode, size, memFlags, &mem), gpuNode);
    ASSERT_SUCCESS_GPU(hsaKmtMapMemoryToGPUNodes(mem, size, NULL,
                                mapFlags, 1, reinterpret_cast<HSAuint32 *>(&gpuNode)), gpuNode);
    /* Set the word before 4M boundary to 0xdeadbeefdeadbeef
     * and the word after 4M boundary to 0xcafebabecafebabe
     */
    mem0 = reinterpret_cast<void *>(reinterpret_cast<HSAuint8 *>(mem) + VRAM_OFFSET);
    mem1 = reinterpret_cast<void *>(reinterpret_cast<HSAuint8 *>(mem) + VRAM_OFFSET + sizeof(HSAuint64));
    PM4Queue queue;
    ASSERT_SUCCESS_GPU(queue.Create(gpuNode), gpuNode);

    queue.PlaceAndSubmitPacket(PM4WriteDataPacket((unsigned int *)mem0,
                                                  data0[0], data0[1]));
    queue.PlaceAndSubmitPacket(PM4WriteDataPacket((unsigned int *)mem1,
                                                  data1[0], data1[1]));
    queue.PlaceAndSubmitPacket(PM4ReleaseMemoryPacket(m_FamilyId, true, 0, 0));
    queue.Wait4PacketConsumption();

    /* Allow any process to trace this one. If kernel is built without
     * Yama, this is not needed, and this call will fail.
     */
#ifdef PR_SET_PTRACER
    prctl(PR_SET_PTRACER, PR_SET_PTRACER_ANY, 0, 0, 0);
#endif

    pthread_mutex_lock(&ptrace_mtx);

    // Find out my pid so the child can trace it
    pid_t tracePid = getpid();

    // Fork the child
    pid_t childPid = fork();
    ASSERT_GE_GPU(childPid, 0, gpuNode);
    if (childPid == 0) {
        int traceStatus;
        int err = 0, r;

        /* Child process: we catch any exceptions to make sure we detach
         * from the traced process, because terminating without detaching
         * leaves the traced process stopped.
         */
        r = ptrace(PTRACE_ATTACH, tracePid, NULL, NULL);
        if (r) {
            WARN() << "PTRACE_ATTACH failed: " << r << std::endl;
            exit(1);
        }
        try {
            do {
                waitpid(tracePid, &traceStatus, 0);
            } while (!WIFSTOPPED(traceStatus));

            /* Peek the memory */
            errno = 0;
            HSAint64 data0 = ptrace(PTRACE_PEEKDATA, tracePid, mem0, NULL);
            EXPECT_EQ_GPU(0, errno, gpuNode);
            EXPECT_EQ_GPU(data[0], data0, gpuNode);
            HSAint64 data1 = ptrace(PTRACE_PEEKDATA, tracePid, mem1, NULL);
            EXPECT_EQ_GPU(0, errno, gpuNode);
            EXPECT_EQ_GPU(data[1], data1, gpuNode);

            /* Swap mem0 and mem1 by poking */
            EXPECT_EQ_GPU(0, ptrace(PTRACE_POKEDATA, tracePid, mem0, reinterpret_cast<void *>(data[1])), gpuNode);
            EXPECT_EQ_GPU(0, errno, gpuNode);
            EXPECT_EQ_GPU(0, ptrace(PTRACE_POKEDATA, tracePid, mem1, reinterpret_cast<void *>(data[0])), gpuNode);
            EXPECT_EQ_GPU(0, errno, gpuNode);
        } catch (...) {
            err = 1;
        }
        r = ptrace(PTRACE_DETACH, tracePid, NULL, NULL);
        if (r) {
            WARN() << "PTRACE_DETACH failed: " << r << std::endl;
            exit(1);
        }
        exit(err);
    } else {
        int childStatus;

        // Parent process, just wait for the child to finish
        EXPECT_EQ_GPU(childPid, waitpid(childPid, &childStatus, 0), gpuNode);
        EXPECT_NE_GPU(0, WIFEXITED(childStatus), gpuNode);
        EXPECT_EQ_GPU(0, WEXITSTATUS(childStatus), gpuNode);
    }

    pthread_mutex_unlock(&ptrace_mtx);

    /* Use shader to read back data to check poke results */
    HsaMemoryBuffer isaBuffer(PAGE_SIZE, gpuNode, true/*zero*/, false/*local*/, true/*exec*/);
    // dstBuffer is cpu accessible gtt memory
    HsaMemoryBuffer dstBuffer(PAGE_SIZE, gpuNode);

    ASSERT_SUCCESS_GPU(m_pAsm->RunAssembleBuf(ScratchCopyDwordIsa, isaBuffer.As<char*>()), gpuNode);

    Dispatch dispatch0(isaBuffer);
    dispatch0.SetArgs(mem0, dstBuffer.As<void*>());
    dispatch0.Submit(queue);
    dispatch0.Sync();
    EXPECT_EQ_GPU(data1[0], dstBuffer.As<unsigned int*>()[0], gpuNode);

    Dispatch dispatch1(isaBuffer);
    dispatch1.SetArgs(mem1, dstBuffer.As<int*>());
    dispatch1.Submit(queue);
    dispatch1.Sync();
    WaitOnValue(dstBuffer.As<uint32_t *>(), data0[0]);
    EXPECT_EQ_GPU(data0[0], dstBuffer.As<unsigned int*>()[0], gpuNode);

    // Clean up
    EXPECT_SUCCESS_GPU(hsaKmtUnmapMemoryToGPU(mem), gpuNode);
    EXPECT_SUCCESS_GPU(hsaKmtFreeMemory(mem, size), gpuNode);
    EXPECT_SUCCESS_GPU(queue.Destroy(), gpuNode);
}

TEST_F(KFDMemoryTest, PtraceAccessInvisibleVram) {
	TEST_START(TESTPROFILE_RUNALL);

    ASSERT_SUCCESS(KFDTest_Launch(PtraceAccessInvisibleVram));

    TEST_END
}

volatile int IntrSignalReceviced;

void CatchSignal(int IntrSignal) {
    IntrSignalReceviced = IntrSignal;
}

static void SignalHandling(KFDTEST_PARAMETERS* pTestParamters) {

	int gpuNode = pTestParamters->gpuNode;
	KFDMemoryTest* pKFDMemoryTest = (KFDMemoryTest*)pTestParamters->pTestObject;

    if (!hsakmt_is_dgpu()) {
        LOG() << "Skipping test: Test not supported on APU." << std::endl;
        return;
    }

    unsigned int *nullPtr = NULL;
    unsigned int* pDb = NULL;
    struct sigaction sa;
    SDMAQueue queue;
    HSAuint64 size, sysMemSize;

    sa.sa_handler = CatchSignal;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;
    pid_t ParentPid = getpid();
    EXPECT_EQ(0, sigaction(SIGUSR1, &sa, NULL)) << "An error occurred while setting a signal handler";

    sysMemSize = pKFDMemoryTest->GetSysMemSize();

    /* System (kernel) memory are limited to 3/8th System RAM
     * Try to allocate 1/4th System RAM
     */
    size = (sysMemSize >> 2) & ~(HSAuint64)(PAGE_SIZE - 1);

    /* We don't need a too large buffer for this test. If it is too large,
     * on some platform, the upcoming hsaKmtAllocMemory() might fail. In
     * order to avoid this flaky behavior, limit the size to 3G.
     */
    size = size > (3ULL << 30) ? (3ULL << 30) : size;

    pKFDMemoryTest->GetHsaMemFlags().ui32.NoNUMABind = 1;
    ASSERT_SUCCESS_GPU(hsaKmtAllocMemory(0 /* system */, size, pKFDMemoryTest->GetHsaMemFlags(), reinterpret_cast<void**>(&pDb)), gpuNode);
    // Verify that pDb is not null before it's being used
    EXPECT_NE_GPU(nullPtr, pDb, gpuNode) << "hsaKmtAllocMemory returned a null pointer";

    pid_t childPid = fork();
    ASSERT_GE_GPU(childPid, 0, gpuNode);
    if (childPid == 0) {
        EXPECT_EQ_GPU(0, kill(ParentPid, SIGUSR1), gpuNode);
        exit(0);
    } else {
        LOG() << "Start Memory Mapping..." << std::endl;
        ASSERT_SUCCESS_GPU(hsaKmtMapMemoryToGPU(pDb, size, NULL), gpuNode);
        LOG() << "Mapping finished" << std::endl;
        int childStatus;
        pid_t pid;

        // Parent process, just wait for the child to finish
        do {
            pid = waitpid(childPid, &childStatus, 0);
            if (IntrSignalReceviced) {
                LOG() << "Interrupt Signal " << std::dec << IntrSignalReceviced
                    << " Received" << std::endl;
                IntrSignalReceviced = 0;
            }
        } while(pid == -1 && errno == EINTR);
        EXPECT_EQ_GPU(childPid, pid, gpuNode);
        EXPECT_NE_GPU(0, WIFEXITED(childStatus), gpuNode);
        EXPECT_EQ_GPU(0, WEXITSTATUS(childStatus), gpuNode);
    }

    pDb[0] = 0x02020202;
    ASSERT_SUCCESS_GPU(queue.Create(gpuNode), gpuNode);
    queue.PlaceAndSubmitPacket(SDMAWriteDataPacket(queue.GetFamilyId(), pDb, 0x01010101) );
    queue.Wait4PacketConsumption();
    EXPECT_TRUE_GPU(WaitOnValue(pDb, 0x01010101), gpuNode);
    EXPECT_SUCCESS_GPU(queue.Destroy(), gpuNode);

    EXPECT_SUCCESS_GPU(hsaKmtUnmapMemoryToGPU(pDb), gpuNode);
    // Release the buffers
    EXPECT_SUCCESS_GPU(hsaKmtFreeMemory(pDb, size), gpuNode);
}

TEST_F(KFDMemoryTest, SignalHandling) {
	TEST_START(TESTPROFILE_RUNALL);

    ASSERT_SUCCESS(KFDTest_Launch(SignalHandling));

    TEST_END
}

static void CheckZeroInitializationSysMem(KFDTEST_PARAMETERS* pTestParamters) {

	int gpuNode = pTestParamters->gpuNode;
	KFDMemoryTest* pKFDMemoryTest = (KFDMemoryTest*)pTestParamters->pTestObject;

    int ret;

    int gpuNum = pKFDMemoryTest->Get_NodeInfo()->GetNodesWithGPU().size();

	/* if no gpu node */
	if (gpuNum <= 0)
		return;

    HSAuint64 sysMemSizeMB = pKFDMemoryTest->GetSysMemSize() >> 20;

    /* Testing system memory */
    HSAuint64 * pDb = NULL;

    HSAuint64 sysBufSizeMB = sysMemSizeMB >> 2;
    HSAuint64 sysBufSize = sysBufSizeMB * 1024 * 1024;

	/* use divided sys ram to test on each gpu to avoid sys ram OOM */
	HSAuint64 sysBufSizePerGPU = sysBufSize/gpuNum;

    int count = 5;

    LOG() << "Using " << std::dec << sysBufSizeMB
            << "MB system buffer to test " << std::dec << count
            << " times" << std::endl;

    unsigned int offset = 257;  // a constant offset, should be smaller than 512.
    unsigned int size = sysBufSizePerGPU / sizeof(*pDb);

    pKFDMemoryTest->GetHsaMemFlags().ui32.NoNUMABind = 1;

    while (count--) {
        ret = hsaKmtAllocMemory(0 /* system */, sysBufSizePerGPU, pKFDMemoryTest->GetHsaMemFlags(),
                                reinterpret_cast<void**>(&pDb));
        if (ret) {
            LOG() << "Failed to allocate system buffer of" << std::dec << sysBufSizeMB
                    << "MB" << std::endl;
            return;
        }

        /* Check the first 64 bits */
        EXPECT_EQ_GPU(0, pDb[0], gpuNode);
        pDb[0] = 1;

        for (HSAuint64 i = offset; i < size;) {
            EXPECT_EQ_GPU(0, pDb[i], gpuNode);
            pDb[i] = i + 1;  // set it to non zero

            i += 4096 / sizeof(*pDb);
        }

        /* check the last 64 bit */
        EXPECT_EQ_GPU(0, pDb[size-1], gpuNode);
        pDb[size-1] = size;

        EXPECT_SUCCESS_GPU(hsaKmtFreeMemory(pDb, sysBufSizePerGPU), gpuNode);
    }
}

TEST_F(KFDMemoryTest, CheckZeroInitializationSysMem) {
	TEST_START(TESTPROFILE_RUNALL);
    TEST_REQUIRE_ENV_CAPABILITIES(ENVCAPS_64BITLINUX);

    ASSERT_SUCCESS(KFDTest_Launch(CheckZeroInitializationSysMem));

    TEST_END
}

static inline void access(volatile void *sd, int size, int rw) {
    /* Most likely sitting in cache*/
    static struct DUMMY {
        char dummy[1024];
    } dummy;

    while ((size -= sizeof(dummy)) >= 0) {
        if (rw == 0)
            dummy = *(struct DUMMY *)((char*)sd + size);
        else
            *(struct DUMMY *)((char*)sd + size) = dummy;
    }
}

/*
 * On large-bar system, test the visible vram access speed.
 * KFD is not allowed to alloc visible vram on non-largebar system.
 */
static void MMBandWidth(KFDTEST_PARAMETERS* pTestParamters) {

	int gpuNode = pTestParamters->gpuNode;
	KFDMemoryTest* pKFDMemoryTest = (KFDMemoryTest*)pTestParamters->pTestObject;

    unsigned nBufs = 1000; /* measure us, report ns */
    unsigned testIndex, sizeIndex, memType;
    const unsigned nMemTypes = 2;
    const char *memTypeStrings[nMemTypes] = {"SysMem", "VRAM"};
    const unsigned nSizes = 4;
    const unsigned bufSizes[nSizes] = {PAGE_SIZE, PAGE_SIZE*4, PAGE_SIZE*16, PAGE_SIZE*64};
    const unsigned nTests = nSizes * nMemTypes;
    const unsigned tmpBufferSize = PAGE_SIZE*64;
#define _TEST_BUFSIZE(index) (bufSizes[index % nSizes])
#define _TEST_MEMTYPE(index) ((index / nSizes) % nMemTypes)

    void *bufs[nBufs];
    HSAuint64 start;
    unsigned i;
    HSAKMT_STATUS ret;
    HsaMemFlags memFlags = {0};
    HsaMemMapFlags mapFlags = {0};

    HSAuint64 vramSizeMB = pKFDMemoryTest->GetVramSize(gpuNode) >> 20;

    LOG() << "Found VRAM of " << std::dec << vramSizeMB << "MB." << std::endl;

    if (!pKFDMemoryTest->Get_NodeInfo()->IsGPUNodeLargeBar(gpuNode) || !vramSizeMB) {
        LOG() << "Skipping test: Test requires a large bar GPU." << std::endl;
        return;
    }

    void *tmp = mmap(0,
            tmpBufferSize,
            PROT_READ | PROT_WRITE,
            MAP_ANONYMOUS | MAP_PRIVATE,
            -1,
            0);
    EXPECT_NE_GPU(tmp, MAP_FAILED, gpuNode);
    memset(tmp, 0, tmpBufferSize);

    LOG() << "Test (avg. ns)\t  memcpyRTime memcpyWTime accessRTime accessWTime" << std::endl;
    for (testIndex = 0; testIndex < nTests; testIndex++) {
        unsigned bufSize = _TEST_BUFSIZE(testIndex);
        unsigned memType = _TEST_MEMTYPE(testIndex);
        HSAuint64 mcpRTime, mcpWTime, accessRTime, accessWTime;
        HSAuint32 allocNode;
        unsigned bufLimit;

        if ((testIndex & (nSizes-1)) == 0)
            LOG() << "----------------------------------------------------------------------" << std::endl;

        if (memType == 0) {
            allocNode = 0;
            memFlags.ui32.PageSize = HSA_PAGE_SIZE_4KB;
            memFlags.ui32.HostAccess = 1;
            memFlags.ui32.NonPaged = 0;
            memFlags.ui32.NoNUMABind = 1;
        } else {
            /* Alloc visible vram*/
            allocNode = gpuNode;
            memFlags.ui32.PageSize = HSA_PAGE_SIZE_4KB;
            memFlags.ui32.HostAccess = 1;
            memFlags.ui32.NonPaged = 1;

	    /* Buffer sizes are 2MB aligned to match new allocation policy.
	     * Upper limit of buffer number to fit 80% vram size.
	     */
            bufLimit = ((vramSizeMB << 20) * 8 / 10) / ALIGN_UP(bufSize, VRAM_ALLOCATION_ALIGN);
            if (bufLimit == 0)
                continue; // skip when bufSize > vram

            /* When vram is too small to fit all the buffers, fill 80% vram size*/
            nBufs = std::min(nBufs , bufLimit);
        }

        for (i = 0; i < nBufs; i++)
            ASSERT_SUCCESS_GPU(hsaKmtAllocMemory(allocNode, bufSize, memFlags,
                        &bufs[i]), gpuNode);

        start = GetSystemTickCountInMicroSec();
        for (i = 0; i < nBufs; i++) {
            memcpy(bufs[i], tmp, bufSize);
        }
        mcpWTime = GetSystemTickCountInMicroSec() - start;

        start = GetSystemTickCountInMicroSec();
        for (i = 0; i < nBufs; i++) {
            access(bufs[i], bufSize, 1);
        }
        accessWTime = GetSystemTickCountInMicroSec() - start;

        start = GetSystemTickCountInMicroSec();
        for (i = 0; i < nBufs; i++) {
            memcpy(tmp, bufs[i], bufSize);
        }
        mcpRTime = GetSystemTickCountInMicroSec() - start;

        start = GetSystemTickCountInMicroSec();
        for (i = 0; i < nBufs; i++) {
            access(bufs[i], bufSize, 0);
        }
        accessRTime = GetSystemTickCountInMicroSec() - start;

        for (i = 0; i < nBufs; i++)
            EXPECT_SUCCESS_GPU(hsaKmtFreeMemory(bufs[i], bufSize), gpuNode);

        LOG() << std::dec
            << std::right << std::setw(3) << (bufSize >> 10) << "K-"
            << std::left << std::setw(14) << memTypeStrings[memType]
            << std::right
            << std::setw(12) << mcpRTime
            << std::setw(12) << mcpWTime
            << std::setw(12) << accessRTime
            << std::setw(12) << accessWTime
            << std::endl;

#define MMBANDWIDTH_KEY_PREFIX memTypeStrings[memType] << "-" \
                               << (bufSize >> 10) << "K" << "-"
        RECORD(mcpRTime) << MMBANDWIDTH_KEY_PREFIX << "mcpRTime";
        RECORD(mcpWTime) << MMBANDWIDTH_KEY_PREFIX << "mcpWTime";
        RECORD(accessRTime) << MMBANDWIDTH_KEY_PREFIX << "accessRTime";
        RECORD(accessWTime) << MMBANDWIDTH_KEY_PREFIX << "accessWTime";

        // skip slow tests
        if (mcpRTime + mcpWTime + accessRTime + accessWTime > 5000000)
            break;
    }

    munmap(tmp, tmpBufferSize);
}

TEST_F(KFDMemoryTest, MMBandWidth) {
	TEST_START(TESTPROFILE_RUNALL);
    TEST_REQUIRE_ENV_CAPABILITIES(ENVCAPS_64BITLINUX);

    ASSERT_SUCCESS(KFDTest_Launch(MMBandWidth));

    TEST_END
}

/* For the purpose of testing HDP flush from CPU.
 * Use CPU to write to coherent vram and check
 * from shader.
 * Asic before gfx9 doesn't support user space
 * HDP flush so only run on vega10 and after.
 * This should only run on large bar system.
 */
static void HostHdpFlush(KFDTEST_PARAMETERS* pTestParamters) {

	int gpuNode = pTestParamters->gpuNode;
	KFDMemoryTest* pKFDMemoryTest = (KFDMemoryTest*)pTestParamters->pTestObject;

	Assembler* m_pAsm;
	m_pAsm = pKFDMemoryTest->GetAssemblerFromNodeId(gpuNode);
	ASSERT_NOTNULL_GPU(m_pAsm, gpuNode);

	HSAuint32 m_FamilyId = pKFDMemoryTest->GetFamilyIdFromNodeId(gpuNode);

    HsaMemFlags memoryFlags = pKFDMemoryTest->GetHsaMemFlags();
    /* buffer[0]: signal; buffer[1]: Input to shader; buffer[2]: Output to
     * shader
     */
    unsigned int *buffer = NULL;
    const HsaNodeProperties *pNodeProperties = pKFDMemoryTest->Get_NodeInfo()->GetNodeProperties(gpuNode);
    HSAuint32 *mmioBase = NULL;
    unsigned int *nullPtr = NULL;

    if (!pNodeProperties) {
        LOG() << "Failed to get gpu node properties." << std::endl;
        return;
    }

    if (m_FamilyId < FAMILY_AI) {
        LOG() << "Skipping test: Test requires gfx9 and later asics." << std::endl;
        return;
    }
    HSAuint64 vramSizeMB = pKFDMemoryTest->GetVramSize(gpuNode) >> 20;

    if (!pKFDMemoryTest->Get_NodeInfo()->IsGPUNodeLargeBar(gpuNode) || !vramSizeMB) {
        LOG() << "Skipping test: Test requires a large bar GPU." << std::endl;
        return;
    }

    HsaMemoryProperties *memoryProperties = new HsaMemoryProperties[pNodeProperties->NumMemoryBanks];
    EXPECT_SUCCESS_GPU(hsaKmtGetNodeMemoryProperties(gpuNode, pNodeProperties->NumMemoryBanks,
                   memoryProperties), gpuNode);
    for (unsigned int bank = 0; bank < pNodeProperties->NumMemoryBanks; bank++) {
        if (memoryProperties[bank].HeapType == HSA_HEAPTYPE_MMIO_REMAP) {
            mmioBase = (unsigned int *)memoryProperties[bank].VirtualBaseAddress;
            break;
        }
    }

    if (mmioBase == nullPtr) {
            LOG() << "Skipping test: bsecause mmioBase is nullPtr, the mmio remap feature is not supported." << std::endl;
            return;
    }

    memoryFlags.ui32.NonPaged = 1;
    memoryFlags.ui32.CoarseGrain = 0;
    ASSERT_SUCCESS_GPU(hsaKmtAllocMemory(gpuNode, PAGE_SIZE, memoryFlags,
                   reinterpret_cast<void**>(&buffer)), gpuNode);
    ASSERT_SUCCESS_GPU(hsaKmtMapMemoryToGPU(buffer, PAGE_SIZE, NULL), gpuNode);

    /* Signal is dead from the beginning*/
    buffer[0] = 0xdead;
    buffer[1] = 0xfeeb;
    buffer[2] = 0xfeed;
    /* Submit a shader to poll the signal*/
    PM4Queue queue;
    ASSERT_SUCCESS_GPU(queue.Create(gpuNode), gpuNode);
    HsaMemoryBuffer isaBuffer(PAGE_SIZE, gpuNode, true/*zero*/, false/*local*/, true/*exec*/);

    ASSERT_SUCCESS_GPU(m_pAsm->RunAssembleBuf(CopyOnSignalIsa, isaBuffer.As<char*>()), gpuNode);

    Dispatch dispatch0(isaBuffer);
    dispatch0.SetArgs(buffer, NULL);
    dispatch0.Submit(queue);

    buffer[1] = 0xbeef;
    /* Flush HDP */
    mmioBase[KFD_MMIO_REMAP_HDP_MEM_FLUSH_CNTL/4] = 0x1;
    buffer[0] = 0xcafe;

    /* Check test result*/
    dispatch0.Sync();
    mmioBase[KFD_MMIO_REMAP_HDP_MEM_FLUSH_CNTL/4] = 0x1;
    EXPECT_EQ_GPU(0xbeef, buffer[2], gpuNode);

    // Clean up
    EXPECT_SUCCESS_GPU(queue.Destroy(), gpuNode);
    delete [] memoryProperties;
    EXPECT_SUCCESS_GPU(hsaKmtUnmapMemoryToGPU(buffer), gpuNode);
    EXPECT_SUCCESS_GPU(hsaKmtFreeMemory(buffer, PAGE_SIZE), gpuNode);
}

TEST_F(KFDMemoryTest, HostHdpFlush) {
    TEST_REQUIRE_ENV_CAPABILITIES(ENVCAPS_64BITLINUX);
	TEST_START(TESTPROFILE_RUNALL);

    ASSERT_SUCCESS(KFDTest_Launch(HostHdpFlush));

    TEST_END
}

/* Test HDP flush from device.
 * Use shader on device 1 to write vram of device 0
 * and flush HDP of device 0. Read vram from device 0
 * and write back to vram to check the result from CPU.
 * Asic before gfx9 doesn't support device HDP flush
 * so only run on vega10 and after.
 * This should only run on system with at least one
 * large bar node (which is used as device 0).
 */
static void DeviceHdpFlush(KFDTEST_PARAMETERS* pTestParamters) {

	int gpuNode = pTestParamters->gpuNode;
	KFDMemoryTest* pKFDMemoryTest = (KFDMemoryTest*)pTestParamters->pTestObject;

	Assembler* m_pAsm;
	m_pAsm = pKFDMemoryTest->GetAssemblerFromNodeId(gpuNode);
	ASSERT_NOTNULL_GPU(m_pAsm, gpuNode);

	HSAuint32 m_FamilyId = pKFDMemoryTest->GetFamilyIdFromNodeId(gpuNode);
    HsaMemFlags memoryFlags = pKFDMemoryTest->GetHsaMemFlags();
    /* buffer is physically on device 0.
     * buffer[0]: Use as signaling b/t devices;
     * buffer[1]: Device 1 write to buffer[1] and device 0 read it
     * buffer[2]: Device 0 copy buffer[1] to buffer[2] for CPU to check
     */
    unsigned int *buffer = NULL;
    const HsaNodeProperties *pNodeProperties;
    HSAuint32 *mmioBase = NULL;
    unsigned int *nullPtr = NULL;
    std::vector<int> nodes;
    int numPeers;

    const std::vector<int> gpuNodes = pKFDMemoryTest->Get_NodeInfo()->GetNodesWithGPU();
    if (gpuNodes.size() < 2) {
        LOG() << "Skipping test: At least two GPUs are required." << std::endl;
        return;
    }

     /* Users can use "--node=gpu1 --dst_node=gpu2" to specify devices */
    if (g_TestDstNodeId != -1 && g_TestNodeId != -1) {
        nodes.push_back(g_TestNodeId);
        nodes.push_back(g_TestDstNodeId);

        if (!pKFDMemoryTest->Get_NodeInfo()->IsPeerAccessibleByNode(g_TestDstNodeId, g_TestNodeId)) {
            LOG() << "Skipping test: first GPU specified is not peer-accessible." << std::endl;
            return;
        }

        if (nodes[0] == nodes[1]) {
            LOG() << "Skipping test: Different GPUs must be specified (2 GPUs required)." << std::endl;
            return;
        }
    } else {
        pKFDMemoryTest->Get_NodeInfo()->FindAccessiblePeers(&nodes, gpuNode);
        if (nodes.size() < 2) {
            LOG() << "Skipping test: Test requires at least one large bar GPU." << std::endl;
            LOG() << "               or two GPUs are XGMI connected." << std::endl;
            return;
        }
    }

    const HsaNodeProperties *pNodePropertiesDev1 = NULL;
    unsigned int m_FamilyIdDev1 = 0;

    pNodeProperties = pKFDMemoryTest->Get_NodeInfo()->GetNodeProperties(nodes[0]);
    pNodePropertiesDev1 = pKFDMemoryTest->Get_NodeInfo()->GetNodeProperties(nodes[1]);
    if (!pNodeProperties || !pNodePropertiesDev1) {
        LOG() << "Failed to get gpu node properties." << std::endl;
        return;
    }

    m_FamilyIdDev1 = FamilyIdFromNode(pNodePropertiesDev1);

    if (m_FamilyId < FAMILY_AI || m_FamilyIdDev1 < FAMILY_AI) {
        LOG() << "Skipping test: Test requires gfx9 and later asics." << std::endl;
        return;
    }

    if (pKFDMemoryTest->Get_NodeInfo()->IsNodeXGMItoCPU(nodes[0])) {
        LOG() << "Skipping test: PCIe link to CPU is required." << std::endl;
        return;
    }

    if (!pKFDMemoryTest->Get_NodeInfo()->IsGPUNodeLargeBar(nodes[0])) {
        LOG() << "Skipping test: Test requires device 0 large bar GPU." << std::endl;
        return;
    }

    HsaMemoryProperties *memoryProperties = new HsaMemoryProperties[pNodeProperties->NumMemoryBanks];
    EXPECT_SUCCESS_GPU(hsaKmtGetNodeMemoryProperties(nodes[0], pNodeProperties->NumMemoryBanks,
                   memoryProperties), gpuNode);
    for (unsigned int bank = 0; bank < pNodeProperties->NumMemoryBanks; bank++) {
        if (memoryProperties[bank].HeapType == HSA_HEAPTYPE_MMIO_REMAP) {
            mmioBase = (unsigned int *)memoryProperties[bank].VirtualBaseAddress;
            break;
        }
    }

    if (mmioBase == nullPtr) {
            LOG() << "Skipping test: bsecause mmioBase is nullPtr, the mmio remap feature is not supported." << std::endl;
            return;
    }

    memoryFlags.ui32.NonPaged = 1;
    memoryFlags.ui32.CoarseGrain = 0;
    ASSERT_SUCCESS_GPU(hsaKmtAllocMemory(nodes[0], PAGE_SIZE, memoryFlags,
                   reinterpret_cast<void**>(&buffer)), gpuNode);
    ASSERT_SUCCESS_GPU(hsaKmtMapMemoryToGPU(buffer, PAGE_SIZE, NULL), gpuNode);

    /* Signal is dead from the beginning*/
    buffer[0] = 0xdead;
    buffer[1] = 0xfeeb;
    buffer[2] = 0xfeeb;
    /* Submit shaders*/
    PM4Queue queue;
    ASSERT_SUCCESS_GPU(queue.Create(nodes[0]), gpuNode);
    HsaMemoryBuffer isaBuffer(PAGE_SIZE, nodes[0], true/*zero*/, false/*local*/, true/*exec*/);

    ASSERT_SUCCESS_GPU(m_pAsm->RunAssembleBuf(CopyOnSignalIsa, isaBuffer.As<char*>()), gpuNode);

    Dispatch dispatch(isaBuffer);
    dispatch.SetArgs(buffer, NULL);
    dispatch.Submit(queue);

    PM4Queue queue0;
    ASSERT_SUCCESS_GPU(queue0.Create(nodes[1]), gpuNode);
    HsaMemoryBuffer isaBuffer0(PAGE_SIZE, nodes[1], true/*zero*/, false/*local*/, true/*exec*/);

    /* Temporarily set target ASIC for Dev1 */
    ASSERT_SUCCESS_GPU(m_pAsm->RunAssembleBuf(WriteAndSignalIsa, isaBuffer0.As<char*>(),
                        PAGE_SIZE, GetGfxVersion(pNodePropertiesDev1)), gpuNode);

    Dispatch dispatch0(isaBuffer0);
    dispatch0.SetArgs(buffer, mmioBase);
    dispatch0.Submit(queue0);

    /* Check test result*/
    dispatch0.Sync();
    dispatch.Sync();
    EXPECT_EQ(0xbeef, buffer[2]);

    // Clean up
    EXPECT_SUCCESS_GPU(queue.Destroy(), gpuNode);
    EXPECT_SUCCESS_GPU(queue0.Destroy(), gpuNode);
    delete [] memoryProperties;
    EXPECT_SUCCESS_GPU(hsaKmtUnmapMemoryToGPU(buffer), gpuNode);
    EXPECT_SUCCESS_GPU(hsaKmtFreeMemory(buffer, PAGE_SIZE), gpuNode);
}

TEST_F(KFDMemoryTest, DeviceHdpFlush) {
    TEST_REQUIRE_ENV_CAPABILITIES(ENVCAPS_64BITLINUX);
	TEST_START(TESTPROFILE_RUNALL);

    ASSERT_SUCCESS(KFDTest_Launch(DeviceHdpFlush));

    TEST_END
}

/* Test should only run on Arcturus series which has the new RW mtype
 * Map a local VRAM with RW mtype (coarse grain for upper layer),
 * read it locally to cache it and write with local SDMA, remote devices(
 * CPU or Remote GPU shader connected with PCIe or XGMI),
 * then read again. The second read should get back what SDMA wrote,
 * since the cache should be invalidated on write and second read
 * should go to physical VRAM instead of cache.
 */
static void CacheInvalidateOnSdmaWrite(KFDTEST_PARAMETERS* pTestParamters) {

	int gpuNode = pTestParamters->gpuNode;
	KFDMemoryTest* pKFDMemoryTest = (KFDMemoryTest*)pTestParamters->pTestObject;

    Assembler* m_pAsm;
    m_pAsm = pKFDMemoryTest->GetAssemblerFromNodeId(gpuNode);
    ASSERT_NOTNULL_GPU(m_pAsm, gpuNode);

    HSAuint32 m_FamilyId = pKFDMemoryTest->GetFamilyIdFromNodeId(gpuNode);

    HsaMemoryBuffer tmpBuffer(PAGE_SIZE, 0, true /* zero */);
    volatile HSAuint32 *tmp = tmpBuffer.As<volatile HSAuint32 *>();
    const int dwLocation = 100;

    if (m_FamilyId != FAMILY_AR) {
        LOG() << "Skipping test: Test requires arcturus series asics." << std::endl;
        return;
    }

    HsaMemoryBuffer buffer(PAGE_SIZE, gpuNode, false/*zero*/, true/*local*/, false/*exec*/);
    SDMAQueue sdmaQueue;
    ASSERT_SUCCESS_GPU(sdmaQueue.Create(gpuNode), gpuNode);
    buffer.Fill(0, sdmaQueue, 0, PAGE_SIZE);
    sdmaQueue.PlacePacket(SDMAWriteDataPacket(sdmaQueue.GetFamilyId(), buffer.As<int*>(), 0x5678));

    /* Read buffer from shader to fill cache */
    PM4Queue queue;
    ASSERT_SUCCESS_GPU(queue.Create(gpuNode), gpuNode);
    HsaMemoryBuffer isaBuffer(PAGE_SIZE, gpuNode, true/*zero*/, false/*local*/, true/*exec*/);

    ASSERT_SUCCESS_GPU(m_pAsm->RunAssembleBuf(PollMemoryIsa, isaBuffer.As<char*>()), gpuNode);

    Dispatch dispatch(isaBuffer);
    dispatch.SetArgs(buffer.As<int*>(), buffer.As<int*>()+dwLocation);
    dispatch.Submit(queue);

    /* Delay 100ms to make sure shader executed*/
    Delay(100);

    /* SDMA writes to buffer. Shader should get what sdma writes and quits*/
    sdmaQueue.SubmitPacket();
    sdmaQueue.Wait4PacketConsumption();

    /* Check test result*/
    dispatch.Sync();
    EXPECT_EQ_GPU(buffer.IsPattern(dwLocation*sizeof(int), 0x5678, sdmaQueue, tmp), true, gpuNode);

    // Clean up
    EXPECT_SUCCESS_GPU(queue.Destroy(), gpuNode);
    EXPECT_SUCCESS_GPU(sdmaQueue.Destroy(), gpuNode);
}

TEST_F(KFDMemoryTest, CacheInvalidateOnSdmaWrite) {
    TEST_REQUIRE_ENV_CAPABILITIES(ENVCAPS_64BITLINUX);
	TEST_START(TESTPROFILE_RUNALL);

    ASSERT_SUCCESS(KFDTest_Launch(CacheInvalidateOnSdmaWrite));

    TEST_END
}

static void CacheInvalidateOnCPUWrite(KFDTEST_PARAMETERS* pTestParamters) {

	int gpuNode = pTestParamters->gpuNode;
	KFDMemoryTest* pKFDMemoryTest = (KFDMemoryTest*)pTestParamters->pTestObject;

    Assembler* m_pAsm;
    m_pAsm = pKFDMemoryTest->GetAssemblerFromNodeId(gpuNode);
    ASSERT_NOTNULL_GPU(m_pAsm, gpuNode);

    HSAuint32 m_FamilyId = pKFDMemoryTest->GetFamilyIdFromNodeId(gpuNode);

    if (m_FamilyId != FAMILY_AR) {
        LOG() << "Skipping test: Test requires arcturus series asics." << std::endl;
        return;
    }

    if (!pKFDMemoryTest->Get_NodeInfo()->IsGPUNodeLargeBar(gpuNode)) {
        LOG() << "Skipping test: Test requires a large bar GPU." << std::endl;
        return;
    }

    int *buffer;
    HsaMemFlags memFlags = {0};
    /* Host accessible vram */
    memFlags.ui32.HostAccess = 1;
    memFlags.ui32.NonPaged = 1;
    memFlags.ui32.CoarseGrain = 1;
    ASSERT_SUCCESS_GPU(hsaKmtAllocMemory(gpuNode, PAGE_SIZE, memFlags, reinterpret_cast<void**>(&buffer)), gpuNode);
    ASSERT_SUCCESS_GPU(hsaKmtMapMemoryToGPU(buffer, PAGE_SIZE, NULL), gpuNode);
    *buffer = 0;

    /* Read buffer from shader to fill cache */
    PM4Queue queue;
    ASSERT_SUCCESS_GPU(queue.Create(gpuNode), gpuNode);
    HsaMemoryBuffer isaBuffer(PAGE_SIZE, gpuNode, true/*zero*/, false/*local*/, true/*exec*/);

    ASSERT_SUCCESS_GPU(m_pAsm->RunAssembleBuf(PollMemoryIsa, isaBuffer.As<char*>()), gpuNode);

    Dispatch dispatch(isaBuffer);
    dispatch.SetArgs(buffer, buffer+100);
    dispatch.Submit(queue);

    /* Delay 100ms to make sure shader executed*/
    Delay(100);

    /* CPU writes to buffer. Shader should get what CPU writes and quits*/
    *buffer = 0x5678;

    /* Check test result*/
    dispatch.Sync();
    EXPECT_EQ_GPU(buffer[100], 0x5678, gpuNode);

    // Clean up
    EXPECT_SUCCESS_GPU(hsaKmtUnmapMemoryToGPU(buffer), gpuNode);
    EXPECT_SUCCESS_GPU(hsaKmtFreeMemory(buffer, PAGE_SIZE), gpuNode);
    EXPECT_SUCCESS_GPU(queue.Destroy(), gpuNode);
}

TEST_F(KFDMemoryTest, CacheInvalidateOnCPUWrite) {
    TEST_REQUIRE_ENV_CAPABILITIES(ENVCAPS_64BITLINUX);
	TEST_START(TESTPROFILE_RUNALL);

    ASSERT_SUCCESS(KFDTest_Launch(CacheInvalidateOnCPUWrite));

    TEST_END
}

static void CacheInvalidateOnRemoteWrite(KFDTEST_PARAMETERS* pTestParamters) {

	int gpuNode = pTestParamters->gpuNode;
	KFDMemoryTest* pKFDMemoryTest = (KFDMemoryTest*)pTestParamters->pTestObject;

    Assembler* m_pAsm;
    m_pAsm = pKFDMemoryTest->GetAssemblerFromNodeId(gpuNode);
    ASSERT_NOTNULL_GPU(m_pAsm, gpuNode);

    HSAuint32 m_FamilyId = pKFDMemoryTest->GetFamilyIdFromNodeId(gpuNode);

    HsaMemoryBuffer tmpBuffer(PAGE_SIZE, 0, true /* zero */);
    volatile HSAuint32 *tmp = tmpBuffer.As<volatile HSAuint32 *>();
    const int dwLocation = 100;
    const int dwLocation1 = 50;

    if (m_FamilyId != FAMILY_AR) {
        LOG() << "Skipping test: Test requires arcturus series asics." << std::endl;
        return;
    }

    const std::vector<int> gpuNodes = pKFDMemoryTest->Get_NodeInfo()->GetNodesWithGPU();
    if (gpuNodes.size() < 2) {
        LOG() << "Skipping test: At least two GPUs are required." << std::endl;
        return;
    }

    HSAuint32 nondefaultNode;
    for (unsigned i = 0; i < gpuNodes.size(); i++) {
        if (gpuNodes.at(i) != gpuNode) {
            nondefaultNode = gpuNodes.at(i);
            break;
        }
    }

    HsaMemoryBuffer buffer(PAGE_SIZE, gpuNode, false/*zero*/, true/*local*/, false/*exec*/);
    buffer.MapMemToNodes(&nondefaultNode, 1);
    SDMAQueue sdmaQueue;
    ASSERT_SUCCESS_GPU(sdmaQueue.Create(gpuNode), gpuNode);
    buffer.Fill(0, sdmaQueue, 0, PAGE_SIZE);

    /* Read buffer from shader to fill cache */
    PM4Queue queue;
    ASSERT_SUCCESS_GPU(queue.Create(gpuNode), gpuNode);
    HsaMemoryBuffer isaBuffer(PAGE_SIZE, gpuNode, true/*zero*/, false/*local*/, true/*exec*/);

    ASSERT_SUCCESS_GPU(m_pAsm->RunAssembleBuf(PollMemoryIsa, isaBuffer.As<char*>()), gpuNode);

    Dispatch dispatch(isaBuffer);
    dispatch.SetArgs(buffer.As<int*>(), buffer.As<int*>()+dwLocation);
    dispatch.Submit(queue);

    /* Delay 100ms to make sure shader executed*/
    Delay(100);

    /* Using a remote shader to copy data from dwLocation1 to the beginning of the buffer.
     * Local shader should get what remote writes and quits
     */
    PM4Queue queue1;
    ASSERT_SUCCESS_GPU(queue1.Create(nondefaultNode), gpuNode);
    buffer.Fill(0x5678, sdmaQueue, dwLocation1*sizeof(int), 4);
    HsaMemoryBuffer isaBuffer1(PAGE_SIZE, nondefaultNode, true/*zero*/, false/*local*/, true/*exec*/);

    ASSERT_SUCCESS_GPU(m_pAsm->RunAssembleBuf(CopyDwordIsa, isaBuffer.As<char*>()), gpuNode);

    Dispatch dispatch1(isaBuffer1);
    dispatch1.SetArgs(buffer.As<int*>()+dwLocation1, buffer.As<int*>());
    dispatch1.Submit(queue1);
    dispatch1.Sync(g_TestTimeOut);

    /* Check test result*/
    dispatch.Sync();
    EXPECT_EQ_GPU(buffer.IsPattern(dwLocation*sizeof(int), 0x5678, sdmaQueue, tmp), true, gpuNode);

    // Clean up
    EXPECT_SUCCESS_GPU(queue.Destroy(), gpuNode);
    EXPECT_SUCCESS_GPU(queue1.Destroy(), gpuNode);
    EXPECT_SUCCESS_GPU(sdmaQueue.Destroy(), gpuNode);
}

TEST_F(KFDMemoryTest, CacheInvalidateOnRemoteWrite) {
    TEST_REQUIRE_ENV_CAPABILITIES(ENVCAPS_64BITLINUX);
	TEST_START(TESTPROFILE_RUNALL);

    ASSERT_SUCCESS(KFDTest_Launch(CacheInvalidateOnRemoteWrite));

    TEST_END
}

/* Test is for new cache coherence on Aldebaran. It is to verify
 * two GPUs can coherently share a fine grain FB.
 */
static void VramCacheCoherenceWithRemoteGPU(KFDTEST_PARAMETERS* pTestParamters) {

	int gpuNode = pTestParamters->gpuNode;
	KFDMemoryTest* pKFDMemoryTest = (KFDMemoryTest*)pTestParamters->pTestObject;

    Assembler* m_pAsm;
    m_pAsm = pKFDMemoryTest->GetAssemblerFromNodeId(gpuNode);
    ASSERT_NOTNULL_GPU(m_pAsm, gpuNode);

    HSAuint32 m_FamilyId = pKFDMemoryTest->GetFamilyIdFromNodeId(gpuNode);

    HsaMemoryBuffer tmpBuffer(PAGE_SIZE, 0, true /* zero */);
    volatile HSAuint32 *tmp = tmpBuffer.As<volatile HSAuint32 *>();
    const int dwSource = 0x40 * sizeof(int); /* At 3rd cache line */
    const int dwLocation = 0x80 * sizeof(int); /* At 5th cache line  */

    if (m_FamilyId != FAMILY_AL && m_FamilyId != FAMILY_AV) {
        LOG() << "Skipping test: Test requires aldebaran or aqua vanjaram series asics." << std::endl;
        return;
    }

    const std::vector<int> gpuNodes = pKFDMemoryTest->Get_NodeInfo()->GetNodesWithGPU();
    if (gpuNodes.size() < 2) {
        LOG() << "Skipping test: At least two GPUs are required." << std::endl;
        return;
    }

    HSAuint32 nondefaultNode;
    for (unsigned i = 0; i < gpuNodes.size(); i++) {
        if (gpuNodes.at(i) != gpuNode) {
            nondefaultNode = gpuNodes.at(i);
            break;
        }
    }

    unsigned int nodes[2] = {(HSAuint32)gpuNode, nondefaultNode};

    /* Allocate a local FB */
    HsaMemoryBuffer buffer(PAGE_SIZE, gpuNode, false/*zero*/, true/*local*/, false/*exec*/);
    buffer.MapMemToNodes(&nodes[0], 2);
    SDMAQueue sdmaQueue;
    ASSERT_SUCCESS_GPU(sdmaQueue.Create(gpuNode), gpuNode);
    buffer.Fill(0, sdmaQueue, 0, PAGE_SIZE);
    buffer.Fill(0x5678, sdmaQueue, dwSource, 4);

    /* Read buffer[0] as flag from local shader to fill cache line (64 dws)
     * which should has 0 at buffer[1]
     */
    PM4Queue queue;
    ASSERT_SUCCESS_GPU(queue.Create(gpuNode), gpuNode);
    HsaMemoryBuffer isaBuffer(PAGE_SIZE, gpuNode, true/*zero*/, false/*local*/, true/*exec*/);

    ASSERT_SUCCESS_GPU(m_pAsm->RunAssembleBuf(PollAndCopyIsa, isaBuffer.As<char*>()), gpuNode);

    Dispatch dispatch(isaBuffer);
    dispatch.SetArgs(buffer.As<char *>(), buffer.As<char *>()+dwLocation);
    dispatch.Submit(queue);

    /* Delay 100ms to make sure shader executed*/
    Delay(100);

    /* Using remote shader to write the flag and copy value from dwSource
     * to dwLocation in buffer.
     * Local shader should get the flag and execute CopyMemory
     */
    PM4Queue queue1;
    ASSERT_SUCCESS_GPU(queue1.Create(nondefaultNode), gpuNode);
    HsaMemoryBuffer isaBuffer1(PAGE_SIZE, nondefaultNode, true/*zero*/, false/*local*/, true/*exec*/);

    ASSERT_SUCCESS_GPU(m_pAsm->RunAssembleBuf(WriteFlagAndValueIsa, isaBuffer1.As<char*>()), gpuNode);

    Dispatch dispatch1(isaBuffer1);
    dispatch1.SetArgs(buffer.As<char *>(), buffer.As<char *>()+dwSource);
    dispatch1.Submit(queue1);
    dispatch1.Sync(g_TestTimeOut);

    /* Check test result*/
    dispatch.Sync(g_TestTimeOut);
    EXPECT_EQ_GPU(buffer.IsPattern(dwLocation, 0x5678, sdmaQueue, tmp), true, gpuNode);

    // Clean up
    EXPECT_SUCCESS_GPU(queue.Destroy(), gpuNode);
    EXPECT_SUCCESS_GPU(queue1.Destroy(), gpuNode);
    EXPECT_SUCCESS_GPU(sdmaQueue.Destroy(), gpuNode);
}

TEST_F(KFDMemoryTest, VramCacheCoherenceWithRemoteGPU) {
    TEST_REQUIRE_ENV_CAPABILITIES(ENVCAPS_64BITLINUX);
	TEST_START(TESTPROFILE_RUNALL);

    ASSERT_SUCCESS(KFDTest_Launch(VramCacheCoherenceWithRemoteGPU));

    TEST_END
}

/* Test is for new cache coherence on A+A(Aldebaran). It is to verify
 * new XGMI coherence HW link in caches between CPU and GPUs
 * in local FB with fine grain mode.
 */
static void VramCacheCoherenceWithCPU(KFDTEST_PARAMETERS* pTestParamters) {

	int gpuNode = pTestParamters->gpuNode;
	KFDMemoryTest* pKFDMemoryTest = (KFDMemoryTest*)pTestParamters->pTestObject;

    Assembler* m_pAsm;
    m_pAsm = pKFDMemoryTest->GetAssemblerFromNodeId(gpuNode);
    ASSERT_NOTNULL_GPU(m_pAsm, gpuNode);

    HSAuint32 m_FamilyId = pKFDMemoryTest->GetFamilyIdFromNodeId(gpuNode);

    if (m_FamilyId != FAMILY_AL && m_FamilyId != FAMILY_AV) {
        LOG() << "Skipping test: Test requires aldebaran or aqua vanjaram series asics." << std::endl;
        return;
    }

    const int dwLocation = 0x80;

    if (!pKFDMemoryTest->Get_NodeInfo()->IsNodeXGMItoCPU(gpuNode)) {
        LOG() << "Skipping test: XGMI link to CPU is required." << std::endl;
        return;
    }

    unsigned int *buffer;
    HsaMemFlags memFlags = {0};
    /* Allocate a fine grain local FB accessed by CPU */
    memFlags.ui32.HostAccess = 1;
    memFlags.ui32.NonPaged = 1;
    ASSERT_SUCCESS_GPU(hsaKmtAllocMemory(gpuNode, PAGE_SIZE, memFlags,
            reinterpret_cast<void**>(&buffer)), gpuNode);
    ASSERT_SUCCESS_GPU(hsaKmtMapMemoryToGPU(buffer, PAGE_SIZE, NULL), gpuNode);
    buffer[0] = 0;
    buffer[dwLocation] = 0;

    /* Read buffer from shader to fill cache */
    PM4Queue queue;
    ASSERT_SUCCESS_GPU(queue.Create(gpuNode), gpuNode);
    HsaMemoryBuffer isaBuffer(PAGE_SIZE, gpuNode, true/*zero*/, false/*local*/, true/*exec*/);

    ASSERT_SUCCESS_GPU(m_pAsm->RunAssembleBuf(PollAndCopyIsa, isaBuffer.As<char*>()), gpuNode);

    Dispatch dispatch(isaBuffer);
    dispatch.SetArgs(buffer, buffer+dwLocation);
    dispatch.Submit(queue);

    /* Delay 100ms to make sure shader executed*/
    Delay(100);

    /* CPU writes to buffer. Shader should get 0x5678 CPU writes
     * after cache invalidating(buffer_invl2) and quits
     */
    buffer[1] = 0x5678;
    buffer[0] = 1;

    /* Check test result*/
    dispatch.Sync(g_TestTimeOut);
    EXPECT_EQ_GPU(buffer[dwLocation], 0x5678, gpuNode);

    // Clean up
    EXPECT_SUCCESS_GPU(hsaKmtUnmapMemoryToGPU(buffer), gpuNode);
    EXPECT_SUCCESS_GPU(hsaKmtFreeMemory(buffer, PAGE_SIZE), gpuNode);
    EXPECT_SUCCESS_GPU(queue.Destroy(), gpuNode);
}

TEST_F(KFDMemoryTest, VramCacheCoherenceWithCPU) {
    TEST_REQUIRE_ENV_CAPABILITIES(ENVCAPS_64BITLINUX);
	TEST_START(TESTPROFILE_RUNALL);

    ASSERT_SUCCESS(KFDTest_Launch(VramCacheCoherenceWithCPU));

    TEST_END
}

/* Test is for new cache coherence on Aldebaran. It is to verify
 * new XGMI coherence HW link in caches between CPU and GPUs
 * in system RAM.
 */
static void SramCacheCoherenceWithGPU(KFDTEST_PARAMETERS* pTestParamters) {

	int gpuNode = pTestParamters->gpuNode;
	KFDMemoryTest* pKFDMemoryTest = (KFDMemoryTest*)pTestParamters->pTestObject;

    Assembler* m_pAsm;
    m_pAsm = pKFDMemoryTest->GetAssemblerFromNodeId(gpuNode);
    ASSERT_NOTNULL_GPU(m_pAsm, gpuNode);

    HSAuint32 m_FamilyId = pKFDMemoryTest->GetFamilyIdFromNodeId(gpuNode);

    if (m_FamilyId != FAMILY_AL && m_FamilyId != FAMILY_AV) {
        LOG() << "Skipping test: Test requires aldebaran or aqua vanjaram series asics." << std::endl;
        return;
    }

    const int dwLocation = 0x80;

    if (!pKFDMemoryTest->Get_NodeInfo()->IsNodeXGMItoCPU(gpuNode)) {
        LOG() << "Skipping test: XGMI link to CPU is required." << std::endl;
        return;
    }

    unsigned int *fineBuffer = NULL;
    unsigned int tmp;

    ASSERT_SUCCESS_GPU(hsaKmtAllocMemory(gpuNode /* system */, PAGE_SIZE, pKFDMemoryTest->GetHsaMemFlags(),
                       reinterpret_cast<void**>(&fineBuffer)), gpuNode);
    ASSERT_SUCCESS_GPU(hsaKmtMapMemoryToGPU(fineBuffer, PAGE_SIZE, NULL), gpuNode);
    fineBuffer[0] = 0;
    fineBuffer[1] = 0;
    /* Read buffer from CPU to fill cache */
    tmp = fineBuffer[dwLocation];

    /* Read fine grain buffer from shader to fill cache */
    PM4Queue queue;
    ASSERT_SUCCESS_GPU(queue.Create(gpuNode), gpuNode);
    HsaMemoryBuffer isaBuffer(PAGE_SIZE, gpuNode, true/*zero*/, false/*local*/, true/*exec*/);

    ASSERT_SUCCESS_GPU(m_pAsm->RunAssembleBuf(PollAndCopyIsa, isaBuffer.As<char*>()), gpuNode);

    Dispatch dispatch(isaBuffer);
    dispatch.SetArgs(fineBuffer, fineBuffer+dwLocation);
    dispatch.Submit(queue);

    /* Delay 100ms to make sure shader executed*/
    Delay(100);

    /* CPU writes to buffer. Shader should get what CPU writes and quits*/
    fineBuffer[1] = 0x5678;
    fineBuffer[0] = 1;

    /* Check test result, based on KFDEventTest.SignalEvent passed.
     * if Sync times out,
     * it means coherence issue that GPU doesn't read what CPU wrote.
     * if buffer value is not expected,
     * it means coherence issue that CPU doesn't read what GPU wrote.
     */
    dispatch.Sync(g_TestTimeOut);
    EXPECT_EQ_GPU(fineBuffer[dwLocation], 0x5678, gpuNode);

    // Clean up
    EXPECT_SUCCESS_GPU(hsaKmtUnmapMemoryToGPU(fineBuffer), gpuNode);
    EXPECT_SUCCESS_GPU(hsaKmtFreeMemory(fineBuffer, PAGE_SIZE), gpuNode);
    EXPECT_SUCCESS_GPU(queue.Destroy(), gpuNode);
}

TEST_F(KFDMemoryTest, SramCacheCoherenceWithGPU) {
    TEST_REQUIRE_ENV_CAPABILITIES(ENVCAPS_64BITLINUX);
	TEST_START(TESTPROFILE_RUNALL);

    ASSERT_SUCCESS(KFDTest_Launch(SramCacheCoherenceWithGPU));

    TEST_END
}

void KFDMemoryTest::AcquireReleaseTestRunCPU(HSAuint32 acquireNode, bool scalar) {

    LOG() << "Testing coherency from CPU to node " << std::dec << acquireNode << std::endl;

    /* Allocate shared buffer - must be at least 64 * 6 bytes */
    HsaMemoryBuffer buffer(PAGE_SIZE, acquireNode, false/*zero*/, false/*local*/, false/*exec*/);
    buffer.MapMemToNodes(&acquireNode, 1);

    /* Allocate output buffer and insert magic numbers */
    HsaMemoryBuffer outputBuffer(PAGE_SIZE, acquireNode, true, false, false);
    outputBuffer.As<char *>()[0x40] = 99;
    outputBuffer.As<char *>()[0x80] = 99;
    outputBuffer.As<char *>()[0xc0] = 99;
    outputBuffer.As<char *>()[0x100] = 99;
    outputBuffer.As<char *>()[0x140] = 99;

    /* Flush results of previous tests from the buffer */
    /* This would be done with SDMA, but SDMA doesn't work on some Aqua Vanjaram emulators */
    PM4Queue flushQueue;
    ASSERT_SUCCESS(flushQueue.Create(acquireNode));
    HsaMemoryBuffer flushBuffer(PAGE_SIZE, acquireNode, true/*zero*/, false/*local*/, true/*exec*/);
    ASSERT_SUCCESS(m_pAsm->RunAssembleBuf(FlushBufferForAcquireReleaseIsa, flushBuffer.As<char*>()));
    Dispatch flushDispatch(flushBuffer);
    flushDispatch.SetArgs(buffer.As<char *>(), NULL);
    flushDispatch.SetDim(1, 1, 1);
    flushDispatch.Submit(flushQueue);
    flushDispatch.Sync(g_TestTimeOut);

    /* Start acquiring thread */
    PM4Queue acquireQueue;
    ASSERT_SUCCESS(acquireQueue.Create(acquireNode));
    HsaMemoryBuffer acquireBuffer(PAGE_SIZE, acquireNode, true/*zero*/, false/*local*/, true/*exec*/);
    if (!scalar)
        ASSERT_SUCCESS(m_pAsm->RunAssembleBuf(ReadAcquireVectorIsa, acquireBuffer.As<char*>()));
    else
        ASSERT_SUCCESS(m_pAsm->RunAssembleBuf(ReadAcquireScalarIsa, acquireBuffer.As<char*>()));
    Dispatch acquireDispatch(acquireBuffer);
    acquireDispatch.SetArgs(buffer.As<char *>(), outputBuffer.As<char *>());
    acquireDispatch.SetDim(1, 1, 1);
    acquireDispatch.Submit(acquireQueue);

    /* Delay 100ms to ensure acquirer is waiting */
    Delay(100);

    if (!scalar) {
        buffer.As<char *>()[0x40] = 0x1;
        buffer.As<char *>()[0x80] = 0x2;
        buffer.As<char *>()[0xc0] = 0x3;
        buffer.As<char *>()[0x100] = 0x4;
        buffer.As<char *>()[0x140] = 0x5;
    } else {
        buffer.As<char *>()[0x40] = 0x6;
        buffer.As<char *>()[0x80] = 0x7;
        buffer.As<char *>()[0xc0] = 0x8;
        buffer.As<char *>()[0x100] = 0x9;
        buffer.As<char *>()[0x140] = 0xa;
    }
    buffer.As<char *>()[0x0] = 0x1;

    acquireDispatch.Sync(g_TestTimeOut);

    /* Check test result*/
    if (!scalar) {
        EXPECT_EQ(0x1, outputBuffer.As<char *>()[0x40]);
        EXPECT_EQ(0x2, outputBuffer.As<char *>()[0x80]);
        EXPECT_EQ(0x3, outputBuffer.As<char *>()[0xc0]);
        EXPECT_EQ(0x4, outputBuffer.As<char *>()[0x100]);
        EXPECT_EQ(0x5, outputBuffer.As<char *>()[0x140]);
    } else {
        EXPECT_EQ(0x6, outputBuffer.As<char *>()[0x40]);
        EXPECT_EQ(0x7, outputBuffer.As<char *>()[0x80]);
        EXPECT_EQ(0x8, outputBuffer.As<char *>()[0xc0]);
        EXPECT_EQ(0x9, outputBuffer.As<char *>()[0x100]);
        EXPECT_EQ(0xa, outputBuffer.As<char *>()[0x140]);
    }

    /*
     * Guide to results:
     * 0x99: acquiring shader did not write to output buffer at all
     * 0x77: coherency error. Either releasing shader did not write or acquiring shader read stale value
     * All five EXPECT_EQ fail: error occurs even when releasing shader bypasses cache
     * Only first four EXPECT_EQ fail: error occurs only when releasing shader uses cache
     */

    /* Clean up */
    EXPECT_SUCCESS(acquireQueue.Destroy());
    EXPECT_SUCCESS(flushQueue.Destroy());
}

void KFDMemoryTest::AcquireReleaseTestRun(HSAuint32 acquireNode, HSAuint32 releaseNode,
                                          bool localToRemote, bool scalar) {

    LOG() << "Testing coherency from node " << std::dec << releaseNode << " to node " << std::dec << acquireNode << std::endl;

    /* Allocate shared buffer - must be at least 64 * 6 bytes */
    HSAuint32 localNode;
    if (!localToRemote)
        localNode = acquireNode;
    else
        localNode = releaseNode;
    HsaMemoryBuffer buffer(PAGE_SIZE, localNode, false/*zero*/, true/*local*/, false/*exec*/);
    unsigned int nodes[2] = {acquireNode, releaseNode};
    buffer.MapMemToNodes(&nodes[0], 2);

    /* Allocate output buffer and insert magic numbers */
    HsaMemoryBuffer outputBuffer(PAGE_SIZE, acquireNode, true, false, false);
    outputBuffer.As<char *>()[0x40] = 99;
    outputBuffer.As<char *>()[0x80] = 99;
    outputBuffer.As<char *>()[0xc0] = 99;
    outputBuffer.As<char *>()[0x100] = 99;
    outputBuffer.As<char *>()[0x140] = 99;

    /* Flush results of previous tests from the buffer */
    /* This would be done with SDMA, but SDMA doesn't work on some Aqua Vanjaram emulators */
    PM4Queue flushQueue;
    ASSERT_SUCCESS(flushQueue.Create(acquireNode));
    HsaMemoryBuffer flushBuffer(PAGE_SIZE, acquireNode, true/*zero*/, false/*local*/, true/*exec*/);
    ASSERT_SUCCESS(m_pAsm->RunAssembleBuf(FlushBufferForAcquireReleaseIsa, flushBuffer.As<char*>()));
    Dispatch flushDispatch(flushBuffer);
    flushDispatch.SetArgs(buffer.As<char *>(), NULL);
    flushDispatch.SetDim(1, 1, 1);
    flushDispatch.Submit(flushQueue);
    flushDispatch.Sync(g_TestTimeOut);

    /* Start acquiring thread */
    PM4Queue acquireQueue;
    ASSERT_SUCCESS(acquireQueue.Create(acquireNode));
    HsaMemoryBuffer acquireBuffer(PAGE_SIZE, acquireNode, true/*zero*/, false/*local*/, true/*exec*/);
    if (!scalar)
        ASSERT_SUCCESS(m_pAsm->RunAssembleBuf(ReadAcquireVectorIsa, acquireBuffer.As<char*>()));
    else
        ASSERT_SUCCESS(m_pAsm->RunAssembleBuf(ReadAcquireScalarIsa, acquireBuffer.As<char*>()));
    Dispatch acquireDispatch(acquireBuffer);
    acquireDispatch.SetArgs(buffer.As<char *>(), outputBuffer.As<char *>());
    acquireDispatch.SetDim(1, 1, 1);
    acquireDispatch.Submit(acquireQueue);

    /* Delay 100ms to ensure acquirer is waiting */
    Delay(100);

    /* Start releasing thread */
    PM4Queue releaseQueue;
    ASSERT_SUCCESS(releaseQueue.Create(releaseNode));
    HsaMemoryBuffer releaseBuffer(PAGE_SIZE, releaseNode, true/*zero*/, false/*local*/, true/*exec*/);
    if (!scalar)
        ASSERT_SUCCESS(m_pAsm->RunAssembleBuf(WriteReleaseVectorIsa, releaseBuffer.As<char*>()));
    else
        ASSERT_SUCCESS(m_pAsm->RunAssembleBuf(WriteReleaseScalarIsa, releaseBuffer.As<char*>()));
    Dispatch releaseDispatch(releaseBuffer);
    releaseDispatch.SetArgs(buffer.As<char *>(), NULL);
    releaseDispatch.SetDim(1, 1, 1);
    releaseDispatch.Submit(releaseQueue);

    /* Wait for threads to finish */
    releaseDispatch.Sync(g_TestTimeOut);
    acquireDispatch.Sync(g_TestTimeOut);

    /* Check test result*/
    if (!scalar) {
        EXPECT_EQ(0x1, outputBuffer.As<char *>()[0x40]);
        EXPECT_EQ(0x2, outputBuffer.As<char *>()[0x80]);
        EXPECT_EQ(0x3, outputBuffer.As<char *>()[0xc0]);
        EXPECT_EQ(0x4, outputBuffer.As<char *>()[0x100]);
        EXPECT_EQ(0x5, outputBuffer.As<char *>()[0x140]);
    } else {
        EXPECT_EQ(0x6, outputBuffer.As<char *>()[0x40]);
        EXPECT_EQ(0x7, outputBuffer.As<char *>()[0x80]);
        EXPECT_EQ(0x8, outputBuffer.As<char *>()[0xc0]);
        EXPECT_EQ(0x9, outputBuffer.As<char *>()[0x100]);
        EXPECT_EQ(0xa, outputBuffer.As<char *>()[0x140]);
    }

    /*
     * Guide to results:
     * 0x99: acquiring shader did not write to output buffer at all
     * 0x77: coherency error. Either releasing shader did not write or acquiring shader read stale value
     * All five EXPECT_EQ fail: error occurs even when releasing shader bypasses cache
     * Only first four EXPECT_EQ fail: error occurs only when releasing shader uses cache
     */

    /* Clean up */
    EXPECT_SUCCESS(acquireQueue.Destroy());
    EXPECT_SUCCESS(releaseQueue.Destroy());
    EXPECT_SUCCESS(flushQueue.Destroy());
}

/* A test of the memory coherence features on Aqua_Vanjaram.
 * One shader stores values at 5 positions in memory, then performs
 * a write-release. The other shader performs a read-acquire, then loads
 * those 5 values, then stores them in a CPU-visible buffer
 *
 * withinGPU: When true, the two shaders will be loaded onto two nodes within
 *            the same GPU. When false, the two shaders will be loaded onto different
 *            GPUs.
 *
 * localToRemote: When true, the shared memory will be local to the releasing node.
 *                When false, the shared memory will be local to the acquiring node.
 *
 * scalar: When true, the shared data will be stored and loaded with scalar instructions.
 *         When false, the shared data will be stored and loaded with vector instructions.
 */
void KFDMemoryTest::AcquireReleaseTest(bool withinGPU, bool localToRemote, bool scalar) {

    if (m_FamilyId != FAMILY_AV) {
        LOG() << "Skipping test: Test requires aqua vanjaram series asics." << std::endl;
        return;
    }

    /* Find second node - nodes with the same DrmRenderMinor are on the same GPU */
    const std::vector<int> gpuNodes = m_NodeInfo.GetNodesWithGPU();
    HSAuint32 acquireNode;
    HSAint32 acquireDRM;
    bool foundSecondNode = false;
    for (unsigned i = 0; i < gpuNodes.size(); i++) {
        acquireNode = gpuNodes.at(i);
        acquireDRM = m_NodeInfo.GetNodeProperties(acquireNode)->DrmRenderMinor;
        for (unsigned j = 0; j < gpuNodes.size(); j++) {
            if (!withinGPU) {
                if (m_NodeInfo.GetNodeProperties(gpuNodes.at(j))->DrmRenderMinor != acquireDRM) {
                    foundSecondNode = true;
                    AcquireReleaseTestRun(acquireNode, gpuNodes.at(j), localToRemote, scalar);
                }
            } else {
                if (m_NodeInfo.GetNodeProperties(gpuNodes.at(j))->DrmRenderMinor == acquireDRM && gpuNodes.at(j) != acquireNode) {
                    foundSecondNode = true;
                    AcquireReleaseTestRun(acquireNode, gpuNodes.at(j), localToRemote, scalar);
                }
            }
        }
    }
    if (!foundSecondNode) {
        if (!withinGPU) {
            LOG() << "Skipping test: At least two GPUs are required." << std::endl;
        } else {
            LOG() << "Skipping test: At least two nodes on the same GPU are required." << std::endl;
        }

    }
}

TEST_F(KFDMemoryTest, AcquireReleaseCPU) {
    if (m_FamilyId != FAMILY_AV) {
        LOG() << "Skipping test: Test requires aqua vanjaram series asics." << std::endl;
        return;
    }

    /* Find second node - nodes with the same DrmRenderMinor are on the same GPU */
    const std::vector<int> gpuNodes = m_NodeInfo.GetNodesWithGPU();
    HSAuint32 acquireNode;
    for (unsigned i = 0; i < gpuNodes.size(); i++) {
        acquireNode = gpuNodes.at(i);
        AcquireReleaseTestRunCPU(acquireNode, true);
        AcquireReleaseTestRunCPU(acquireNode, false);
    }
}

TEST_F(KFDMemoryTest, AcquireReleaseFarLocalVector) {
    TEST_REQUIRE_ENV_CAPABILITIES(ENVCAPS_64BITLINUX);
    TEST_START(TESTPROFILE_RUNALL);

    AcquireReleaseTest(false /* multi-GPU */, false /* acquirer is local */, false /* vector */);

    TEST_END
}

TEST_F(KFDMemoryTest, AcquireReleaseFarLocalScalar) {
    TEST_REQUIRE_ENV_CAPABILITIES(ENVCAPS_64BITLINUX);
    TEST_START(TESTPROFILE_RUNALL);

    AcquireReleaseTest(false /* multi-GPU */, false /* acquirer is local */, true /* scalar */);

    TEST_END
}

TEST_F(KFDMemoryTest, AcquireReleaseFarRemoteVector) {
    TEST_REQUIRE_ENV_CAPABILITIES(ENVCAPS_64BITLINUX);
    TEST_START(TESTPROFILE_RUNALL);

    AcquireReleaseTest(false /* multi-GPU */, true /* releaser is local */, false /* vector */);

    TEST_END
}

TEST_F(KFDMemoryTest, AcquireReleaseFarRemoteScalar) {
    TEST_REQUIRE_ENV_CAPABILITIES(ENVCAPS_64BITLINUX);
    TEST_START(TESTPROFILE_RUNALL);

    AcquireReleaseTest(false /* multi-GPU */, true /* releaser is local */, true /* scalar */);

    TEST_END
}

TEST_F(KFDMemoryTest, AcquireReleaseCloseLocalVector) {
    TEST_REQUIRE_ENV_CAPABILITIES(ENVCAPS_64BITLINUX);
    TEST_START(TESTPROFILE_RUNALL);

    AcquireReleaseTest(true /* within-GPU */, false /* acquirer is local */, false /* vector */);

    TEST_END
}

TEST_F(KFDMemoryTest, AcquireReleaseCloseLocalScalar) {
    TEST_REQUIRE_ENV_CAPABILITIES(ENVCAPS_64BITLINUX);
    TEST_START(TESTPROFILE_RUNALL);

    AcquireReleaseTest(true /* within-GPU */, false /* acquirer is local */, true /* scalar */);

    TEST_END
}

TEST_F(KFDMemoryTest, AcquireReleaseCloseRemoteVector) {
    TEST_REQUIRE_ENV_CAPABILITIES(ENVCAPS_64BITLINUX);
    TEST_START(TESTPROFILE_RUNALL);

    AcquireReleaseTest(true /* within-GPU */, true /* releaser is local */, false /* vector */);

    TEST_END
}

TEST_F(KFDMemoryTest, AcquireReleaseCloseRemoteScalar) {
    TEST_REQUIRE_ENV_CAPABILITIES(ENVCAPS_64BITLINUX);
    TEST_START(TESTPROFILE_RUNALL);

    AcquireReleaseTest(true /* within-GPU */, true /* releaser is local */, true /* scalar */);

    TEST_END
}


/* Application register same userptr to multiple GPUs using multiple threads
 * Test multiple threads register/deregister same userptr, to verify Thunk race handling
 */
struct ThreadParams {
    void* pBuf;
    HSAuint64 BufferSize;
    HSAuint64 VAGPU;
    pthread_barrier_t *barrier;
};
static unsigned int RegisterThread(void* p) {
    struct ThreadParams* pArgs = reinterpret_cast<struct ThreadParams*>(p);

    pthread_barrier_wait(pArgs->barrier);
    EXPECT_SUCCESS(hsaKmtRegisterMemory(pArgs->pBuf, pArgs->BufferSize));
    EXPECT_SUCCESS(hsaKmtMapMemoryToGPU(pArgs->pBuf, pArgs->BufferSize, &pArgs->VAGPU));

    return 0;
}
static unsigned int UnregisterThread(void* p) {
    struct ThreadParams* pArgs = reinterpret_cast<struct ThreadParams*>(p);

    EXPECT_SUCCESS(hsaKmtUnmapMemoryToGPU(reinterpret_cast<void *>(pArgs->VAGPU)));
    pthread_barrier_wait(pArgs->barrier);
    EXPECT_SUCCESS(hsaKmtDeregisterMemory(reinterpret_cast<void *>(pArgs->VAGPU)));

    return 0;
}

#define N_THREADS   32

TEST_F(KFDMemoryTest, MultiThreadRegisterUserptrTest) {
    TEST_REQUIRE_ENV_CAPABILITIES(ENVCAPS_64BITLINUX);
    TEST_START(TESTPROFILE_RUNALL);

    int defaultGPUNode = m_NodeInfo.HsaDefaultGPUNode();
    ASSERT_GE(defaultGPUNode, 0) << "failed to get default GPU Node";

    HSAuint32 test_loops = 1;
    HSAuint64 BufferSize = 1UL << 27;

    void *pBuf = mmap(NULL, BufferSize, PROT_READ | PROT_WRITE,
                      MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
    ASSERT_NE(pBuf, MAP_FAILED);

    struct ThreadParams params[N_THREADS];
    HSAuint64 threadId[N_THREADS];

    pthread_barrier_t barrier;
    ASSERT_SUCCESS(pthread_barrier_init(&barrier, NULL, N_THREADS));

    for (HSAuint32 loop = 0; loop < test_loops; loop++) {
        for (HSAuint32 i = 0; i < N_THREADS; i++) {
            params[i].pBuf = pBuf;
            params[i].BufferSize = BufferSize;
            params[i].VAGPU = 0;
            params[i].barrier = &barrier;
        }

        for (HSAuint32 i = 0; i < N_THREADS; i++)
            ASSERT_EQ(true, StartThread(&RegisterThread, &params[i], threadId[i]));
        for (HSAuint32 i = 0; i < N_THREADS; i++)
            WaitForThread(threadId[i]);

        for (HSAuint32 i = 0; i < N_THREADS; i++)
            ASSERT_EQ(params[0].VAGPU, params[i].VAGPU);

        for (HSAuint32 i = 0; i < N_THREADS; i++)
            ASSERT_EQ(true, StartThread(&UnregisterThread, &params[i], threadId[i]));
        for (HSAuint32 i = 0; i < N_THREADS; i++)
            WaitForThread(threadId[i]);
    }

    pthread_barrier_destroy(&barrier);
    munmap(pBuf, BufferSize);

    TEST_END
}

static void ExportDMABufTest(KFDTEST_PARAMETERS* pTestParamters) {

    int gpuNode = pTestParamters->gpuNode;
    KFDMemoryTest* pKFDMemoryTest = (KFDMemoryTest*)pTestParamters->pTestObject;

    Assembler* m_pAsm;
    m_pAsm = pKFDMemoryTest->GetAssemblerFromNodeId(gpuNode);
    ASSERT_NOTNULL_GPU(m_pAsm, gpuNode);

    if (pKFDMemoryTest->Get_Version()->KernelInterfaceMinorVersion < 12) {
        LOG() << "Skipping test, requires KFD ioctl version 1.12 or newer" << std::endl;
        return;
    }

    // Use a GTT BO for export because it's conveniently CPU accessible.
    // On multi-GPU systems this also checks for interactions with driver-
    // internal DMA buf use for DMA attachment to multiple GPUs
    HsaMemFlags memFlags = pKFDMemoryTest->GetHsaMemFlags();
    memFlags.ui32.NonPaged = 1;

    HSAuint32 *buf;
    ASSERT_SUCCESS_GPU(hsaKmtAllocMemory(0, PAGE_SIZE, memFlags,
                                          reinterpret_cast<void**>(&buf)), gpuNode);
    ASSERT_SUCCESS_GPU(hsaKmtMapMemoryToGPU(buf, PAGE_SIZE, NULL), gpuNode);

    for (int i = 0; i < PAGE_SIZE/4; i++)
        buf[i] = i;
    const HSAuint64 INDEX = 25;
    const HSAuint64 SIZE = 25;
    HSAuint64 offset;
    int fd;

    // Expected error: address out of range (not a BO)
    ASSERT_EQ_GPU(HSAKMT_STATUS_INVALID_PARAMETER,
            hsaKmtExportDMABufHandle(buf + PAGE_SIZE/4, SIZE*4, &fd, &offset), gpuNode);
    // Expected error: size out of range
    ASSERT_EQ_GPU(HSAKMT_STATUS_INVALID_PARAMETER,
            hsaKmtExportDMABufHandle(buf + INDEX, PAGE_SIZE, &fd, &offset), gpuNode);

    // For real this time. Check that the offset matches
    ASSERT_SUCCESS_GPU(hsaKmtExportDMABufHandle(buf + INDEX, SIZE*4, &fd, &offset), gpuNode);
    ASSERT_EQ_GPU(INDEX*4, offset, gpuNode);

    // Free the original BO. The memory should persist as long as the DMA buf
    // handle exists.
    ASSERT_SUCCESS_GPU(hsaKmtUnmapMemoryToGPU(buf), gpuNode);
    ASSERT_SUCCESS_GPU(hsaKmtFreeMemory(buf, PAGE_SIZE), gpuNode);

    // Import the BO using the Interop API and check the contents. It doesn't
    // map the import for CPU access, which gives us an excuse to test GPU
    // mapping of the imported BO as well.
    HsaGraphicsResourceInfo info;
    ASSERT_SUCCESS_GPU(hsaKmtRegisterGraphicsHandleToNodes(fd, &info, 1, (HSAuint32 *)&gpuNode), gpuNode);
    buf = reinterpret_cast<HSAuint32 *>(info.MemoryAddress);
    ASSERT_EQ_GPU(info.SizeInBytes, PAGE_SIZE, gpuNode);

    HsaMemMapFlags mapFlags = {0};
    ASSERT_SUCCESS_GPU(hsaKmtMapMemoryToGPUNodes(buf, PAGE_SIZE, NULL, mapFlags, 1,
                                             (HSAuint32 *)&gpuNode), gpuNode);

    PM4Queue pm4Queue;
    ASSERT_SUCCESS_GPU(pm4Queue.Create(gpuNode), gpuNode);
    HsaMemoryBuffer dstBuffer(PAGE_SIZE, gpuNode);
    HsaMemoryBuffer isaBuffer(PAGE_SIZE, gpuNode, true/*zero*/, false/*local*/, true/*exec*/);
    ASSERT_SUCCESS_GPU(m_pAsm->RunAssembleBuf(CopyDwordIsa, isaBuffer.As<char*>()), gpuNode);
    for (int i = 0; i < PAGE_SIZE/4; i++) {
        Dispatch dispatch(isaBuffer);
        dispatch.SetArgs(&buf[i], dstBuffer.As<void*>());
        dispatch.Submit(pm4Queue);
        dispatch.Sync(g_TestTimeOut);
        ASSERT_EQ(i, *dstBuffer.As<HSAuint32 *>());
    }
    ASSERT_SUCCESS_GPU(pm4Queue.Destroy(), gpuNode);

    ASSERT_SUCCESS_GPU(hsaKmtUnmapMemoryToGPU(buf), gpuNode);
    ASSERT_SUCCESS_GPU(hsaKmtDeregisterMemory(buf), gpuNode);

    ASSERT_EQ_GPU(0, close(fd), gpuNode);
}

TEST_F(KFDMemoryTest, ExportDMABufTest) {
    TEST_REQUIRE_ENV_CAPABILITIES(ENVCAPS_64BITLINUX);
	TEST_START(TESTPROFILE_RUNALL);

    ASSERT_SUCCESS(KFDTest_Launch(ExportDMABufTest));

    TEST_END
}

static void VA_VRAM_Only_AllocTest(KFDTEST_PARAMETERS* pTestParamters) {

    int gpuNode = pTestParamters->gpuNode;
    KFDMemoryTest* pKFDMemoryTest = (KFDMemoryTest*)pTestParamters->pTestObject;

    Assembler* m_pAsm;
    m_pAsm = pKFDMemoryTest->GetAssemblerFromNodeId(gpuNode);
    ASSERT_NOTNULL_GPU(m_pAsm, gpuNode);

   if (pKFDMemoryTest->Get_Version()->KernelInterfaceMinorVersion < 12) {
        LOG() << "Skipping test, requires KFD ioctl version 1.12 or newer" << std::endl;
        return;
    }

    HsaMemFlags memFlags = pKFDMemoryTest->GetHsaMemFlags();
    memFlags.ui32.NonPaged = 1;
    memFlags.ui32.HostAccess = 0;

    HsaMemMapFlags mapFlags = {0};

    HSAuint32 *buf;

    /*alloc va without vram alloc*/
    memFlags.ui32.OnlyAddress = 1;
    ASSERT_SUCCESS(hsaKmtAllocMemory(gpuNode, PAGE_SIZE, memFlags,
                                          reinterpret_cast<void**>(&buf)));

    /*mapping VA allocated by kfd api would fail*/
    ASSERT_EQ(HSAKMT_STATUS_INVALID_PARAMETER, hsaKmtMapMemoryToGPU(buf, PAGE_SIZE, NULL));
    ASSERT_EQ(HSAKMT_STATUS_INVALID_PARAMETER, hsaKmtMapMemoryToGPUNodes(buf, PAGE_SIZE, NULL,
                               mapFlags, 1, reinterpret_cast<HSAuint32 *>(&gpuNode)));

    ASSERT_SUCCESS(hsaKmtFreeMemory(buf, PAGE_SIZE));

    /*alloc vram without va assigned*/
    memFlags.ui32.OnlyAddress = 0;
    memFlags.ui32.NoAddress = 1;
    ASSERT_SUCCESS(hsaKmtAllocMemory(gpuNode, PAGE_SIZE, memFlags,
                                      reinterpret_cast<void**>(&buf)));

    /*mapping handle allocated by kfd API would fail*/
    ASSERT_EQ(HSAKMT_STATUS_INVALID_PARAMETER, hsaKmtMapMemoryToGPU(buf, PAGE_SIZE, NULL));
    ASSERT_EQ(HSAKMT_STATUS_INVALID_PARAMETER, hsaKmtMapMemoryToGPUNodes(buf, PAGE_SIZE, NULL,
                               mapFlags, 1, reinterpret_cast<HSAuint32 *>(&gpuNode)));

    ASSERT_SUCCESS(hsaKmtFreeMemory(buf, PAGE_SIZE));
}

TEST_F(KFDMemoryTest, VA_VRAM_Only_AllocTest) {
    TEST_REQUIRE_ENV_CAPABILITIES(ENVCAPS_64BITLINUX);
	TEST_START(TESTPROFILE_RUNALL);

    ASSERT_SUCCESS(KFDTest_Launch(VA_VRAM_Only_AllocTest));

    TEST_END
}
