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

#include "KFDLocalMemoryTest.hpp"
#include "PM4Queue.hpp"
#include "PM4Packet.hpp"
#include "SDMAPacket.hpp"
#include "SDMAQueue.hpp"
#include "Dispatch.hpp"

void KFDLocalMemoryTest::SetUp() {
    ROUTINE_START

    KFDBaseComponentTest::SetUp();

    ROUTINE_END
}

void KFDLocalMemoryTest::TearDown() {
    ROUTINE_START

    KFDBaseComponentTest::TearDown();

    ROUTINE_END
}

static void AccessLocalMem(KFDTEST_PARAMETERS* pTestParamters) {

    /* Skip test if not on dGPU path, which the test depends on */
    if (!hsakmt_is_dgpu()) {
        LOG() << "Not dGPU path, skipping the test" << std::endl;
        return;
    }

    int gpuNode = pTestParamters->gpuNode;

    //local memory
    HsaMemoryBuffer destBuf(PAGE_SIZE, gpuNode, false, true);
    HsaEvent *event;
    ASSERT_SUCCESS_GPU(CreateQueueTypeEvent(false, false, gpuNode, &event), gpuNode);

    PM4Queue queue;

    ASSERT_SUCCESS_GPU(queue.Create(gpuNode), gpuNode);

    queue.PlaceAndSubmitPacket(PM4WriteDataPacket(destBuf.As<unsigned int*>(), 0, 0));

    queue.Wait4PacketConsumption(event);

    hsaKmtDestroyEvent(event);
    EXPECT_SUCCESS_GPU(queue.Destroy(), gpuNode);

}

TEST_F(KFDLocalMemoryTest, AccessLocalMem) {
    TEST_START(TESTPROFILE_RUNALL)

    ASSERT_SUCCESS(KFDTest_Launch(AccessLocalMem));

    TEST_END
}

static void BasicTest(KFDTEST_PARAMETERS* pTestParamters) {

    PM4Queue queue;
    HSAuint64 AlternateVAGPU;
    unsigned int BufferSize = PAGE_SIZE;
    HsaMemMapFlags mapFlags = {0};

    int gpuNode = pTestParamters->gpuNode;
    KFDLocalMemoryTest* pKFDLocalMemoryTest = (KFDLocalMemoryTest*)pTestParamters->pTestObject;

    Assembler* m_pAsm;
    m_pAsm = pKFDLocalMemoryTest->GetAssemblerFromNodeId(gpuNode);
    ASSERT_NOTNULL_GPU(m_pAsm, gpuNode);

    HsaMemoryBuffer isaBuffer(PAGE_SIZE, gpuNode, true/*zero*/, false/*local*/, true/*exec*/);
    HsaMemoryBuffer srcSysBuffer(BufferSize, gpuNode, false);
    HsaMemoryBuffer destSysBuffer(BufferSize, gpuNode);
    HsaMemoryBuffer srcLocalBuffer(BufferSize, gpuNode, false, true);
    HsaMemoryBuffer dstLocalBuffer(BufferSize, gpuNode, false, true);

    srcSysBuffer.Fill(0x01010101);

    ASSERT_SUCCESS_GPU(m_pAsm->RunAssembleBuf(CopyDwordIsa, isaBuffer.As<char*>()), gpuNode);

    ASSERT_SUCCESS_GPU(hsaKmtMapMemoryToGPUNodes(srcLocalBuffer.As<void*>(), srcLocalBuffer.Size(), &AlternateVAGPU,
                       mapFlags, 1, reinterpret_cast<HSAuint32 *>(&gpuNode)), gpuNode);
    ASSERT_SUCCESS_GPU(hsaKmtMapMemoryToGPUNodes(dstLocalBuffer.As<void*>(), dstLocalBuffer.Size(), &AlternateVAGPU,
                       mapFlags, 1, reinterpret_cast<HSAuint32 *>(&gpuNode)), gpuNode);

    ASSERT_SUCCESS_GPU(queue.Create(gpuNode), gpuNode);
    queue.SetSkipWaitConsump(0);

    Dispatch dispatch(isaBuffer);

    dispatch.SetArgs(srcSysBuffer.As<void*>(), srcLocalBuffer.As<void*>());
    dispatch.Submit(queue);
    dispatch.Sync(g_TestTimeOut);

    dispatch.SetArgs(srcLocalBuffer.As<void*>(), dstLocalBuffer.As<void*>());
    dispatch.Submit(queue);
    dispatch.Sync(g_TestTimeOut);

    dispatch.SetArgs(dstLocalBuffer.As<void*>(), destSysBuffer.As<void*>());
    dispatch.Submit(queue);
    dispatch.Sync(g_TestTimeOut);

    EXPECT_SUCCESS_GPU(queue.Destroy(), gpuNode);

    ASSERT_SUCCESS_GPU(hsaKmtUnmapMemoryToGPU(srcLocalBuffer.As<void*>()), gpuNode);
    ASSERT_SUCCESS_GPU(hsaKmtUnmapMemoryToGPU(dstLocalBuffer.As<void*>()), gpuNode);
    EXPECT_EQ_GPU(destSysBuffer.As<unsigned int*>()[0], 0x01010101, gpuNode);

}

TEST_F(KFDLocalMemoryTest, BasicTest) {
    TEST_REQUIRE_ENV_CAPABILITIES(ENVCAPS_64BITLINUX);
    TEST_START(TESTPROFILE_RUNALL);

    ASSERT_SUCCESS(KFDTest_Launch(BasicTest));

    TEST_END
}

static void VerifyContentsAfterUnmapAndMap(KFDTEST_PARAMETERS* pTestParamters)
{
    PM4Queue queue;
    HSAuint64 AlternateVAGPU;
    unsigned int BufferSize = PAGE_SIZE;
    HsaMemMapFlags mapFlags = {0};

    int gpuNode = pTestParamters->gpuNode;
    KFDLocalMemoryTest* pKFDLocalMemoryTest = (KFDLocalMemoryTest*)pTestParamters->pTestObject;

    Assembler* m_pAsm;
    m_pAsm = pKFDLocalMemoryTest->GetAssemblerFromNodeId(gpuNode);
    ASSERT_NOTNULL_GPU(m_pAsm, gpuNode);

    HsaMemoryBuffer isaBuffer(PAGE_SIZE, gpuNode, true/*zero*/, false/*local*/, true/*exec*/);
    HsaMemoryBuffer SysBufferA(BufferSize, gpuNode, false);
    HsaMemoryBuffer SysBufferB(BufferSize, gpuNode, true);
    HsaMemoryBuffer LocalBuffer(BufferSize, gpuNode, false, true);

    SysBufferA.Fill(0x01010101);

    ASSERT_SUCCESS_GPU(m_pAsm->RunAssembleBuf(CopyDwordIsa, isaBuffer.As<char*>()), gpuNode);

    ASSERT_SUCCESS_GPU(queue.Create(gpuNode), gpuNode);
    queue.SetSkipWaitConsump(0);

    if (!hsakmt_is_dgpu())
        ASSERT_SUCCESS_GPU(hsaKmtMapMemoryToGPUNodes(LocalBuffer.As<void*>(), LocalBuffer.Size(), &AlternateVAGPU,
                           mapFlags, 1, reinterpret_cast<HSAuint32 *>(&gpuNode)), gpuNode);

    Dispatch dispatch(isaBuffer);

    dispatch.SetArgs(SysBufferA.As<void*>(), LocalBuffer.As<void*>());
    dispatch.Submit(queue);
    dispatch.Sync(g_TestTimeOut);

    EXPECT_SUCCESS_GPU(hsaKmtUnmapMemoryToGPU(LocalBuffer.As<void*>()), gpuNode);
    EXPECT_SUCCESS_GPU(hsaKmtMapMemoryToGPUNodes(LocalBuffer.As<void*>(), LocalBuffer.Size(), &AlternateVAGPU,
                       mapFlags, 1, reinterpret_cast<HSAuint32 *>(&gpuNode)), gpuNode);

    dispatch.SetArgs(LocalBuffer.As<void*>(), SysBufferB.As<void*>());
    dispatch.Submit(queue);
    dispatch.Sync(g_TestTimeOut);

    EXPECT_SUCCESS_GPU(queue.Destroy(), gpuNode);
    EXPECT_EQ_GPU(SysBufferB.As<unsigned int*>()[0], 0x01010101, gpuNode);
    if (!hsakmt_is_dgpu())
        EXPECT_SUCCESS_GPU(hsaKmtUnmapMemoryToGPU(LocalBuffer.As<void*>()), gpuNode);
}

TEST_F(KFDLocalMemoryTest, VerifyContentsAfterUnmapAndMap) {
    TEST_REQUIRE_ENV_CAPABILITIES(ENVCAPS_64BITLINUX);
    TEST_START(TESTPROFILE_RUNALL);

    ASSERT_SUCCESS(KFDTest_Launch(VerifyContentsAfterUnmapAndMap));

    TEST_END
}

/* Deliberately fragment GPUVM aperture to fill up address space
 *
 * General idea: Allocate buffers, but don't map them to GPU. This
 * will reserve virtual address space without pinning physical
 * memory. It should allow using more address space than physically
 * available memory.
 *
 * Even without pinning memory, TTM will still commit memory at
 * allocation time and swap out movable buffers to system memory or
 * even the hard drive, if it needs to. So we can't allocate arbitrary
 * amounts of virtual memory.
 *
 * Strategy to maximize the amount of allocated, fragmented address
 * space while keeping the amount of committed memory bounded at all
 * times:
 *
 * 1. Allocate N blocks of a given size, initially 1 page
 * 2. Free every other block, creating holes in the address space.
 *    This frees up half the memory
 * 3. Allocate N/4 blocks of 2-pages each. This requires as much
 *    memory as was freed in step 2. The block size is bigger than
 *    the 1-page holes, so new address space will be used.
 * 4. Free half the blocks just allocated, and half of the
 *    remaining blocks of step 1. This creates 3-page holes between
 *    the 1-page blocks from step 1, and 2-page holes between the
 *    2-page blocks from step 3. It frees up half of the total
 *    memory.
 * 5. Double the block size to 4, devide number of blocks by 2.
 *    Again, this will require the amount of memory freed in step 4.
 *    The block size 4 is bigger than the biggest hole (3 pages).
 * 6. Free half the memory again, creating 7-page holes between
 *    1-page blocks, 6-page holes between 2-page blocks, and 4-page
 *    holes between 4-page blocks.
 *
 * Repeat, doubling block size and halving number of blocks in each
 * iteration. Each iteration starts and ends with half the total
 * memory free. Because the block size is always bigger than the
 * biggest hole, each iteration increases the amount of address space
 * occupied by half the total memory size. Once the block size reaches
 * half of the free memory (1/4 of total memory) the limit is reached.
 *
 * With 2^n pages available memory, n * 2^(n-1) pages of address space
 * can be reserved. At the end of that process, half the memory will
 * be free.
 *
 *     Total memory     | Fragmented address space
 * order | pages | size | pages |  size | ratio
 * ------+-------+------+-------+-------+-------
 *     2 |    4  |  16K |    4  |   16K |   1
 *     3 |    8  |  32K |   12  |   48K |   1.5
 *     4 |   16  |  64K |   32  |  128K |   2
 *     5 |   32  | 128K |   80  |  320K |   2.5
 *     6 |   64  | 256K |  192  |  768K |   3
 *     7 |  128  | 512K |  448  | 1.75M |   3.5
 *     8 |  256  |   1M |    1M |    4M |   4
 *     9 |  512  |   2M | 2.25M |    9M |   4.5
 *    10 |    1K |   4M |    5M |   20M |   5
 *    11 |    2K |   8M |   11M |   44M |   5.5
 *    12 |    4K |  16M |   24M |   96M |   6
 *    13 |    8K |  32M |   52M |  208M |   6.5
 *    14 |   16K |  64M |  112M |  448M |   7
 *    15 |   32K | 128M |  240M |  960M |   7.5
 *    16 |   64K | 256M |  512M |    2G |   8
 *    17 |  128K | 512M | 1088M | 4.25G |   8.5
 *    18 |  256K |   1G | 2.25G |    9G |   9
 *    19 |  512K |   2G | 4.75G |   19G |   9.5
 *    20 |    1M |   4G |   10G |   40G |  10
 */

static void Fragmentation(KFDTEST_PARAMETERS* pTestParamters){

    int gpuNode = pTestParamters->gpuNode;
    KFDLocalMemoryTest* pKFDLocalMemoryTest = (KFDLocalMemoryTest*)pTestParamters->pTestObject;

    HSAuint64 fbSize;

    fbSize = pKFDLocalMemoryTest->GetVramSize(gpuNode);

    if (!fbSize) {
        LOG() << "Skipping test: No VRAM found." << std::endl;
        return;
    } else {
        LOG() << "Found VRAM of " << std::dec << (fbSize >> 20) << "MB." << std::endl;
    }

    /* Use up to half of available memory. Using more results in
     * excessive memory movement in TTM and slows down the test too
     * much. maxOrder is the size of the biggest block that will be
     * allocated. It's 1/4 of the usable memory, so 1/8 the total FB
     * size in pages.
     *
     * Use 8x bigger page size on dGPU to match Tonga alignment
     * workaround. Also nicely matches the 8x bigger GPUVM address
     * space on AMDGPU compared to RADEON.
     */
    unsigned pageSize = hsakmt_is_dgpu() ? PAGE_SIZE*8 : PAGE_SIZE;
    fbSize /= pageSize;
    unsigned maxOrder = 0;
    // Limit maxOrder up to 14 so this test doesn't run longer than 10 mins
    while (((fbSize >> maxOrder) >= 16) && (maxOrder < 14))
        maxOrder++;

    /* Queue and memory used by the shader copy tests */
    HsaMemoryBuffer sysBuffer(PAGE_SIZE, gpuNode, false);
    PM4Queue queue;
    ASSERT_SUCCESS_GPU(queue.Create(gpuNode), gpuNode);
    HsaMemoryBuffer isaBuffer(PAGE_SIZE, gpuNode, true/*zero*/, false/*local*/, true/*exec*/);

    /* instantiate Assembler for gpuNode */
    HsaNodeInfo* m_NodeInfo = pKFDLocalMemoryTest->Get_NodeInfo();
    const HsaNodeProperties *nodeProperties = m_NodeInfo->GetNodeProperties(gpuNode);
    Assembler* m_pAsm = new Assembler(GetGfxVersion(nodeProperties));

    ASSERT_SUCCESS_GPU(m_pAsm->RunAssembleBuf(CopyDwordIsa, isaBuffer.As<char*>()), gpuNode);
   /* not need assember now */
    if (m_pAsm)
        delete m_pAsm;

    /* Allocate and test memory using the strategy explained at the top */
    HSAKMT_STATUS status;
    HsaMemFlags memFlags = {0};
    HsaMemMapFlags mapFlags = {0};
    memFlags.ui32.PageSize = HSA_PAGE_SIZE_4KB;
    memFlags.ui32.HostAccess = 0;
    memFlags.ui32.NonPaged = 1;
    struct {
        void **pointers;
        unsigned long nPages;
    } pages[maxOrder+1];
    unsigned order, o;
    unsigned long p;
    HSAuint64 size;
    unsigned value = 0;
    memset(pages, 0, sizeof(pages));
    for (order = 0; order <= maxOrder; order++) {
        // At maxOrder, block size is 1/4 of available memory
        pages[order].nPages = 1UL << (maxOrder - order + 2);
        // At order != 0, 1/2 the memory is already allocated
        if (order > 0)
            pages[order].nPages >>= 1;
        // Allocate page pointers
        pages[order].pointers = new void *[pages[order].nPages];
        EXPECT_NE_GPU((void **)NULL, pages[order].pointers, gpuNode)
            << "Couldn't allocate memory for " << pages[order].nPages
            << " pointers at order " << order << std::endl;
        if (!pages[order].pointers) {
            pages[order].nPages = 0;
            break;
        }
        /* Allocate buffers and access the start and end of every one:
         * 1. Copy from sysBuffer[0] to start of block
         * 2. Copy from start of block to end of block
         * 3. Copy from end of block to sysBuffer[1]
         * 4. Compare results */
        size = (HSAuint64)(1 << order) * pageSize;
        LOG() << std::dec << "Trying to allocate " << pages[order].nPages
              << " order " << order << " blocks " << std::endl;
        for (p = 0; p < pages[order].nPages; p++) {
            status = hsaKmtAllocMemory(gpuNode, size,
                                       memFlags, &pages[order].pointers[p]);
            if (status != HSAKMT_STATUS_SUCCESS) {
                EXPECT_EQ_GPU(HSAKMT_STATUS_NO_MEMORY, status, gpuNode);
                pages[order].nPages = p;
                break;
            }

            void *bufferEnd = reinterpret_cast<void *>(reinterpret_cast<unsigned long>(pages[order].pointers[p])
                                       + size - sizeof(unsigned));
            sysBuffer.As<unsigned *>()[0] = ++value;

            status = hsaKmtMapMemoryToGPUNodes(pages[order].pointers[p], size, NULL,
                               mapFlags, 1, reinterpret_cast<HSAuint32 *>(&gpuNode));
            if (status != HSAKMT_STATUS_SUCCESS) {
                ASSERT_SUCCESS_GPU(hsaKmtFreeMemory(pages[order].pointers[p],
                                                size), gpuNode);
                pages[order].nPages = p;
                break;
            }
            Dispatch dispatch1(isaBuffer);
            dispatch1.SetArgs(sysBuffer.As<void*>(), pages[order].pointers[p]);
            dispatch1.Submit(queue);
            // no sync needed for multiple GPU dispatches to the same queue

            Dispatch dispatch2(isaBuffer);
            dispatch2.SetArgs(pages[order].pointers[p], bufferEnd);
            dispatch2.Submit(queue);
            // no sync needed for multiple GPU dispatches to the same queue

            Dispatch dispatch3(isaBuffer);
            dispatch3.SetArgs(bufferEnd,
                              reinterpret_cast<void *>(&(sysBuffer.As<unsigned*>()[1])));
            dispatch3.Submit(queue);
            dispatch3.Sync(g_TestTimeOut);
            EXPECT_EQ_GPU(value, sysBuffer.As<unsigned *>()[1], gpuNode);

            EXPECT_SUCCESS_GPU(hsaKmtUnmapMemoryToGPU(pages[order].pointers[p]), gpuNode);
        }
        LOG() << "  Got " << pages[order].nPages
              << ", end of last block addr: "
              << reinterpret_cast<void *>(reinterpret_cast<unsigned long>(pages[order].pointers[p-1]) + size - 1)
              << std::endl;

        // Now free half the memory
        for (o = 0; o <= order; o++) {
            unsigned long step = 1UL << (order - o + 1);
            unsigned long offset = (step >> 1) - 1;
            size = (HSAuint64)(1 << o) * pageSize;
            LOG() << "  Freeing every " << step << "th order "
                  << o << " block starting with " << offset << std::endl;
            for (p = offset; p < pages[o].nPages; p += step) {
                ASSERT_NE_GPU((void **)NULL, pages[o].pointers[p], gpuNode);
                EXPECT_SUCCESS_GPU(hsaKmtFreeMemory(pages[o].pointers[p], size), gpuNode);
                pages[o].pointers[p] = NULL;
            }
        }
    }

    /* Clean up */
    for (order = 0; order <= maxOrder; order++) {
        if (pages[order].pointers == NULL)
            continue;

        size = (HSAuint64)(1 << order) * pageSize;
        for (p = 0; p < pages[order].nPages; p++)
            if (pages[order].pointers[p] != NULL)
                EXPECT_SUCCESS_GPU(hsaKmtFreeMemory(pages[order].pointers[p], size), gpuNode);

        delete[] pages[order].pointers;
    }

    EXPECT_SUCCESS_GPU(queue.Destroy(), gpuNode);
}

TEST_F(KFDLocalMemoryTest, DISABLED_Fragmentation) {
    TEST_REQUIRE_ENV_CAPABILITIES(ENVCAPS_64BITLINUX);
    TEST_START(TESTPROFILE_RUNALL);

    ASSERT_SUCCESS(KFDTest_Launch(Fragmentation));

    TEST_END
}

static void CheckZeroInitializationVram(KFDTEST_PARAMETERS* pTestParamters){

    int gpuNode = pTestParamters->gpuNode;
    KFDLocalMemoryTest* pKFDLocalMemoryTest = (KFDLocalMemoryTest*)pTestParamters->pTestObject;

    /* Testing VRAM */
    HSAuint64 vramSizeMB = pKFDLocalMemoryTest->GetVramSize(gpuNode) >> 20;

   if (!vramSizeMB) {
        LOG() << "Skipping test: No VRAM found." << std::endl;
        return;
   }

    HSAuint64 vramBufSizeMB = vramSizeMB >> 2;
    /* limit the buffer size in order not to overflow the SDMA queue buffer. */
    if (vramBufSizeMB > 1024) {
        vramBufSizeMB = 1024;
    }
    HSAuint64 vramBufSize = vramBufSizeMB * 1024 * 1024;

    /* Make sure the entire VRAM is used at least once */
    int count = (vramSizeMB + vramBufSizeMB - 1) / vramBufSizeMB + 1;

    LOG() << "Using " << std::dec << vramBufSizeMB
            << "MB VRAM buffer to test " << std::dec << count
            << " times"<< std::endl;

    SDMAQueue sdmaQueue;
    ASSERT_SUCCESS_GPU(sdmaQueue.Create(gpuNode, 8 * PAGE_SIZE), gpuNode);

    HsaMemoryBuffer tmpBuffer(PAGE_SIZE, 0, true /* zero */);
    volatile HSAuint32 *tmp = tmpBuffer.As<volatile HSAuint32 *>();

    unsigned int offset = 2060;  // a constant offset, should be 4 aligned.

    while (count--) {
        HsaMemoryBuffer localBuffer(vramBufSize, gpuNode, false, true);

        EXPECT_TRUE_GPU(localBuffer.IsPattern(0, 0, sdmaQueue, tmp), gpuNode);

        for (HSAuint64 i = offset; i < vramBufSize;) {
            EXPECT_TRUE_GPU(localBuffer.IsPattern(i, 0, sdmaQueue, tmp), gpuNode);
            i += 4096;
        }

        /* Checking last 4 bytes */
        EXPECT_TRUE_GPU(localBuffer.IsPattern(vramBufSize - 4, 0, sdmaQueue, tmp), gpuNode);

        localBuffer.Fill(0xABCDEFFF, sdmaQueue);
    }

}

TEST_F(KFDLocalMemoryTest, CheckZeroInitializationVram) {
    TEST_REQUIRE_ENV_CAPABILITIES(ENVCAPS_64BITLINUX);
    TEST_START(TESTPROFILE_RUNALL);

    ASSERT_SUCCESS(KFDTest_Launch(CheckZeroInitializationVram));

    TEST_END
}

TEST_F(KFDLocalMemoryTest, MapVramToGPUNodesTest) {
    TEST_START(TESTPROFILE_RUNALL);

    HSAint32 src_node;
    HSAint32 dst_node;
    HsaPointerInfo info;

    const std::vector<int> gpuNodes = m_NodeInfo.GetNodesWithGPU();
    if (gpuNodes.size() < 2) {
        LOG() << "Skipping test: Test requires at least two GPUs." << std::endl;
        return;
    }

    if (g_TestDstNodeId != -1 && g_TestNodeId != -1) {
        src_node = g_TestNodeId;
        dst_node = g_TestDstNodeId;
    } else {
        int defaultGPUNode = m_NodeInfo.HsaDefaultGPUNode();

        dst_node = m_NodeInfo.FindLargeBarGPUNode();
        if (dst_node < 0) {
            LOG() << "Skipping test: Test requires at least one large bar GPU." << std::endl;
            return;
        }

        if (dst_node != defaultGPUNode) {
            /* At least one node should be defaultGPUNode */
            src_node = defaultGPUNode;
        } else {
            for (auto node : gpuNodes) {
                if (node != dst_node) {
                    src_node = node;
                    break;
                }
            }
        }
    }

    if (!m_NodeInfo.IsPeerAccessibleByNode(dst_node, src_node)) {
        LOG() << "Skipping test: GPUs are not peer-accessible" << std::endl;
        return;
    }

    LOG() << "Testing from GPU " << src_node << " to GPU " << dst_node << std::endl;

    void *shared_addr;
    HSAuint32 nodes[] = { (HSAuint32)src_node, (HSAuint32)dst_node };
    HsaMemFlags memFlags = {0};
    memFlags.ui32.PageSize = HSA_PAGE_SIZE_4KB;
    memFlags.ui32.HostAccess = 1;
    memFlags.ui32.NonPaged = 1;
    memFlags.ui32.ExecuteAccess = 1;

    HsaMemMapFlags mapFlags = {0};

    EXPECT_SUCCESS(hsaKmtAllocMemory(nodes[1], PAGE_SIZE, memFlags, &shared_addr));
    EXPECT_SUCCESS(hsaKmtMapMemoryToGPUNodes(shared_addr, PAGE_SIZE, NULL, mapFlags, 2, nodes));
    EXPECT_SUCCESS(hsaKmtQueryPointerInfo(shared_addr, &info));
    EXPECT_EQ(info.NMappedNodes, 2);

    EXPECT_SUCCESS(hsaKmtMapMemoryToGPUNodes(shared_addr, PAGE_SIZE, NULL, mapFlags, 1, &nodes[0]));
    EXPECT_SUCCESS(hsaKmtQueryPointerInfo(shared_addr, &info));
    EXPECT_EQ(info.NMappedNodes, 1);
    EXPECT_EQ(info.MappedNodes[0], nodes[0]);

    EXPECT_SUCCESS(hsaKmtMapMemoryToGPUNodes(shared_addr, PAGE_SIZE, NULL, mapFlags, 1, &nodes[1]));
    EXPECT_SUCCESS(hsaKmtQueryPointerInfo(shared_addr, &info));
    EXPECT_EQ(info.NMappedNodes, 1);
    EXPECT_EQ(info.MappedNodes[0], nodes[1]);

    EXPECT_SUCCESS(hsaKmtUnmapMemoryToGPU(shared_addr));
    EXPECT_SUCCESS(hsaKmtQueryPointerInfo(shared_addr, &info));
    EXPECT_EQ(info.NMappedNodes, 0);

    EXPECT_SUCCESS(hsaKmtMapMemoryToGPUNodes(shared_addr, PAGE_SIZE, NULL, mapFlags, 1, &nodes[0]));
    EXPECT_SUCCESS(hsaKmtQueryPointerInfo(shared_addr, &info));
    EXPECT_EQ(info.NMappedNodes, 1);
    EXPECT_EQ(info.MappedNodes[0], nodes[0]);

    EXPECT_SUCCESS(hsaKmtUnmapMemoryToGPU(shared_addr));
    EXPECT_SUCCESS(hsaKmtFreeMemory(shared_addr, PAGE_SIZE));

    TEST_END
}
