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
#include "KFDSVMRangeTest.hpp"
#include <poll.h>
#include <sys/mman.h>
#include <vector>
#include "PM4Queue.hpp"
#include "PM4Packet.hpp"
#include "SDMAPacket.hpp"
#include "SDMAQueue.hpp"
#include "Dispatch.hpp"

extern unsigned int g_TestGPUsNum;

void KFDSVMRangeTest::SetUp() {
    ROUTINE_START

    KFDBaseComponentTest::SetUp();

    SVMSetXNACKMode(GetParam());

    ROUTINE_END
}

void KFDSVMRangeTest::TearDown() {
    ROUTINE_START

    SVMRestoreXNACKMode();

    KFDBaseComponentTest::TearDown();

    ROUTINE_END
}

static void BasicSystemMemTest(KFDTEST_PARAMETERS* pTestParamters) {

    int gpuNode = pTestParamters->gpuNode;
    KFDSVMRangeTest* pKFDSVMRangeTest = (KFDSVMRangeTest*)pTestParamters->pTestObject;

    if (!pKFDSVMRangeTest->SVMAPISupported_GPU(gpuNode))
        return;

    PM4Queue queue;
    HSAuint64 AlternateVAGPU;
    unsigned int BufferSize = PAGE_SIZE;

    if (!pKFDSVMRangeTest->GetVramSize(gpuNode)) {
        LOG() << "Skipping test: No VRAM found." << std::endl;
        return;
    }

    HsaMemoryBuffer isaBuffer(PAGE_SIZE, gpuNode, true/*zero*/, false/*local*/, true/*exec*/);
    HsaSVMRange srcSysBuffer(BufferSize, gpuNode);
    HsaSVMRange destSysBuffer(BufferSize,gpuNode);

    Assembler* m_pAsm;
    m_pAsm = pKFDSVMRangeTest->GetAssemblerFromNodeId(gpuNode);
    ASSERT_NOTNULL_GPU(m_pAsm, gpuNode);

    srcSysBuffer.Fill(0x01010101);

    ASSERT_SUCCESS_GPU(m_pAsm->RunAssembleBuf(CopyDwordIsa, isaBuffer.As<char*>()), gpuNode);

    ASSERT_SUCCESS_GPU(queue.Create(gpuNode), gpuNode);
    queue.SetSkipWaitConsump(0);

    Dispatch dispatch(isaBuffer);

    dispatch.SetArgs(srcSysBuffer.As<void*>(), destSysBuffer.As<void*>());
    dispatch.Submit(queue);
    dispatch.Sync(g_TestTimeOut);

    EXPECT_SUCCESS_GPU(queue.Destroy(), gpuNode);

    EXPECT_EQ_GPU(destSysBuffer.As<unsigned int*>()[0], 0x01010101, gpuNode);
}

TEST_P(KFDSVMRangeTest, BasicSystemMemTest) {
    TEST_REQUIRE_ENV_CAPABILITIES(ENVCAPS_64BITLINUX);
    TEST_START(TESTPROFILE_RUNALL);

    ASSERT_SUCCESS(KFDTest_Launch(BasicSystemMemTest));

    TEST_END
}

static void SetGetAttributesTest(KFDTEST_PARAMETERS* pTestParamters) {

    int gpuNode = pTestParamters->gpuNode;
    KFDSVMRangeTest* pKFDSVMRangeTest = (KFDSVMRangeTest*)pTestParamters->pTestObject;

    if (!pKFDSVMRangeTest->SVMAPISupported_GPU(gpuNode))
        return;

    unsigned int m_FamilyId = pKFDSVMRangeTest->GetFamilyIdFromNodeId(gpuNode);
    if (m_FamilyId < FAMILY_AI) {
        LOG() << std::hex << "Skipping test: No svm range support for family ID 0x" << m_FamilyId << "." << std::endl;
        return;
    }

    int i;
    unsigned int BufSize = PAGE_SIZE;
    HsaSVMRange *sysBuffer = new HsaSVMRange(BufSize);
    HSAuint32 nAttributes = 5;
    HSA_SVM_ATTRIBUTE outputAttributes[nAttributes];
    HSA_SVM_ATTRIBUTE inputAttributes[] = {
                                                {HSA_SVM_ATTR_PREFETCH_LOC, (HSAuint32)gpuNode},
                                                {HSA_SVM_ATTR_PREFERRED_LOC, (HSAuint32)gpuNode},
                                                {HSA_SVM_ATTR_SET_FLAGS,
                                                 HSA_SVM_FLAG_HOST_ACCESS | HSA_SVM_FLAG_GPU_EXEC | HSA_SVM_FLAG_COHERENT},
                                                {HSA_SVM_ATTR_GRANULARITY, 0x3F},
                                                {HSA_SVM_ATTR_ACCESS, (HSAuint32)gpuNode},
                                          };

    HSAuint32 expectedDefaultResults[] = {
                                             INVALID_NODEID,
                                             INVALID_NODEID,
                                             HSA_SVM_FLAG_HOST_ACCESS | HSA_SVM_FLAG_COHERENT,
                                             9,
                                             0,
                                         };
    HSAint32 enable = -1;
    EXPECT_SUCCESS_GPU(hsaKmtGetXNACKMode(&enable), gpuNode);
    expectedDefaultResults[4] = (enable) ?
                                 HSA_SVM_ATTR_ACCESS : HSA_SVM_ATTR_NO_ACCESS;
    char *pBuf = sysBuffer->As<char *>();

    LOG() << "Get default atrributes" << std::endl;
    memcpy(outputAttributes, inputAttributes, nAttributes * sizeof(HSA_SVM_ATTRIBUTE));
    EXPECT_SUCCESS_GPU(hsaKmtSVMGetAttr(pBuf, BufSize,
                                    nAttributes, outputAttributes), gpuNode);

    for (i = 0; i < nAttributes; i++) {
        /* Default granularity could be specified using module parameter,
         * therefore it is incorrect to expect a particular value
         */
        if (outputAttributes[i].type == HSA_SVM_ATTR_GRANULARITY)
            continue;

        if (outputAttributes[i].type == HSA_SVM_ATTR_ACCESS ||
            outputAttributes[i].type == HSA_SVM_ATTR_ACCESS_IN_PLACE ||
            outputAttributes[i].type == HSA_SVM_ATTR_NO_ACCESS)
            EXPECT_EQ_GPU(outputAttributes[i].type, expectedDefaultResults[i], gpuNode);
        else
            EXPECT_EQ_GPU(outputAttributes[i].value, expectedDefaultResults[i], gpuNode);
    }
    LOG() << "Setting/Getting atrributes" << std::endl;
    memcpy(outputAttributes, inputAttributes, nAttributes * sizeof(HSA_SVM_ATTRIBUTE));
    EXPECT_SUCCESS_GPU(hsaKmtSVMSetAttr(pBuf, BufSize,
                                    nAttributes, inputAttributes), gpuNode);
    EXPECT_SUCCESS_GPU(hsaKmtSVMGetAttr(pBuf, BufSize,
                                    nAttributes, outputAttributes), gpuNode);
    for (i = 0; i < nAttributes; i++) {
        if (outputAttributes[i].type == HSA_SVM_ATTR_ACCESS ||
            outputAttributes[i].type == HSA_SVM_ATTR_ACCESS_IN_PLACE ||
            outputAttributes[i].type == HSA_SVM_ATTR_NO_ACCESS)
            EXPECT_EQ_GPU(inputAttributes[i].type, outputAttributes[i].type, gpuNode);
        else
            EXPECT_EQ_GPU(inputAttributes[i].value, outputAttributes[i].value, gpuNode);
    }
    delete sysBuffer;

}

TEST_P(KFDSVMRangeTest, SetGetAttributesTest) {
    TEST_REQUIRE_ENV_CAPABILITIES(ENVCAPS_64BITLINUX);
    TEST_START(TESTPROFILE_RUNALL)

    ASSERT_SUCCESS(KFDTest_Launch(SetGetAttributesTest));

    TEST_END
}

TEST_P(KFDSVMRangeTest, XNACKModeTest) {
    TEST_REQUIRE_ENV_CAPABILITIES(ENVCAPS_64BITLINUX);
    TEST_START(TESTPROFILE_RUNALL);

    if (!SVMAPISupported())
        return;

    HSAuint32 i, j;
    HSAint32 r;
    PM4Queue queue;
    HSAint32 enable = 0;
    const std::vector<int> gpuNodes = m_NodeInfo.GetNodesWithGPU();

    EXPECT_SUCCESS(hsaKmtGetXNACKMode(&enable));
    for (i = 0; i < 2; i++) {
        enable = !enable;
        r = hsaKmtSetXNACKMode(enable);
        if (r == HSAKMT_STATUS_SUCCESS) {
            LOG() << "XNACK mode: " << std::boolalpha << enable <<
                     " supported" << std::endl;

            for (j = 0; j < gpuNodes.size(); j++) {
                LOG() << "Creating queue and try to set xnack mode on node: "
                      << gpuNodes.at(j) << std::endl;
                ASSERT_SUCCESS(queue.Create(gpuNodes.at(j)));
                EXPECT_EQ(HSAKMT_STATUS_ERROR,
                        hsaKmtSetXNACKMode(enable));
                EXPECT_SUCCESS(queue.Destroy());
            }
        } else if (r == HSAKMT_STATUS_NOT_SUPPORTED) {
            LOG() << "XNACK mode: " << std::boolalpha << enable <<
                     " NOT supported" << std::endl;
        }
    }

    TEST_END
}

static void InvalidRangeTest(KFDTEST_PARAMETERS* pTestParamters) {

    HSAuint32 Flags;;
    HSAKMT_STATUS ret;

    int gpuNode = pTestParamters->gpuNode;
    KFDSVMRangeTest* pKFDSVMRangeTest = (KFDSVMRangeTest*)pTestParamters->pTestObject;

    if (!pKFDSVMRangeTest->SVMAPISupported_GPU(gpuNode))
        return;

    Flags = HSA_SVM_FLAG_HOST_ACCESS | HSA_SVM_FLAG_COHERENT;

    ret = RegisterSVMRange(gpuNode, reinterpret_cast<void *>(0x10000), 0x1000, 0, Flags);
    EXPECT_NE_GPU(ret, HSAKMT_STATUS_SUCCESS, gpuNode);

}

TEST_P(KFDSVMRangeTest, InvalidRangeTest) {
    TEST_START(TESTPROFILE_RUNALL)

    ASSERT_SUCCESS(KFDTest_Launch(InvalidRangeTest));

    TEST_END
}

void KFDSVMRangeTest::SplitRangeTest(int gpuNode, int prefetch_location) {
    unsigned int BufSize = 16 * PAGE_SIZE;

    HsaSVMRange *sysBuffer;
    HsaSVMRange *sysBuffer2;
    HsaSVMRange *sysBuffer3;
    HsaSVMRange *sysBuffer4;

    void *pBuf;

    // case 1
    pBuf = mmap(0, BufSize, PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
    sysBuffer = new HsaSVMRange(pBuf, BufSize, gpuNode, prefetch_location);
    sysBuffer2 = new HsaSVMRange(reinterpret_cast<char *>(pBuf) + 8192, PAGE_SIZE, gpuNode, prefetch_location);
    delete sysBuffer2;
    delete sysBuffer;
    munmap(pBuf, BufSize);

    // case 2.1
    pBuf = mmap(0, BufSize, PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
    sysBuffer = new HsaSVMRange(pBuf, BufSize, gpuNode, prefetch_location);
    sysBuffer2 = new HsaSVMRange(reinterpret_cast<char *>(pBuf) + 4096, BufSize - 4096, gpuNode,
                                 prefetch_location);
    delete sysBuffer2;
    delete sysBuffer;
    munmap(pBuf, BufSize);

    // case 2.2
    pBuf = mmap(0, BufSize + 8192, PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
    sysBuffer = new HsaSVMRange(pBuf, BufSize, gpuNode, prefetch_location);
    sysBuffer2 = new HsaSVMRange(reinterpret_cast<char *>(pBuf) + 8192, BufSize, gpuNode, prefetch_location);
    delete sysBuffer2;
    delete sysBuffer;
    munmap(pBuf, BufSize + 8192);

    // case 3
    pBuf = mmap(0, BufSize, PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
    sysBuffer = new HsaSVMRange(pBuf, BufSize, gpuNode, prefetch_location);
    sysBuffer2 = new HsaSVMRange(reinterpret_cast<char *>(pBuf), BufSize - 8192, gpuNode, prefetch_location);
    delete sysBuffer2;
    delete sysBuffer;
    munmap(pBuf, BufSize);

    // case 4.1
    pBuf = mmap(0, BufSize, PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
    sysBuffer = new HsaSVMRange(pBuf, BufSize, gpuNode, prefetch_location);
    sysBuffer2 = new HsaSVMRange(pBuf, BufSize, gpuNode, prefetch_location);
    delete sysBuffer2;
    delete sysBuffer;
    munmap(pBuf, BufSize);

    // case 4.2
    pBuf = mmap(0, BufSize + 8192, PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
    sysBuffer = new HsaSVMRange(pBuf, BufSize, gpuNode, prefetch_location);
    sysBuffer2 = new HsaSVMRange(pBuf, BufSize + 8192, gpuNode, prefetch_location);
    delete sysBuffer2;
    delete sysBuffer;
    munmap(pBuf, BufSize + 8192);

    // case 5
    pBuf = mmap(0, BufSize + 65536, PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
    sysBuffer = new HsaSVMRange(reinterpret_cast<char *>(pBuf) + 8192, 8192, gpuNode, prefetch_location);
    sysBuffer2 = new HsaSVMRange(reinterpret_cast<char *>(pBuf) + 32768, 8192, gpuNode, prefetch_location);
    sysBuffer3 = new HsaSVMRange(pBuf, BufSize + 65536, gpuNode, prefetch_location);
    delete sysBuffer2;
    delete sysBuffer3;
    delete sysBuffer;
    munmap(pBuf, BufSize + 65536);

    // case 6, unregister after free
    pBuf = mmap(0, BufSize, PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
    sysBuffer = new HsaSVMRange(reinterpret_cast<char *>(pBuf) + 8192, 8192, gpuNode, prefetch_location);
    munmap(pBuf, BufSize);
    delete sysBuffer;
}

static void SplitSystemRangeTest(KFDTEST_PARAMETERS* pTestParamters) {

    int gpuNode = pTestParamters->gpuNode;
    KFDSVMRangeTest* pKFDSVMRangeTest = (KFDSVMRangeTest*)pTestParamters->pTestObject;

    unsigned int m_FamilyId = pKFDSVMRangeTest->GetFamilyIdFromNodeId(gpuNode);
    if (m_FamilyId < FAMILY_AI) {
        LOG() << std::hex << "Skipping test: No svm range support for family ID 0x" << m_FamilyId << "." << std::endl;
        return;
    }

    if (!pKFDSVMRangeTest->SVMAPISupported_GPU(gpuNode))
        return;

    pKFDSVMRangeTest->SplitRangeTest(gpuNode, 0);

}

TEST_P(KFDSVMRangeTest, SplitSystemRangeTest) {
    TEST_START(TESTPROFILE_RUNALL)

    ASSERT_SUCCESS(KFDTest_Launch(SplitSystemRangeTest));

    TEST_END
}

static void EvictSystemRangeTest(KFDTEST_PARAMETERS* pTestParamters) {

    int gpuNode = pTestParamters->gpuNode;
    KFDSVMRangeTest* pKFDSVMRangeTest = (KFDSVMRangeTest*)pTestParamters->pTestObject;

    unsigned int m_FamilyId = pKFDSVMRangeTest->GetFamilyIdFromNodeId(gpuNode);
    if (m_FamilyId < FAMILY_AI) {
        LOG() << std::hex << "Skipping test: No svm range support for family ID 0x" << m_FamilyId << "." << std::endl;
        return;
    }

    if (!pKFDSVMRangeTest->SVMAPISupported_GPU(gpuNode))
        return;

    Assembler* m_pAsm;
    m_pAsm = pKFDSVMRangeTest->GetAssemblerFromNodeId(gpuNode);
    ASSERT_NOTNULL_GPU(m_pAsm, gpuNode);

    HSAuint32 stackData[2 * PAGE_SIZE] = {0};
    char *pBuf = reinterpret_cast<char *>(((uint64_t)stackData + PAGE_SIZE) & ~(PAGE_SIZE - 1));
    HSAuint32 *globalData = reinterpret_cast<uint32_t *>(pBuf);
    const unsigned dstOffset = ((uint64_t)pBuf + 2 * PAGE_SIZE - (uint64_t)stackData) / 4;
    const unsigned sdmaOffset = dstOffset + PAGE_SIZE;

    *globalData = 0xdeadbeef;

    HsaSVMRange srcBuffer((globalData), PAGE_SIZE, gpuNode);
    HsaSVMRange dstBuffer(&stackData[dstOffset], PAGE_SIZE, gpuNode);
    HsaSVMRange sdmaBuffer(&stackData[sdmaOffset], PAGE_SIZE, gpuNode);

    /* Create PM4 and SDMA queues before fork+COW to test queue
     * eviction and restore
     */
    PM4Queue pm4Queue;
    SDMAQueue sdmaQueue;
    ASSERT_SUCCESS_GPU(pm4Queue.Create(gpuNode), gpuNode);
    ASSERT_SUCCESS_GPU(sdmaQueue.Create(gpuNode), gpuNode);

    HsaMemoryBuffer isaBuffer(PAGE_SIZE, gpuNode, true/*zero*/, false/*local*/, true/*exec*/);

    ASSERT_SUCCESS_GPU(m_pAsm->RunAssembleBuf(CopyDwordIsa, isaBuffer.As<char*>()), gpuNode);

    Dispatch dispatch0(isaBuffer);
    dispatch0.SetArgs(srcBuffer.As<void*>(), dstBuffer.As<void*>());
    dispatch0.Submit(pm4Queue);
    dispatch0.Sync(g_TestTimeOut);

    sdmaQueue.PlaceAndSubmitPacket(SDMAWriteDataPacket(sdmaQueue.GetFamilyId(),
                                   sdmaBuffer.As<HSAuint32 *>(), 0x12345678));

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
    *globalData = 0xD00BED00;
    stackData[dstOffset] = 0xdeadbeef;
    stackData[sdmaOffset] = 0xdeadbeef;

    /* Terminate the child process before a possible test failure that
     * would leave it spinning in the background indefinitely.
     */
    int status;
    EXPECT_EQ_GPU(0, kill(pid, SIGTERM),gpuNode);
    EXPECT_EQ_GPU(pid, waitpid(pid, &status, 0), gpuNode);
    EXPECT_NE_GPU(0, WIFSIGNALED(status), gpuNode);
    EXPECT_EQ_GPU(SIGTERM, WTERMSIG(status), gpuNode);

    /* Now check that the GPU is accessing the correct page */
    Dispatch dispatch1(isaBuffer);
    dispatch1.SetArgs(srcBuffer.As<void*>(), dstBuffer.As<void*>());
    dispatch1.Submit(pm4Queue);
    dispatch1.Sync(g_TestTimeOut);

    sdmaQueue.PlaceAndSubmitPacket(SDMAWriteDataPacket(sdmaQueue.GetFamilyId(),
                                   sdmaBuffer.As<HSAuint32 *>(), 0xD0BED0BE));
    sdmaQueue.Wait4PacketConsumption();

    EXPECT_SUCCESS_GPU(pm4Queue.Destroy(), gpuNode);
    EXPECT_SUCCESS_GPU(sdmaQueue.Destroy(), gpuNode);

    EXPECT_EQ_GPU(0xD00BED00, *globalData, gpuNode);
    EXPECT_EQ_GPU(0xD00BED00, stackData[dstOffset], gpuNode);
    EXPECT_EQ_GPU(0xD0BED0BE, stackData[sdmaOffset],gpuNode);

}

TEST_P(KFDSVMRangeTest, EvictSystemRangeTest) {
    TEST_START(TESTPROFILE_RUNALL)

    ASSERT_SUCCESS(KFDTest_Launch(EvictSystemRangeTest));

    TEST_END
}

static void PartialUnmapSysMemTest(KFDTEST_PARAMETERS* pTestParamters) {

    int gpuNode = pTestParamters->gpuNode;
    KFDSVMRangeTest* pKFDSVMRangeTest = (KFDSVMRangeTest*)pTestParamters->pTestObject;

    if (!pKFDSVMRangeTest->SVMAPISupported_GPU(gpuNode))
        return;

    Assembler* m_pAsm;
    m_pAsm = pKFDSVMRangeTest->GetAssemblerFromNodeId(gpuNode);
    ASSERT_NOTNULL_GPU(m_pAsm, gpuNode);

    unsigned int BufSize = 16 * PAGE_SIZE;
    void *pBuf;

    PM4Queue queue;
    HsaMemoryBuffer isaBuffer(PAGE_SIZE, gpuNode, true/*zero*/, false/*local*/, true/*exec*/);
    HsaSVMRange *sysBuffer;
    HsaSVMRange destSysBuffer(BufSize, gpuNode);

    pBuf = mmap(0, BufSize, PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
    sysBuffer = new HsaSVMRange(pBuf, BufSize, gpuNode, 0);
    sysBuffer->Fill(0x01010101);

    char *pBuf2 = reinterpret_cast<char *>(pBuf) + 8192;
    unsigned int Buf2Size = 4 * PAGE_SIZE;
    char *pBuf3 = pBuf2 + Buf2Size;

    munmap(pBuf2, Buf2Size);

    ASSERT_SUCCESS_GPU(m_pAsm->RunAssembleBuf(CopyDwordIsa, isaBuffer.As<char*>()), gpuNode);

    ASSERT_SUCCESS_GPU(queue.Create(gpuNode), gpuNode);

    Dispatch dispatch(isaBuffer);
    Dispatch dispatch2(isaBuffer);

    dispatch.SetArgs(pBuf3, destSysBuffer.As<void*>());
    dispatch.Submit(queue);
    dispatch.Sync(g_TestTimeOut);
    EXPECT_EQ_GPU(destSysBuffer.As<unsigned int*>()[0], 0x01010101, gpuNode);

    dispatch2.SetArgs(pBuf, destSysBuffer.As<void*>());
    dispatch2.Submit(queue);
    dispatch2.Sync(g_TestTimeOut);

    EXPECT_EQ_GPU(destSysBuffer.As<unsigned int*>()[0], 0x01010101, gpuNode);

    EXPECT_SUCCESS_GPU(queue.Destroy(), gpuNode);

    //munmap(pBuf, BufSize);
    /* munmpa vm ranges that has not been done */
    munmap(pBuf, 8192);
    munmap(pBuf3, BufSize - 8192 - Buf2Size);

}

TEST_P(KFDSVMRangeTest, PartialUnmapSysMemTest) {
    TEST_REQUIRE_ENV_CAPABILITIES(ENVCAPS_64BITLINUX);
    TEST_START(TESTPROFILE_RUNALL);

   ASSERT_SUCCESS(KFDTest_Launch(PartialUnmapSysMemTest));

    TEST_END
}

static void BasicVramTest(KFDTEST_PARAMETERS* pTestParamters) {

    int gpuNode = pTestParamters->gpuNode;
    KFDSVMRangeTest* pKFDSVMRangeTest = (KFDSVMRangeTest*)pTestParamters->pTestObject;

    if (!pKFDSVMRangeTest->SVMAPISupported_GPU(gpuNode))
        return;

    Assembler* m_pAsm;
    m_pAsm = pKFDSVMRangeTest->GetAssemblerFromNodeId(gpuNode);
    ASSERT_NOTNULL_GPU(m_pAsm, gpuNode);

    PM4Queue queue;
    HSAuint64 AlternateVAGPU;
    unsigned int BufferSize = PAGE_SIZE;

    if (!pKFDSVMRangeTest->GetVramSize(gpuNode)) {
        LOG() << "Skipping test: No VRAM found." << std::endl;
        return;
    }

    HsaMemoryBuffer isaBuffer(PAGE_SIZE, gpuNode, true/*zero*/, false/*local*/, true/*exec*/);
    HsaSVMRange srcSysBuffer(BufferSize, gpuNode);
    HsaSVMRange locBuffer(BufferSize, gpuNode, gpuNode);
    HsaSVMRange destSysBuffer(BufferSize, gpuNode);

    srcSysBuffer.Fill(0x01010101);

    ASSERT_SUCCESS_GPU(m_pAsm->RunAssembleBuf(CopyDwordIsa, isaBuffer.As<char*>()), gpuNode);

    ASSERT_SUCCESS_GPU(queue.Create(gpuNode), gpuNode);
    queue.SetSkipWaitConsump(0);

    Dispatch dispatch(isaBuffer);
    Dispatch dispatch2(isaBuffer);

    dispatch.SetArgs(srcSysBuffer.As<void*>(), locBuffer.As<void*>());
    dispatch.Submit(queue);
    dispatch.Sync(g_TestTimeOut);

    dispatch2.SetArgs(locBuffer.As<void*>(), destSysBuffer.As<void*>());
    dispatch2.Submit(queue);
    dispatch2.Sync(g_TestTimeOut);

    EXPECT_SUCCESS_GPU(queue.Destroy(), gpuNode);

    EXPECT_EQ_GPU(destSysBuffer.As<unsigned int*>()[0], 0x01010101, gpuNode);

}

TEST_P(KFDSVMRangeTest, BasicVramTest) {
    TEST_REQUIRE_ENV_CAPABILITIES(ENVCAPS_64BITLINUX);
    TEST_START(TESTPROFILE_RUNALL);

    ASSERT_SUCCESS(KFDTest_Launch(BasicVramTest));

    TEST_END
}

static void SplitVramRangeTest(KFDTEST_PARAMETERS* pTestParamters) {

    int gpuNode = pTestParamters->gpuNode;
    KFDSVMRangeTest* pKFDSVMRangeTest = (KFDSVMRangeTest*)pTestParamters->pTestObject;

    if (!pKFDSVMRangeTest->SVMAPISupported_GPU(gpuNode))
        return;

    unsigned int m_FamilyId = pKFDSVMRangeTest->GetFamilyIdFromNodeId(gpuNode);
    if (m_FamilyId < FAMILY_AI) {
        LOG() << std::hex << "Skipping test: No svm range support for family ID 0x" << m_FamilyId << "." << std::endl;
        return;
    }

    pKFDSVMRangeTest->SplitRangeTest(gpuNode, gpuNode);

}

TEST_P(KFDSVMRangeTest, SplitVramRangeTest) {
    TEST_START(TESTPROFILE_RUNALL)

    ASSERT_SUCCESS(KFDTest_Launch(SplitVramRangeTest));

    TEST_END
}

static void PrefetchTest(KFDTEST_PARAMETERS* pTestParamters) {

    int gpuNode = pTestParamters->gpuNode;
    KFDSVMRangeTest* pKFDSVMRangeTest = (KFDSVMRangeTest*)pTestParamters->pTestObject;

    if (!pKFDSVMRangeTest->SVMAPISupported_GPU(gpuNode))
        return;

    unsigned int BufSize = 16 << 10;
    HsaSVMRange *sysBuffer;
    uint32_t node_id;

    sysBuffer = new HsaSVMRange(BufSize, gpuNode);
    char *pBuf = sysBuffer->As<char *>();
    delete sysBuffer;

    /* after mumap sysBuffer it should be not accessible from gpuNode */
    HSA_SVM_ATTRIBUTE attr;
    attr.type = HSA_SVM_ATTR_ACCESS;
    attr.value = 0;
    /* hsaKmtSVMGetAttr for HSA_SVM_ATTR_ACCESS is either fail or
     * returned attr.value not equal gpuNode
     */
    if (hsaKmtSVMGetAttr(pBuf, BufSize, 1, &attr) == HSAKMT_STATUS_SUCCESS)
        EXPECT_NE_GPU(attr.value, gpuNode, gpuNode);

    sysBuffer = new HsaSVMRange(BufSize, gpuNode);
    pBuf = sysBuffer->As<char *>();
    char *pLocBuf = pBuf + BufSize / 2;

    EXPECT_SUCCESS_GPU(SVMRangeGetPrefetchNode(pBuf, BufSize, &node_id), gpuNode);
    EXPECT_EQ_GPU(node_id, 0, gpuNode);

    EXPECT_SUCCESS_GPU(SVMRangePrefetchToNode(pLocBuf, BufSize / 2, gpuNode), gpuNode);

    EXPECT_SUCCESS_GPU(SVMRangeGetPrefetchNode(pLocBuf, BufSize / 2, &node_id), gpuNode);
    EXPECT_EQ_GPU(node_id, gpuNode, gpuNode);

    EXPECT_SUCCESS_GPU(SVMRangeGetPrefetchNode(pBuf, BufSize, &node_id), gpuNode);
    EXPECT_EQ_GPU(node_id, 0xffffffff, gpuNode);
    delete sysBuffer;

}

TEST_P(KFDSVMRangeTest, PrefetchTest) {
    TEST_START(TESTPROFILE_RUNALL);

    ASSERT_SUCCESS(KFDTest_Launch(PrefetchTest));

    TEST_END
}

static void MigrateTest(KFDTEST_PARAMETERS* pTestParamters) {

    int gpuNode = pTestParamters->gpuNode;
    KFDSVMRangeTest* pKFDSVMRangeTest = (KFDSVMRangeTest*)pTestParamters->pTestObject;

    if (!pKFDSVMRangeTest->SVMAPISupported_GPU(gpuNode))
        return;

    unsigned int m_FamilyId = pKFDSVMRangeTest->GetFamilyIdFromNodeId(gpuNode);
    if (m_FamilyId < FAMILY_AI) {
        LOG() << std::hex << "Skipping test: No svm range support for family ID 0x" << m_FamilyId << "." << std::endl;
        return;
    }

    if (!pKFDSVMRangeTest->GetVramSize(gpuNode)) {
        LOG() << "Skipping test: No VRAM found." << std::endl;
        return;
    }

    HSAuint32 migrateRepeat = 8;
    unsigned int BufferSize = 16 << 20;

    HsaSVMRange DataBuffer(BufferSize, gpuNode);
    HSAuint32 *pData = DataBuffer.As<HSAuint32 *>();

    HsaSVMRange SysBuffer(BufferSize, gpuNode);
    HSAuint32 *pBuf = SysBuffer.As<HSAuint32 *>();
    EXPECT_SUCCESS_GPU(SVMRangePrefetchToNode(pBuf, BufferSize, 0), gpuNode);

    HsaSVMRange SysBuffer2(BufferSize, gpuNode);
    HSAuint32 *pBuf2 = SysBuffer2.As<HSAuint32 *>();
    EXPECT_SUCCESS_GPU(SVMRangePrefetchToNode(pBuf2, BufferSize, 0), gpuNode);

    SDMAQueue sdmaQueue;
    ASSERT_SUCCESS_GPU(sdmaQueue.Create(gpuNode), gpuNode);

    for (HSAuint32 i = 0; i < BufferSize / 4; i++)
        pData[i] = i;

    while (migrateRepeat--) {
        /* Migrate from ram to vram */
        EXPECT_SUCCESS_GPU(SVMRangePrefetchToNode(pBuf, BufferSize, gpuNode), gpuNode);
        EXPECT_SUCCESS_GPU(SVMRangePrefetchToNode(pBuf2, BufferSize, gpuNode), gpuNode);
        /* Update content in migrated buffer in vram */
        sdmaQueue.PlaceAndSubmitPacket(SDMACopyDataPacket(sdmaQueue.GetFamilyId(),
                    pBuf, pData, BufferSize));
        sdmaQueue.Wait4PacketConsumption();
        sdmaQueue.PlaceAndSubmitPacket(SDMACopyDataPacket(sdmaQueue.GetFamilyId(),
                    pBuf2, pData, BufferSize));
        sdmaQueue.Wait4PacketConsumption();

        /* Migrate from vram to ram
         * CPU access the buffer migrated to vram have page fault
         * page fault trigger migration from vram back to ram
         * so SysBuffer should have same value as in vram
         */
        for (HSAuint32 i = 0; i < BufferSize / 4; i++) {
            ASSERT_EQ_GPU(i, pBuf[i], gpuNode);
            ASSERT_EQ_GPU(i, pBuf2[i], gpuNode);
        }
   }

    /* If xnack off, after migrating back to ram, GPU mapping should be updated to ram
     * test if shade can read from ram
     * If xnack on, GPU mapping should be cleared, test if GPU vm fault can update
     * page table and shade can read from ram.
     */
    sdmaQueue.PlaceAndSubmitPacket(SDMACopyDataPacket(sdmaQueue.GetFamilyId(),
                pBuf, pData, BufferSize));
    sdmaQueue.Wait4PacketConsumption();
    for (HSAuint32 i = 0; i < BufferSize / 4; i++)
        ASSERT_EQ_GPU(i, pBuf[i], gpuNode);

}

TEST_P(KFDSVMRangeTest, MigrateTest) {
    TEST_REQUIRE_ENV_CAPABILITIES(ENVCAPS_64BITLINUX);
    TEST_START(TESTPROFILE_RUNALL);

    ASSERT_SUCCESS(KFDTest_Launch(MigrateTest));

    TEST_END
}

static void MigrateAccessInPlaceTest(KFDTEST_PARAMETERS* pTestParamters) {

    int gpuNode = pTestParamters->gpuNode;
    KFDSVMRangeTest* pKFDSVMRangeTest = (KFDSVMRangeTest*)pTestParamters->pTestObject;

    if (!pKFDSVMRangeTest->SVMAPISupported_GPU(gpuNode))
        return;

    unsigned int m_FamilyId = pKFDSVMRangeTest->GetFamilyIdFromNodeId(gpuNode);
    if (m_FamilyId < FAMILY_AI) {
        LOG() << std::hex << "Skipping test: No svm range support for family ID 0x" << m_FamilyId << "." << std::endl;
        return;
    }

    if (!pKFDSVMRangeTest->GetVramSize(gpuNode)) {
        LOG() << "Skipping test: No VRAM found." << std::endl;
        return;
    }

    unsigned int BufferSize = MIN(256ULL << 20, pKFDSVMRangeTest->GetVramSize(gpuNode) / 2);
    SDMAQueue sdmaQueue;
    ASSERT_SUCCESS_GPU(sdmaQueue.Create(gpuNode),gpuNode);

    HsaSVMRange DataBuffer(BufferSize, gpuNode);
    HSAuint32 *pData = DataBuffer.As<HSAuint32 *>();

    EXPECT_SUCCESS_GPU(SVMRangeMapInPlaceToNode(pData, BufferSize, gpuNode), gpuNode);
    EXPECT_SUCCESS_GPU(SVMRangePrefetchToNode(pData, BufferSize, gpuNode), gpuNode);

    for (HSAuint32 i = 0; i < BufferSize / 4; i += 1024)
        pData[i] = i;

    /* GPU/SDMA update content in buffer migrated back to system memory */
    sdmaQueue.PlaceAndSubmitPacket(SDMAFillDataPacket(sdmaQueue.GetFamilyId(),
           pData, 0x55AAAA55, BufferSize));
    sdmaQueue.Wait4PacketConsumption();

    for (HSAuint32 i = 0; i < BufferSize / 4; i += 1024)
        ASSERT_EQ_GPU(0x55AAAA55, pData[i], gpuNode);

    ASSERT_SUCCESS_GPU(sdmaQueue.Destroy(), gpuNode);

}
/*
 * Test if GPU mapping to system memory is correct after range on VRAM split and migrate back
 * to system memory.
 *
 * Steps, it is same for XNACK on or off
 *   1. alloc 256MB range on system memory, set ACCESS_IN_PLACE by GPU
 *   2. Prefetcg to migrate range to GPU VRAM
 *   3. Use CPU to fill the range, range is migrated back to system memory, and split by granularity,
 *      GPU mapping update to system memory
 *   4. Use GPU sdma to fill the range in system memory
 *   5. Check if data is correct in system memory
 */
TEST_P(KFDSVMRangeTest, MigrateAccessInPlaceTest) {
    TEST_REQUIRE_ENV_CAPABILITIES(ENVCAPS_64BITLINUX);
    TEST_START(TESTPROFILE_RUNALL);

    ASSERT_SUCCESS(KFDTest_Launch(MigrateAccessInPlaceTest));

    TEST_END
}

/*
 * The test changes migration granularity, then trigger CPU page fault to migrate
 * the svm range from vram to ram.
 * Check the dmesg driver output to confirm the number of CPU page fault is correct
 * based on granularity.
 *
 * For example, this is BufferPages = 5, while granularity change from 2 to 0
 * [  292.623498] amdgpu:svm_migrate_to_ram:744: CPU page fault address 0x7f22597ee000
 * [  292.623727] amdgpu:svm_migrate_to_ram:744: CPU page fault address 0x7f22597f0000
 * [  292.724414] amdgpu:svm_migrate_to_ram:744: CPU page fault address 0x7f22597ee000
 * [  292.724824] amdgpu:svm_migrate_to_ram:744: CPU page fault address 0x7f22597f0000
 * [  292.725094] amdgpu:svm_migrate_to_ram:744: CPU page fault address 0x7f22597f2000
 * [  292.728186] amdgpu:svm_migrate_to_ram:744: CPU page fault address 0x7f22597ee000
 * [  292.729171] amdgpu:svm_migrate_to_ram:744: CPU page fault address 0x7f22597ef000
 * [  292.729576] amdgpu:svm_migrate_to_ram:744: CPU page fault address 0x7f22597f0000
 * [  292.730010] amdgpu:svm_migrate_to_ram:744: CPU page fault address 0x7f22597f1000
 * [  292.730931] amdgpu:svm_migrate_to_ram:744: CPU page fault address 0x7f22597f2000
 */

static void MigrateGranularityTest(KFDTEST_PARAMETERS* pTestParamters) {

    int gpuNode = pTestParamters->gpuNode;
    KFDSVMRangeTest* pKFDSVMRangeTest = (KFDSVMRangeTest*)pTestParamters->pTestObject;

    if (!pKFDSVMRangeTest->SVMAPISupported_GPU(gpuNode))
        return;

    unsigned int m_FamilyId = pKFDSVMRangeTest->GetFamilyIdFromNodeId(gpuNode);
    if (m_FamilyId < FAMILY_AI) {
        LOG() << std::hex << "Skipping test on gpuNode: No svm range support for family ID 0x" << gpuNode << m_FamilyId << "." << std::endl;
        return;
    }

    if (!pKFDSVMRangeTest->GetVramSize(gpuNode)) {
        LOG() << "Skipping test: No VRAM found on gpuNode." << gpuNode << std::endl;
        return;
    }

    HSAuint64 BufferPages = 16384;
    HSAuint64 BufferSize = BufferPages * PAGE_SIZE;
    HsaSVMRange SysBuffer(BufferSize, gpuNode);
    HSAint32 *pBuf = SysBuffer.As<HSAint32*>();

    HsaSVMRange SysBuffer2(BufferSize, gpuNode);
    HSAint32 *pBuf2 = SysBuffer2.As<HSAint32*>();

    HSAint32 Granularity;

    SDMAQueue sdmaQueue;
    ASSERT_SUCCESS_GPU(sdmaQueue.Create(gpuNode), gpuNode);

    for (Granularity = 0; (1ULL << Granularity) <= BufferPages; Granularity++);
    for (HSAuint32 i = 0; i < BufferPages; i++)
        pBuf2[i * PAGE_SIZE / 4] = i;

    while (Granularity--) {
        /* Prefetch the entire range to vram */
        EXPECT_SUCCESS_GPU(SVMRangePrefetchToNode(pBuf, BufferSize, gpuNode), gpuNode);
        EXPECT_SUCCESS_GPU(SVMRangSetGranularity(pBuf, BufferSize, Granularity), gpuNode);

        /* Change Buffer content in vram, then migrate it back to ram */
        sdmaQueue.PlaceAndSubmitPacket(SDMACopyDataPacket(sdmaQueue.GetFamilyId(),
                        pBuf, pBuf2, BufferSize));
        sdmaQueue.Wait4PacketConsumption();

        /* Migrate from vram to ram */
        for (HSAuint32 i = 0; i < BufferPages; i++)
            ASSERT_EQ_GPU(i, pBuf[i * PAGE_SIZE / 4], gpuNode);
    }

}

TEST_P(KFDSVMRangeTest, MigrateGranularityTest) {
    TEST_REQUIRE_ENV_CAPABILITIES(ENVCAPS_64BITLINUX);
    TEST_START(TESTPROFILE_RUNALL);

    ASSERT_SUCCESS(KFDTest_Launch(MigrateGranularityTest));

    TEST_END
}

static void MigrateLargeBufTest(KFDTEST_PARAMETERS* pTestParamters) {

    int gpuNode = pTestParamters->gpuNode;
    KFDSVMRangeTest* pKFDSVMRangeTest = (KFDSVMRangeTest*)pTestParamters->pTestObject;

    if (!pKFDSVMRangeTest->SVMAPISupported_GPU(gpuNode))
        return;

    PM4Queue queue;
    HSAuint64 AlternateVAGPU;
    unsigned long BufferSize = 1L << 30;

    unsigned long maxSDMASize = 128L << 20;  /* IB size is 4K */
    unsigned long Size, i;

    HSAuint64 vramSize;
    vramSize = pKFDSVMRangeTest->GetVramSize(gpuNode);
    if (!vramSize) {
        LOG() << "Skipping test: No VRAM found." << std::endl;
        return;
    }

    BufferSize = MIN(BufferSize, vramSize * 3 / 4);

    /* Check if the system memory size is sufficient
     * to register the system buffer and system buffer 2
     */
    if(BufferSize * 2 > pKFDSVMRangeTest->GetSysMemSize() / 2) {
        LOG() << "Skipping test: Not enough system memory." << std::endl;
        return;
    }
    HsaSVMRange SysBuffer(BufferSize, gpuNode);
    SysBuffer.Fill(0x1);

    HsaSVMRange SysBuffer2(BufferSize, gpuNode);
    SysBuffer2.Fill(0x2);

    /* Migrate from ram to vram
     * using same address to register to GPU to trigger migration
     * so LocalBuffer will have same value as SysBuffer
     */
    HsaSVMRange LocalBuffer(SysBuffer.As<void*>(), BufferSize, gpuNode, gpuNode);

    SDMAQueue sdmaQueue;

    ASSERT_SUCCESS_GPU(sdmaQueue.Create(gpuNode), gpuNode);
    for (i = 0; i < BufferSize; i += Size) {
        Size = (BufferSize - i) > maxSDMASize ? maxSDMASize : (BufferSize - i);
        sdmaQueue.PlaceAndSubmitPacket(SDMACopyDataPacket(sdmaQueue.GetFamilyId(),
                    SysBuffer2.As<char*>() + i, LocalBuffer.As<char*>() + i, Size));
        sdmaQueue.Wait4PacketConsumption();
    }

    /* Check content in migrated buffer in vram */
    for (i = 0; i < BufferSize / 4; i += 1024)
        ASSERT_EQ_GPU(0x1, SysBuffer2.As<unsigned int*>()[i], gpuNode);

    /* Change LocalBuffer content in vram, then migrate it back to ram */
    SysBuffer2.Fill(0x3);

    for (i = 0; i < BufferSize; i += Size) {
        Size = (BufferSize - i) > maxSDMASize ? maxSDMASize : (BufferSize - i);
        sdmaQueue.PlaceAndSubmitPacket(SDMACopyDataPacket(sdmaQueue.GetFamilyId(),
                    LocalBuffer.As<char*>() + i, SysBuffer2.As<char*>() + i, Size));
        sdmaQueue.Wait4PacketConsumption();
    }

    /* Migrate from vram to ram
     * CPU access the buffer migrated to vram have page fault
     * page fault trigger migration from vram back to ram
     * so SysBuffer should have same value as in LocalBuffer
     */
    EXPECT_SUCCESS_GPU(SVMRangSetGranularity(SysBuffer.As<unsigned int*>(), BufferSize, 30),gpuNode);
    for (i = 0; i < BufferSize / 4; i += 1024)
        ASSERT_EQ_GPU(0x3, SysBuffer.As<unsigned int*>()[i], gpuNode);

    /* After migrating back to ram, GPU mapping should be updated to ram
     * test if shade can read from ram
     */
    SysBuffer.Fill(0x4);

    for (i = 0; i < BufferSize; i += Size) {
        Size = (BufferSize - i) > maxSDMASize ? maxSDMASize : (BufferSize - i);
        sdmaQueue.PlaceAndSubmitPacket(SDMACopyDataPacket(sdmaQueue.GetFamilyId(),
                    SysBuffer2.As<char*>() + i, LocalBuffer.As<char*>() + i, Size));
        sdmaQueue.Wait4PacketConsumption();
    }

    for (i = 0; i < BufferSize / 4; i += 1024)
        ASSERT_EQ_GPU(0x4, SysBuffer2.As<unsigned int*>()[i],gpuNode);

}

TEST_P(KFDSVMRangeTest, MigrateLargeBufTest) {
    TEST_REQUIRE_ENV_CAPABILITIES(ENVCAPS_64BITLINUX);
    TEST_START(TESTPROFILE_RUNALL);

    ASSERT_SUCCESS(KFDTest_Launch(MigrateLargeBufTest));

    TEST_END
}

static void MigratePolicyTest(KFDTEST_PARAMETERS* pTestParamters) {

    int gpuNode = pTestParamters->gpuNode;
    KFDSVMRangeTest* pKFDSVMRangeTest = (KFDSVMRangeTest*)pTestParamters->pTestObject;

    if (!pKFDSVMRangeTest->SVMAPISupported_GPU(gpuNode))
        return;

    unsigned int m_FamilyId = pKFDSVMRangeTest->GetFamilyIdFromNodeId(gpuNode);
    if (m_FamilyId < FAMILY_AI) {
        LOG() << std::hex << "Skipping test on gpuNode: No svm range support for family ID 0x" << gpuNode << m_FamilyId << "." << std::endl;
        return;
    }

    if (!pKFDSVMRangeTest->GetVramSize(gpuNode)) {
        LOG() << "Skipping test: No VRAM found." << std::endl;
        return;
    }

    unsigned long BufferSize = 1UL << 20;

    HsaSVMRange DataBuffer(BufferSize, gpuNode);
    HSAuint64 *pData = DataBuffer.As<HSAuint64 *>();

    HsaSVMRange SysBuffer(BufferSize, gpuNode);
    HSAuint64 *pBuf = SysBuffer.As<HSAuint64 *>();

    SDMAQueue sdmaQueue;
    ASSERT_SUCCESS_GPU(sdmaQueue.Create(gpuNode),gpuNode);

    for (HSAuint64 i = 0; i < BufferSize / 8; i++)
        pData[i] = i;

    /* Prefetch to migrate from ram to vram */
    EXPECT_SUCCESS_GPU(SVMRangePrefetchToNode(pBuf, BufferSize, gpuNode),gpuNode);

    /* Update content in migrated buffer in vram */
    sdmaQueue.PlaceAndSubmitPacket(SDMACopyDataPacket(sdmaQueue.GetFamilyId(),
                pBuf, pData, BufferSize));
    sdmaQueue.Wait4PacketConsumption(NULL, HSA_EVENTTIMEOUT_INFINITE);

    /* Migrate from vram to ram
     * CPU access the buffer migrated to vram have page fault
     * page fault trigger migration from vram back to ram
     * so SysBuffer should have same value as in vram
     */
    for (HSAuint64 i = 0; i < BufferSize / 8; i++) {
        ASSERT_EQ_GPU(i, pBuf[i],gpuNode);
        /* Update buf */
        pBuf[i] = i + 1;
    }

    /* Migrate from ram to vram if xnack on
     * If xnack off, after migrating back to ram, GPU mapping should be updated to ram
     * test if shade can read from ram
     * If xnack on, GPU mapping should be cleared, test if GPU vm fault can update
     * page table and shade can read from ram.
     */
//#define USE_PM4_QUEUE_TRIGGER_VM_FAULT
#ifdef USE_PM4_QUEUE_TRIGGER_VM_FAULT
    HsaMemoryBuffer isaBuffer(PAGE_SIZE, gpuNode, true/*zero*/, false/*local*/, true/*exec*/);
    PM4Queue queue;

    ASSERT_SUCCESS_GPU(m_pAsm->RunAssembleBuf(CopyDwordIsa, isaBuffer.As<char*>()),gpuNode);

    ASSERT_SUCCESS_GPU(queue.Create(gpuNode),gpuNode);

    for (HSAuint64 i = 0; i < BufferSize / 8; i += 512) {
        Dispatch dispatch(isaBuffer);
        
        dispatch.SetArgs(pBuf + i, pData + i);
        dispatch.Submit(queue);
        dispatch.Sync(HSA_EVENTTIMEOUT_INFINITE);
    }
#else
    sdmaQueue.PlaceAndSubmitPacket(SDMACopyDataPacket(sdmaQueue.GetFamilyId(),
                pData, pBuf, BufferSize));
    sdmaQueue.Wait4PacketConsumption(NULL, HSA_EVENTTIMEOUT_INFINITE);
#endif

    for (HSAuint64 i = 0; i < BufferSize / 8; i += 512)
        ASSERT_EQ_GPU(i + 1, pData[i],gpuNode);

    ASSERT_SUCCESS_GPU(sdmaQueue.Destroy(),gpuNode);

}

TEST_P(KFDSVMRangeTest, MigratePolicyTest) {
    TEST_REQUIRE_ENV_CAPABILITIES(ENVCAPS_64BITLINUX);
    TEST_START(TESTPROFILE_RUNALL);

    ASSERT_SUCCESS(KFDTest_Launch(MigratePolicyTest));

    TEST_END
}

/* Multiple GPU migration test
 *
 * Steps:
 *     1. Prefetch pBuf, pData to all GPUs, to test migration from GPU to GPU
 *     2. Use sdma queue on all GPUs, to copy data from pBuf to pData
 *     3. Check pData data
 *
 * Notes:
 *     With xnack on, step 2 will have retry fault on pBuf, to migrate from GPU to GPU,
 *     retry fault on pData, to migrate from CPU to GPU
 *
 *     With xnack off, pBuf and pData should prefetch to CPU to ensure multiple GPU access
 *
 *     step3 migrate pData from GPU to CPU
 *
 * Test will skip if only one GPU found
 */
TEST_P(KFDSVMRangeTest, MultiGPUMigrationTest) {
    TEST_REQUIRE_ENV_CAPABILITIES(ENVCAPS_64BITLINUX);
    TEST_START(TESTPROFILE_RUNALL);

    if (!SVMAPISupported())
        return;

    int defaultGPUNode = m_NodeInfo.HsaDefaultGPUNode();
    ASSERT_GE(defaultGPUNode, 0) << "failed to get default GPU Node";

    if (m_FamilyId < FAMILY_AI) {
        LOG() << std::hex << "Skipping test: No svm range support for family ID 0x" << m_FamilyId << "." << std::endl;
        return;
    }

    const std::vector<int> gpuNodesAll = m_NodeInfo.GetNodesWithGPU();
    std::vector<int> gpuNodes;

    for (auto node : gpuNodesAll) {
        const HsaNodeProperties *pNodeProperties;

        pNodeProperties = m_NodeInfo.GetNodeProperties(node);
        if (pNodeProperties->Capability.ui32.SVMAPISupported)
            gpuNodes.push_back(node);
    }
    if (gpuNodes.size() < 2) {
        LOG() << "Skipping test: at least two SVM supported GPUs needed." << std::endl;
        return;
    }

    unsigned long BufferSize = 1UL << 20;

    HsaSVMRange SysBuffer(BufferSize, defaultGPUNode);
    HSAuint64 *pBuf = SysBuffer.As<HSAuint64 *>();
    HsaSVMRange DataBuffer(BufferSize, defaultGPUNode);
    HSAuint64 *pData = DataBuffer.As<HSAuint64 *>();

    SDMAQueue sdmaQueue;

    for (HSAuint64 i = 0; i < BufferSize / 8; i++)
        pBuf[i] = i;

    for (auto node : gpuNodes) {
        EXPECT_SUCCESS(SVMRangeMapToNode(pBuf, BufferSize, node));
        EXPECT_SUCCESS(SVMRangePrefetchToNode(pBuf, BufferSize, node));

        EXPECT_SUCCESS(SVMRangeMapToNode(pData, BufferSize, node));
        EXPECT_SUCCESS(SVMRangePrefetchToNode(pData, BufferSize, node));
    }

    for (auto node : gpuNodes) {
        ASSERT_SUCCESS(sdmaQueue.Create(node));

        sdmaQueue.PlaceAndSubmitPacket(SDMACopyDataPacket(sdmaQueue.GetFamilyId(),
                    pData, pBuf, BufferSize));
        sdmaQueue.Wait4PacketConsumption();

        for (HSAuint64 i = 0; i < BufferSize / 8; i += 512)
            ASSERT_EQ(i, pData[i]);

        EXPECT_SUCCESS(sdmaQueue.Destroy());
    }

    TEST_END
}

/* Multiple GPU access in place test
 *
 * Steps:
 *     1. Prefetch pBuf, pData to all GPUs, with ACCESS_IN_PLACE on GPUs
 *     2. Use sdma queue on all GPUs, to copy data from pBuf to pData
 *     3. Prefetch pData to CPU, check pData data
 *
 * Notes:
 *     With xnack on, step 2 will have retry fault on pBuf, to migrate from GPU to GPU.
 *     If multiple GPU on xGMI same hive, there should not have retry fault on pBuf
 *     because mapping should update to another GPU vram through xGMI
 *
 *     With xnack off, pBuf and pData should prefetch to CPU to ensure multiple GPU access
 *
 *     step3 migrate pData from GPU to CPU, should not have retry fault on GPUs.
 *
 * Test will skip if only one GPU found
 */
TEST_P(KFDSVMRangeTest, MultiGPUAccessInPlaceTest) {
    TEST_REQUIRE_ENV_CAPABILITIES(ENVCAPS_64BITLINUX);
    TEST_START(TESTPROFILE_RUNALL);

    if (!SVMAPISupported())
        return;

    int defaultGPUNode = m_NodeInfo.HsaDefaultGPUNode();
    ASSERT_GE(defaultGPUNode, 0) << "failed to get default GPU Node";

    if (m_FamilyId < FAMILY_AI) {
        LOG() << std::hex << "Skipping test: No svm range support for family ID 0x" << m_FamilyId << "." << std::endl;
        return;
    }

    const std::vector<int> gpuNodesAll = m_NodeInfo.GetNodesWithGPU();
    std::vector<int> gpuNodes;

    for (auto node : gpuNodesAll) {
        const HsaNodeProperties *pNodeProperties;

        pNodeProperties = m_NodeInfo.GetNodeProperties(node);
        if (pNodeProperties->Capability.ui32.SVMAPISupported)
            gpuNodes.push_back(node);
    }
    if (gpuNodes.size() < 2) {
        LOG() << "Skipping test: at least two SVM supported GPUs needed." << std::endl;
        return;
    }

    unsigned long BufferSize = 1UL << 20;

    HsaSVMRange SysBuffer(BufferSize, defaultGPUNode);
    HSAuint64 *pBuf = SysBuffer.As<HSAuint64 *>();
    HsaSVMRange DataBuffer(BufferSize, defaultGPUNode);
    HSAuint64 *pData = DataBuffer.As<HSAuint64 *>();

    SDMAQueue sdmaQueue;

    for (HSAuint64 i = 0; i < BufferSize / 8; i++)
        pBuf[i] = i;

    for (auto node : gpuNodes) {
        EXPECT_SUCCESS(SVMRangeMapInPlaceToNode(pBuf, BufferSize, node));
        EXPECT_SUCCESS(SVMRangePrefetchToNode(pBuf, BufferSize, node));

        EXPECT_SUCCESS(SVMRangeMapInPlaceToNode(pData, BufferSize, node));
        EXPECT_SUCCESS(SVMRangePrefetchToNode(pData, BufferSize, node));
    }

    for (auto node : gpuNodes) {
        ASSERT_SUCCESS(sdmaQueue.Create(node));

        sdmaQueue.PlaceAndSubmitPacket(SDMACopyDataPacket(sdmaQueue.GetFamilyId(),
                    pData, pBuf, BufferSize));
        sdmaQueue.Wait4PacketConsumption();

        for (HSAuint64 i = 0; i < BufferSize / 8; i += 512)
            ASSERT_EQ(i, pData[i]);

        EXPECT_SUCCESS(sdmaQueue.Destroy());
    }

    TEST_END
}

/* Multiple thread migration test
 *
 * 2 threads do migration at same time to test range migration race conditon handle.
 *
 * Steps:
 * 1. register 128MB range on system memory, don't map to GPU, 128MB is max size to put in
 *    sdma queue 4KB IB buffer.
 * 2. one thread prefetch range to GPU, another thread use sdma queue to access range at same
 *    time to generate retry vm fault to migrate range to GPU
 * 3. one thread prefetch range to CPU, another thread read range to generate CPU page fault
 *    to migrate range to CPU at same time
 * 4. loop test step 2 and 3 twice, to random CPU/GPU fault and prefetch migration order
 */
struct ReadThreadParams {
    HSAuint64* pBuf;
    HSAint64 BufferSize;
    int defaultGPUNode;
};

unsigned int CpuReadThread(void* p) {
    struct ReadThreadParams* pArgs = reinterpret_cast<struct ReadThreadParams*>(p);

    for (HSAuint64 i = 0; i < pArgs->BufferSize / 8; i += 512)
         EXPECT_EQ(i, pArgs->pBuf[i]);
    return 0;
}

unsigned int GpuReadThread(void* p) {
    struct ReadThreadParams* pArgs = reinterpret_cast<struct ReadThreadParams*>(p);

    EXPECT_SUCCESS(SVMRangePrefetchToNode(pArgs->pBuf, pArgs->BufferSize, pArgs->defaultGPUNode));
    return 0;
}

static void MultiThreadMigrationTest(KFDTEST_PARAMETERS* pTestParamters) {

    int gpuNode = pTestParamters->gpuNode;
    KFDSVMRangeTest* pKFDSVMRangeTest = (KFDSVMRangeTest*)pTestParamters->pTestObject;

    if (!pKFDSVMRangeTest->SVMAPISupported_GPU(gpuNode))
        return;

    unsigned int m_FamilyId = pKFDSVMRangeTest->GetFamilyIdFromNodeId(gpuNode);
    if (m_FamilyId < FAMILY_AI) {
        LOG() << std::hex << "Skipping test on gpuNode: No svm range support for family ID 0x" << gpuNode << m_FamilyId << "." << std::endl;
        return;
    }

    unsigned long test_loops = 2;
    unsigned long BufferSize = 1UL << 27;
    HsaSVMRange SysBuffer(BufferSize, gpuNode);
    HSAuint64 *pBuf = SysBuffer.As<HSAuint64 *>();
    HsaSVMRange DataBuffer(BufferSize, gpuNode);
    HSAuint64 *pData = DataBuffer.As<HSAuint64 *>();
    SDMAQueue sdmaQueue;
    uint64_t threadId;
    struct ReadThreadParams params;

    params.pBuf = pBuf;
    params.BufferSize = BufferSize;
    params.defaultGPUNode = gpuNode;

    EXPECT_SUCCESS_GPU(sdmaQueue.Create(gpuNode), gpuNode);

    for (HSAuint64 i = 0; i < BufferSize / 8; i++)
        pBuf[i] = i;

    for (HSAuint64 i = 0; i < test_loops; i++) {
        /* 2 threads migrate to GPU */
        sdmaQueue.PlaceAndSubmitPacket(SDMACopyDataPacket(sdmaQueue.GetFamilyId(),
                    pData, pBuf, BufferSize));
        ASSERT_EQ_GPU(true, StartThread(&GpuReadThread, &params, threadId), gpuNode);
        sdmaQueue.Wait4PacketConsumption();
        WaitForThread(threadId);

        /* 2 threads migrate to cpu */
        ASSERT_EQ_GPU(true, StartThread(&CpuReadThread, &params, threadId), gpuNode);
        EXPECT_SUCCESS_GPU(SVMRangePrefetchToNode(pBuf, BufferSize, 0), gpuNode);
        WaitForThread(threadId);
    }

    EXPECT_SUCCESS_GPU(sdmaQueue.Destroy(), gpuNode);

}

TEST_P(KFDSVMRangeTest, MultiThreadMigrationTest) {
    TEST_REQUIRE_ENV_CAPABILITIES(ENVCAPS_64BITLINUX);
    TEST_START(TESTPROFILE_RUNALL);

    ASSERT_SUCCESS(KFDTest_Launch(MultiThreadMigrationTest));

    TEST_END
}

/*
 * Test SVM support file backed range
 *
 * Create temp file, mmap to alloc memory backed on file.
 * Create file backed svm range, to map to GPU for xnack on or off
 * Use sdma to write data to memory, should write to file
 * Close file, and then check if file data is updated correctly
 */
static void MigrateFileBackedRangeTest(KFDTEST_PARAMETERS* pTestParamters) {

    int gpuNode = pTestParamters->gpuNode;
    KFDSVMRangeTest* pKFDSVMRangeTest = (KFDSVMRangeTest*)pTestParamters->pTestObject;

    if (!pKFDSVMRangeTest->SVMAPISupported_GPU(gpuNode))
        return;

    unsigned int m_FamilyId = pKFDSVMRangeTest->GetFamilyIdFromNodeId(gpuNode);
    if (m_FamilyId < FAMILY_AI) {
        LOG() << std::hex << "Skipping test on gpuNode: No svm range support for family ID 0x"
            << gpuNode << m_FamilyId << "." << std::endl;
        return;
    }

    char tmpfname[] = "/tmp/kfdtest-XXXXXX";
    int fd = mkostemp(tmpfname, 0600);
    ASSERT_NE(-1, fd);

    size_t size = PAGE_SIZE;
    char *buf = reinterpret_cast<char *>(alloca(size));
    memset(buf, 0x30, size);

    ASSERT_EQ(size, write(fd, buf, size));

    void *MmapedFile = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    ASSERT_NE(MAP_FAILED, MmapedFile);

    HsaSVMRange filebackedRange(MmapedFile, size, gpuNode, gpuNode);

    SDMAQueue sdmaQueue;
    EXPECT_SUCCESS(sdmaQueue.Create(gpuNode));

    sdmaQueue.PlaceAndSubmitPacket(SDMAFillDataPacket(sdmaQueue.GetFamilyId(),
                    MmapedFile, 0x33333333, size));
    sdmaQueue.Wait4PacketConsumption();

    EXPECT_SUCCESS(sdmaQueue.Destroy());
    munmap(MmapedFile, size);
    EXPECT_SUCCESS(close(fd));

    fd = open(tmpfname, O_RDONLY);
    ASSERT_NE(-1, fd);

    ASSERT_EQ(size, read(fd, buf, size));
    EXPECT_EQ(0x33, buf[0]);

    EXPECT_SUCCESS(close(fd));
    EXPECT_SUCCESS(remove(tmpfname));
}

TEST_P(KFDSVMRangeTest, MigrateFileBackedRangeTest) {
    TEST_REQUIRE_ENV_CAPABILITIES(ENVCAPS_64BITLINUX);
    TEST_START(TESTPROFILE_RUNALL);

    ASSERT_SUCCESS(KFDTest_Launch(MigrateFileBackedRangeTest));

    TEST_END
}

/*
 * Test SVM support read only range
 *
 * Map read only range to GPU, test sdma can read the range
 * write to range should trigger GPU vm fault for both xnack on and off
 */

TEST_P(KFDSVMRangeTest, ReadOnlyRangeTest) {
    TEST_REQUIRE_ENV_CAPABILITIES(ENVCAPS_64BITLINUX);
    TEST_START(TESTPROFILE_RUNALL);

    if (!SVMAPISupported())
        return;

    int defaultGPUNode = m_NodeInfo.HsaDefaultGPUNode();
    ASSERT_GE(defaultGPUNode, 0) << "failed to get default GPU Node";

    if (!GetVramSize(defaultGPUNode)) {
        LOG() << "Skipping test: No VRAM found." << std::endl;
        return;
    }

    /*
     * Use child process to run test because the test trigger GPU vm fault, KFD evict all user queues
     * of the process and no more test can run after vm fault on the process.
     */
    int pid = fork();
    if (pid == 0) {
        TearDown();
        SetUp();
    } else {
        int childStatus;

        waitpid(pid, &childStatus, 0);
        if (hsakmt_is_dgpu()) {
            EXPECT_EQ(true, WIFEXITED(childStatus));
            EXPECT_EQ(0, WEXITSTATUS(childStatus));
        } else {
            EXPECT_EQ(true, WIFSIGNALED(childStatus));
            EXPECT_EQ(SIGSEGV, WTERMSIG(childStatus));
        }

        return;
    }

    /* Use child process to run test */
    int ret = 0;
    HsaSVMRange inBuffer(PAGE_SIZE * 2, defaultGPUNode);
    HSAuint8 *pinBuf = inBuffer.As<HSAuint8 *>();

    memset(pinBuf, 0x55, PAGE_SIZE);

    /* Map readonly pinBuf to GPU, sDMA should be able to read it */
    mprotect(pinBuf, PAGE_SIZE, PROT_READ);

    HsaSVMRange outputBuffer(PAGE_SIZE, defaultGPUNode);
    HSAuint8 *pBuf = outputBuffer.As<HSAuint8 *>();

    HsaEvent *vmFaultEvent;
    HSAuint64 faultAddress;
    HsaEventDescriptor eventDesc;
    eventDesc.EventType = HSA_EVENTTYPE_MEMORY;
    eventDesc.NodeId = defaultGPUNode;
    eventDesc.SyncVar.SyncVar.UserData = NULL;
    eventDesc.SyncVar.SyncVarSize = 0;

    ret = hsaKmtCreateEvent(&eventDesc, true, false, &vmFaultEvent);
    if (ret != HSAKMT_STATUS_SUCCESS) {
        WARN() << "Event create failed" << std::endl;
        exit(ret);
    }

    SDMAQueue sdmaQueue;

    ret = sdmaQueue.Create(defaultGPUNode);
    if (ret != HSAKMT_STATUS_SUCCESS) {
        WARN() << "Queue create failed" << std::endl;
        goto queue_fail;
    }
    sdmaQueue.PlaceAndSubmitPacket(SDMACopyDataPacket(sdmaQueue.GetFamilyId(),
                    pBuf, reinterpret_cast<void *>(pinBuf), PAGE_SIZE));
    sdmaQueue.Wait4PacketConsumption();
    EXPECT_EQ(0x55, pBuf[0]);
    if (pBuf[0] != 0x55)
        goto event_fail;

    /* sDMA write to readonly pinBuf should fail with GPU vm fault, check if pinBuf content is
     * not changed, and KFD send HSA_EVENTTYPE_MEMORY event back with fault address pinBuf.
     *
     * This must be the last step of test because all queues are evicted after vm fault.
     */

    memset(pBuf, 0xAA, PAGE_SIZE);
    sdmaQueue.PlaceAndSubmitPacket(SDMACopyDataPacket(sdmaQueue.GetFamilyId(),
                    pinBuf, reinterpret_cast<void *>(pBuf), PAGE_SIZE));

    ret = hsaKmtWaitOnEvent(vmFaultEvent, g_TestTimeOut);
    if (ret != HSAKMT_STATUS_SUCCESS) {
        WARN() << "Wait failed. No Exception triggered" << std::endl;
        goto event_fail;
    }
    if (vmFaultEvent->EventData.EventType != HSA_EVENTTYPE_MEMORY) {
        WARN() << "Unexpected Event Received " << vmFaultEvent->EventData.EventType << std::endl;
        ret = HSAKMT_STATUS_ERROR;

        goto event_fail;
    }
    faultAddress = vmFaultEvent->EventData.EventData.MemoryAccessFault.VirtualAddress;
    if (faultAddress != (HSAuint64)pinBuf) {
        WARN() << "Unexpected Fault Address " << faultAddress << std::endl;
        ret = HSAKMT_STATUS_ERROR;
    }

event_fail:
    EXPECT_SUCCESS(sdmaQueue.Destroy());
queue_fail:
    hsaKmtDestroyEvent(vmFaultEvent);
    /* Child process exit, otherwise it will continue to run remaining tests */
    exit(ret);

    TEST_END
}

/*
 * Test SMI HMM SVM profiling event
 * Use separate thread to read event the same way as ROCr and ROCProfiler
 */
struct ReadEventThreadParams {
    int nodeid;
    HSAuint64 *pBuf;
    int BufSize;
    pthread_barrier_t *barrier;
};

unsigned int ReadSMIEventThread(void* p) {
    struct ReadEventThreadParams *pArgs = (struct ReadEventThreadParams *)p;
    char msg[HSA_SMI_EVENT_MSG_SIZE];
    struct pollfd fds = {0};
    HSAuint64 events;
    int fd;

    EXPECT_SUCCESS_GPU(hsaKmtOpenSMI(pArgs->nodeid, &fd), pArgs->nodeid);

    events = HSA_SMI_EVENT_MASK_FROM_INDEX(HSA_SMI_EVENT_INDEX_MAX) - 1;
    EXPECT_EQ_GPU(write(fd, &events, sizeof(events)), sizeof(events), pArgs->nodeid);

    pthread_barrier_wait(pArgs->barrier);

    fds.fd = fd;
    fds.events = POLLIN;
    EXPECT_GE(poll(&fds, 1, 1000), 0);

    memset(msg, 0, sizeof(msg));
    EXPECT_GE_GPU(read(fd, msg, HSA_SMI_EVENT_MSG_SIZE), 0, pArgs->nodeid);

    int event_id, pid, size, trigger, unused;
    unsigned int id;
    HSAuint64 timestamp;
    HSAuint64 addr;

    sscanf(msg, "%x", &event_id);

    /* check each possible response event message format */
    if (event_id == HSA_SMI_EVENT_MIGRATE_START) {
        /* the message is HSA_SMI_EVENT_MIGRATE_START */
        EXPECT_EQ_GPU(sscanf(msg + sizeof(event_id), "%ld -%d @%lx(%d) %d->%x %x:%d %d\n", &timestamp, &pid,
                     &addr, &size, &unused, &unused, &unused, &unused, &trigger), 9, pArgs->nodeid);
        EXPECT_EQ_GPU((HSAuint64 *)(addr << PAGE_SHIFT), pArgs->pBuf, pArgs->nodeid);
        EXPECT_EQ_GPU(size << PAGE_SHIFT, pArgs->BufSize, pArgs->nodeid);
        EXPECT_EQ_GPU(pid, getpid(), pArgs->nodeid);
        EXPECT_EQ_GPU(trigger, HSA_MIGRATE_TRIGGER_PREFETCH, pArgs->nodeid);

     }else if (event_id == HSA_SMI_EVENT_QUEUE_EVICTION) {
        /* the message is HSA_SMI_EVENT_QUEUE_EVICTION */
        EXPECT_EQ_GPU(sscanf(msg + sizeof(event_id), "%ld -%d %x %d\n",  &timestamp, &pid, &id, &trigger),
                      4, pArgs->nodeid);
        EXPECT_EQ_GPU(pid, getpid(), pArgs->nodeid);
        EXPECT_EQ_GPU(trigger, HSA_QUEUE_EVICTION_TRIGGER_SVM, pArgs->nodeid);

    } else if (event_id == HSA_SMI_EVENT_QUEUE_RESTORE) {
      /* the message is HSA_SMI_EVENT_QUEUE_RESTORE */
        EXPECT_EQ_GPU(sscanf(msg + sizeof(event_id), "%ld -%d %x\n", &timestamp, &pid, &id), 3, pArgs->nodeid);
        EXPECT_EQ_GPU(pid, getpid(), pArgs->nodeid);

    } else if (event_id == HSA_SMI_EVENT_UNMAP_FROM_GPU) {
        /* the message is HSA_SMI_EVENT_UNMAP_FROM_GPU */
        EXPECT_EQ_GPU(sscanf(msg + sizeof(event_id), "%ld -%d @%lx(%d) %x %d\n", &timestamp, &pid,
                      &addr, &size, &id, &trigger), 6, pArgs->nodeid);
        /* unmap address can be from different gpus */
        EXPECT_EQ_GPU(size << PAGE_SHIFT, pArgs->BufSize, pArgs->nodeid);
        EXPECT_EQ_GPU(pid, getpid(), pArgs->nodeid);
        EXPECT_EQ_GPU(trigger, HSA_SVM_UNMAP_TRIGGER_UNMAP_FROM_CPU, pArgs->nodeid);
    } else {
        WARN() << "HMMProfilingEvent failed on gpuNode: " <<  pArgs->nodeid << std::endl;
    }

    close(fd);
    return 0;
}

static void HMMProfilingEvent(KFDTEST_PARAMETERS* pTestParamters) {

    int gpuNode = pTestParamters->gpuNode;
    KFDSVMRangeTest* pKFDSVMRangeTest = (KFDSVMRangeTest*)pTestParamters->pTestObject;

    if (!pKFDSVMRangeTest->SVMAPISupported_GPU(gpuNode))
        return;

    if (pKFDSVMRangeTest->Get_Version()->KernelInterfaceMinorVersion < 10)
        return;

    const HsaNodeProperties *pNodeProperties =
        pKFDSVMRangeTest->Get_NodeInfo()->GetNodeProperties(gpuNode);
    if (pNodeProperties->Integrated) {
        LOG() << "Skipping test on APU." << std::endl;
        return;
    }

    if (!pKFDSVMRangeTest->GetVramSize(gpuNode)) {
        LOG() << "Skipping test: No VRAM found." << std::endl;
        return;
    }

    if (pKFDSVMRangeTest->Get_NodeInfo()->IsAppAPU(gpuNode)) {
        LOG() << "Skipping test on AppAPU." << std::endl;
        return;
    }

    pthread_barrier_t barrier;
    ASSERT_SUCCESS(pthread_barrier_init(&barrier, NULL, 2));

    int BufSize = 16 << 10;
    HsaSVMRange SysBuffer(BufSize, gpuNode);
    HSAuint64 *pBuf = SysBuffer.As<HSAuint64 *>();

    struct ReadEventThreadParams pArgs = {gpuNode, pBuf, BufSize, &barrier};
    uint64_t threadId;
    ASSERT_EQ(true, StartThread(&ReadSMIEventThread, &pArgs, threadId));

    pthread_barrier_wait(&barrier);

    EXPECT_SUCCESS(SVMRangePrefetchToNode(pBuf, BufSize, gpuNode));

    WaitForThread(threadId);

}

TEST_P(KFDSVMRangeTest, HMMProfilingEvent) {
    TEST_REQUIRE_ENV_CAPABILITIES(ENVCAPS_64BITLINUX);
    TEST_START(TESTPROFILE_RUNALL);

    ASSERT_SUCCESS(KFDTest_Launch(HMMProfilingEvent));

    TEST_END
}

/*
 * Test SVM support VRAM overcommitment
 *
 * Prefetch total VRAM size plus overCommitSize SVM range to VRAM. after VRAM is full,
 * KFD should support VRAM overcommitment by evicting SVM ranges to system memory to alloc
 * VRAM for new ranges.
 */
static void VramOvercommitTest(KFDTEST_PARAMETERS* pTestParamters) {

    int gpuNode = pTestParamters->gpuNode;
    KFDSVMRangeTest* pKFDSVMRangeTest = (KFDSVMRangeTest*)pTestParamters->pTestObject;

    if (!pKFDSVMRangeTest->SVMAPISupported_GPU(gpuNode))
        return;

    unsigned int m_FamilyId = pKFDSVMRangeTest->GetFamilyIdFromNodeId(gpuNode);
    if (m_FamilyId < FAMILY_AI) {
        LOG() << std::hex << "Skipping test on gpuNode: No svm range support for family ID 0x" << gpuNode << m_FamilyId << "." << std::endl;
        return;
    }

    HSAuint64 vramSize = pKFDSVMRangeTest->GetVramSize(gpuNode);
    if (!vramSize) {
        LOG() << "Skipping test: No VRAM found." << std::endl;
        return;
    }

    unsigned long overCommitSize = 1UL << 30;

    /* With XNACK off, KFD checks that all SVM memory will fit into system memory */
	if (!g_TestGPUsNum && vramSize + overCommitSize > pKFDSVMRangeTest->GetSysMemSize() / 2) {
        LOG() << "Skipping test: Not enough system memory." << std::endl;
        return;
	} else if (g_TestGPUsNum && g_TestGPUsNum *(vramSize + overCommitSize)
			    > pKFDSVMRangeTest->GetSysMemSize() / 2) {
        LOG() << "Skipping test: Not enough system memory." << std::endl;
        return;
	}

    unsigned long BufSize = 512UL << 20;
    unsigned long numBufs = (vramSize + overCommitSize) / BufSize;
    HSAKMT_STATUS ret;

    void *pBuf[numBufs];
    unsigned long i;

    for (i = 0; i < numBufs; i++) {
        pBuf[i] = mmap(0, BufSize, PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
        ASSERT_NE(MAP_FAILED, pBuf[i]);

        ret = RegisterSVMRange(gpuNode, pBuf[i], BufSize, gpuNode, 0);
        if (ret != HSAKMT_STATUS_SUCCESS)
            break;
    }

    EXPECT_EQ_GPU(numBufs, i, gpuNode);

    while (i--)
        munmap(pBuf[i], BufSize);

}

TEST_P(KFDSVMRangeTest, VramOvercommitTest) {
    TEST_REQUIRE_ENV_CAPABILITIES(ENVCAPS_64BITLINUX);
    TEST_START(TESTPROFILE_RUNALL);

    ASSERT_SUCCESS(KFDTest_Launch(VramOvercommitTest));

    TEST_END
}

/*
 * Test SVM support VRAM overcommitment
 *
 * Prefetch giant overcommit SVM range to VRAM, KFD should support VRAM overcommitment
 * by spliting giant range into smaller ranges, evicting SVM ranges to system memory to
 * alloc VRAM for overcommitment ranges.
 */
TEST_P(KFDSVMRangeTest, VramOvercommitGiantRangeTest) {
    TEST_REQUIRE_ENV_CAPABILITIES(ENVCAPS_64BITLINUX);
    TEST_START(TESTPROFILE_RUNALL);

    if (!SVMAPISupported())
        return;

    int defaultGPUNode = m_NodeInfo.HsaDefaultGPUNode();
    ASSERT_GE(defaultGPUNode, 0) << "failed to get default GPU Node";

    if (m_FamilyId < FAMILY_AI) {
        LOG() << std::hex << "Skipping test: No svm range support for family "
                             "ID 0x" << m_FamilyId << "." << std::endl;
        return;
    }

    HSAuint64 vramSize = GetVramSize(defaultGPUNode);
    if (!vramSize) {
        LOG() << "Skipping test: No VRAM found." << std::endl;
        return;
    }

    unsigned long overCommitSize = 1UL << 30;

    /* With XNACK off, KFD checks that all SVM memory will fit into system memory */
    if (vramSize + overCommitSize > GetSysMemSize() / 2) {
        LOG() << "Skipping test: no enough system memory." << std::endl;
        return;
    }

    unsigned long BufSize = vramSize + overCommitSize;
    HSAKMT_STATUS ret;
    void *pBuf;

    pBuf = mmap(0, BufSize, PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
    ASSERT_NE(MAP_FAILED, pBuf);

    ret = RegisterSVMRange(defaultGPUNode, pBuf, BufSize, defaultGPUNode, 0);
    EXPECT_EQ (HSAKMT_STATUS_SUCCESS, ret);

    munmap(pBuf, BufSize);
    TEST_END
}

/*
 * Test partial range prefault
 *
 * mmap alloc 4 pages range, memset middle 2 pages, prefetch entire range to VRAM,
 * use sdma to memset the rest 2 pages, each page has different value 0x1, 0x2, 0x3, 0x4
 * then check if all page have the specific value after migrating 4 pages to system memory.
 */
static void PrefaultPartialRangeTest(KFDTEST_PARAMETERS* pTestParamters) {

    int gpuNode = pTestParamters->gpuNode;
    KFDSVMRangeTest* pKFDSVMRangeTest = (KFDSVMRangeTest*)pTestParamters->pTestObject;

    if (!pKFDSVMRangeTest->SVMAPISupported_GPU(gpuNode))
        return;

    unsigned int m_FamilyId = pKFDSVMRangeTest->GetFamilyIdFromNodeId(gpuNode);
    if (m_FamilyId < FAMILY_AI) {
        LOG() << std::hex << "Skipping test on gpuNode: No svm range support for family ID 0x" << gpuNode << m_FamilyId << "." << std::endl;
        return;
    }

    unsigned long BufSize = 4 * PAGE_SIZE;
    HSAKMT_STATUS ret;
    char *pBuf;

    pBuf = (char *)mmap(0, BufSize, PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
    ASSERT_NE_GPU(MAP_FAILED, pBuf, gpuNode);

    memset(pBuf + PAGE_SIZE, 0x2, PAGE_SIZE);
    memset(pBuf + 2 * PAGE_SIZE, 0x3, PAGE_SIZE);

    EXPECT_SUCCESS_GPU(RegisterSVMRange(gpuNode, pBuf, BufSize, 0, 0), gpuNode);
    EXPECT_SUCCESS_GPU(SVMRangePrefetchToNode(pBuf, BufSize, gpuNode), gpuNode);

    SDMAQueue sdmaQueue;
    EXPECT_SUCCESS_GPU(sdmaQueue.Create(gpuNode), gpuNode);

    sdmaQueue.PlaceAndSubmitPacket(SDMAFillDataPacket(sdmaQueue.GetFamilyId(),
                       pBuf, 0x01010101, PAGE_SIZE));
    sdmaQueue.PlaceAndSubmitPacket(SDMAFillDataPacket(sdmaQueue.GetFamilyId(),
                       pBuf + 3 * PAGE_SIZE, 0x04040404, PAGE_SIZE));
    sdmaQueue.Wait4PacketConsumption();

    EXPECT_SUCCESS_GPU(sdmaQueue.Destroy(), gpuNode);

    for (int i = 0; i < 4; i++)
        EXPECT_EQ_GPU(pBuf[i * PAGE_SIZE], i + 1, gpuNode);

    munmap(pBuf, BufSize);

}

TEST_P(KFDSVMRangeTest, PrefaultPartialRangeTest) {
    TEST_REQUIRE_ENV_CAPABILITIES(ENVCAPS_64BITLINUX);
    TEST_START(TESTPROFILE_RUNALL);

    ASSERT_SUCCESS(KFDTest_Launch(PrefaultPartialRangeTest));

    TEST_END
}

INSTANTIATE_TEST_CASE_P(, KFDSVMRangeTest,::testing::Values(0, 1));
